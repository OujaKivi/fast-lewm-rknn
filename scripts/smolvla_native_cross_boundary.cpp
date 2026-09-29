// Actual Q/attention/output boundary; all inter-graph edges use checked native views.
#include "rknn_native_graph.hpp"
#include <chrono>
#include <cstdlib>
#include <sstream>

using Clock = std::chrono::steady_clock;
static thread_local std::string last_error;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void share(NativeGraph& destination, unsigned index, NativeGraph& source,
                  bool input, unsigned ordinal) {
    const auto& target = destination.inputs.at(index);
    const auto& original = (input ? source.inputs : source.outputs).at(ordinal);
    if (target.n_dims != original.n_dims || target.fmt != original.fmt || target.type != original.type ||
        target.size_with_stride != original.size_with_stride || target.h_stride != original.h_stride ||
        (target.w_stride ? target.w_stride : target.dims[3]) != (original.w_stride ? original.w_stride : original.dims[3]))
        throw std::runtime_error("Native view layout mismatch");
    for (unsigned dim = 0; dim < target.n_dims; ++dim)
        if (target.dims[dim] != original.dims[dim]) throw std::runtime_error("Native view shape mismatch");
    auto* buffer = (input ? source.input_memory : source.output_memory).at(ordinal);
    auto* view = rknn_create_mem_from_fd(destination.context, buffer->fd, buffer->virt_addr, target.size_with_stride, 0);
    if (!view) throw std::runtime_error("Native view allocation failed");
    destination.owned.push_back(view);
    destination.bind(true, index, view);
}

static void pack(NativeGraph& graph, unsigned index, const float* values,
                 unsigned channels, unsigned height, unsigned width) {
    const auto& attr = graph.inputs.at(index);
    auto* buffer = graph.input_memory.at(index);
    auto* result = static_cast<__fp16*>(buffer->virt_addr);
    if (attr.fmt == RKNN_TENSOR_NC1HWC2 && attr.n_dims == 5 && attr.dims[0] == 1 &&
        attr.dims[1] == (channels + 7) / 8 && attr.dims[2] == height && attr.dims[3] == width && attr.dims[4] == 8) {
        const unsigned stride = attr.w_stride ? attr.w_stride : width;
        if (attr.h_stride || attr.size_with_stride != ((channels + 7) / 8) * height * stride * 8 * sizeof(__fp16))
            throw std::runtime_error("Unexpected native input stride");
        for (unsigned channel = 0; channel < channels; ++channel)
            for (unsigned row = 0; row < height; ++row)
                for (unsigned column = 0; column < width; ++column)
                    result[(((channel / 8) * height + row) * stride + column) * 8 + channel % 8] =
                        values[(channel * height + row) * width + column];
    } else if ((attr.fmt == RKNN_TENSOR_UNDEFINED || attr.fmt == RKNN_TENSOR_NCHW) &&
               attr.n_elems == channels * height * width && attr.size_with_stride == channels * height * width * sizeof(__fp16)) {
        for (unsigned element = 0; element < channels * height * width; ++element) result[element] = values[element];
    } else if (attr.fmt == RKNN_TENSOR_NHWC && attr.n_dims == 4 && attr.dims[0] == 1 &&
               attr.dims[1] == height && attr.dims[2] == width && attr.dims[3] == channels &&
               attr.size_with_stride == channels * height * width * sizeof(__fp16)) {
        for (unsigned token = 0; token < height * width; ++token)
            for (unsigned channel = 0; channel < channels; ++channel)
                result[token * channels + channel] = values[channel * height * width + token];
    } else throw std::runtime_error("Unsupported native input layout");
    native_check(rknn_mem_sync(graph.context, buffer, RKNN_MEMORY_SYNC_TO_DEVICE), "input sync");
}

class CrossBoundary {
public:
    NativeGraph original, ready, compact, grouped, producer, attention, consumer, prep_compact, prep_ready,
                interleaved, propagated, group_prepare, group_ready, grouped_ready_control;
    bool prefix_ready = false;

    CrossBoundary(const std::string& directory, rknn_core_mask cores)
        : original(directory + "/block_original.rknn", cores),
          ready(directory + (std::getenv("SMOLVLA_CROSS_OPT3") ? "/block_ready_opt3.rknn" : "/block_ready.rknn"), cores),
          compact(directory + "/block_compact_repeat.rknn", cores),
          grouped(directory + "/block_grouped_joint.rknn", cores),
          producer(directory + "/query_producer.rknn", cores),
          attention(directory + "/grouped_attention.rknn", cores),
          consumer(directory + "/output_consumer.rknn", cores),
          prep_compact(directory + "/prepare_compact.rknn", cores),
          prep_ready(directory + "/prepare_ready.rknn", cores),
          interleaved(directory + "/block_grouped_interleaved.rknn", cores),
          propagated(directory + "/block_grouped_propagated.rknn", cores),
          group_prepare(directory + "/prepare_group_ready.rknn", cores),
          group_ready(directory + "/block_grouped_propagated_ready.rknn", cores),
          grouped_ready_control(directory + (std::getenv("SMOLVLA_CROSS_OPT3") ? "/block_grouped_joint_ready_opt3.rknn" : "/block_grouped_joint_ready.rknn"), cores) {
        if (original.inputs.size() != 3 || original.outputs.size() != 1 ||
            producer.inputs.size() != 1 || consumer.inputs.size() != 2)
            throw std::runtime_error("Unexpected block IO");
        for (auto* graph : {&ready, &compact, &grouped, &producer, &consumer, &interleaved, &propagated, &group_ready, &grouped_ready_control}) share(*graph, 0, original, true, 0);
        for (unsigned index = 0; index < 2; ++index) {
            share(prep_compact, index, original, true, index + 1);
            share(prep_ready, index, original, true, index + 1);
            share(group_prepare, index, original, true, index + 1);
            share(ready, index + 1, prep_ready, false, index);
            share(group_ready, index + 1, group_prepare, false, index);
            share(grouped_ready_control, index + 1, group_prepare, false, index);
            for (auto* graph : {&compact, &grouped, &attention, &interleaved, &propagated}) share(*graph, index + 1, prep_compact, false, index);
        }
        share(attention, 0, producer, false, 0);
        share(consumer, 1, attention, false, 0);
    }

    void set_prefix(const float* key, const float* value) {
        prefix_ready = false;
        pack(original, 1, key, 149, 5, 64);
        pack(original, 2, value, 149, 5, 64);
        prefix_ready = true;
    }

    void run(unsigned mode, const float* hidden, unsigned steps, float* output, double* times) {
        if (!prefix_ready || mode > 8 || !steps) throw std::runtime_error("Invalid block run");
        std::fill(times, times + 8, 0.0);
        const auto start = Clock::now();
        if (mode) {
            const auto tick = Clock::now();
            (mode == 1 ? prep_ready : mode >= 7 ? group_prepare : prep_compact).run();
            times[0] = elapsed(tick);
        }
        NativeGraph* graph = mode == 0 ? &original : mode == 1 ? &ready : mode == 2 ? &compact :
                             mode == 5 ? &interleaved : mode == 6 ? &propagated : mode == 7 ? &group_ready :
                             mode == 8 ? &grouped_ready_control : &grouped;
        for (unsigned step = 0; step < steps; ++step) {
            auto tick = Clock::now();
            pack(original, 0, hidden + step * 50 * 480, 50, 1, 480);
            times[1] += elapsed(tick);
            if (mode == 4) {
                tick = Clock::now(); producer.run(); times[3] += elapsed(tick);
                tick = Clock::now(); attention.run(); times[4] += elapsed(tick);
                tick = Clock::now(); consumer.run(); times[5] += elapsed(tick);
            } else {
                tick = Clock::now(); graph->run(); times[2] += elapsed(tick);
            }
        }
        const auto tick = Clock::now();
        const auto values = (mode == 4 ? consumer : *graph).read_output();
        if (values.size() != 50 * 480) throw std::runtime_error("Unexpected hidden output");
        for (unsigned index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index])) throw std::runtime_error("Non-finite hidden output");
            output[index] = values[index];
        }
        times[6] = elapsed(tick);
        times[7] = elapsed(start);
    }

    std::string information() {
        std::ostringstream result;
        result << "{";
        const char* names[] = {"original", "ready", "compact", "grouped", "producer", "attention", "consumer", "prep_compact", "prep_ready", "interleaved", "propagated", "group_prepare", "group_ready", "grouped_ready_control"};
        NativeGraph* graphs[] = {&original, &ready, &compact, &grouped, &producer, &attention, &consumer, &prep_compact, &prep_ready, &interleaved, &propagated, &group_prepare, &group_ready, &grouped_ready_control};
        for (unsigned ordinal = 0; ordinal < 14; ++ordinal) {
            auto& graph = *graphs[ordinal];
            rknn_mem_size size{};
            native_check(rknn_query(graph.context, RKNN_QUERY_MEM_SIZE, &size, sizeof(size)), "memory query");
            result << (ordinal ? "," : "") << "\"" << names[ordinal] << "\":{\"weight_bytes\":" << size.total_weight_size
                   << ",\"internal_bytes\":" << size.total_internal_size << ",\"native_inputs\":[";
            for (unsigned item = 0; item < graph.inputs.size() + graph.outputs.size(); ++item) {
                if (item == graph.inputs.size()) result << "],\"native_outputs\":[";
                const bool input = item < graph.inputs.size();
                const unsigned index = input ? item : item - graph.inputs.size();
                const auto& attr = (input ? graph.inputs : graph.outputs)[index];
                result << (index ? "," : "") << "{\"bytes\":" << attr.size_with_stride
                       << ",\"format\":\"" << get_format_string(attr.fmt) << "\",\"dims\":[";
                for (unsigned dim = 0; dim < attr.n_dims; ++dim) result << (dim ? "," : "") << attr.dims[dim];
                result << "]}";
            }
            result << "]}";
        }
        result << "}";
        return result.str();
    }
};

extern "C" {
const char* cross_boundary_error() { return last_error.c_str(); }
void* cross_boundary_create(const char* directory, unsigned core_mask) {
    try { last_error.clear(); return new CrossBoundary(directory, static_cast<rknn_core_mask>(core_mask)); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void cross_boundary_destroy(void* handle) { delete static_cast<CrossBoundary*>(handle); }
int cross_boundary_prefix(void* handle, const float* key, const float* value) {
    try { static_cast<CrossBoundary*>(handle)->set_prefix(key, value); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int cross_boundary_run(void* handle, unsigned mode, const float* hidden, unsigned steps, float* output, double* times) {
    try { static_cast<CrossBoundary*>(handle)->run(mode, hidden, steps, output, times); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
const char* cross_boundary_info(void* handle) {
    static thread_local std::string value;
    try { value = static_cast<CrossBoundary*>(handle)->information(); return value.c_str(); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
}
