// Real cross-attention fragment controls; graph transitions use native FD views.
#include "rknn_native_graph.hpp"
#include <chrono>
#include <memory>
#include <sstream>

using Clock = std::chrono::steady_clock;
static thread_local std::string last_error;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void share(NativeGraph& target, unsigned index, NativeGraph& producer,
                  bool producer_input, unsigned source_index) {
    const auto& attr = target.inputs.at(index);
    const auto& source = (producer_input ? producer.inputs : producer.outputs).at(source_index);
    if (attr.n_dims != source.n_dims || attr.fmt != source.fmt || attr.type != source.type ||
        attr.size_with_stride != source.size_with_stride ||
        (attr.w_stride ? attr.w_stride : attr.dims[3]) != (source.w_stride ? source.w_stride : source.dims[3]) ||
        attr.h_stride != source.h_stride)
        throw std::runtime_error("Cannot alias different native layouts");
    for (unsigned dim = 0; dim < attr.n_dims; ++dim)
        if (attr.dims[dim] != source.dims[dim]) throw std::runtime_error("Cannot alias different native shapes");
    auto* buffer = (producer_input ? producer.input_memory : producer.output_memory).at(source_index);
    auto* view = rknn_create_mem_from_fd(target.context, buffer->fd, buffer->virt_addr, attr.size_with_stride, 0);
    if (!view) throw std::runtime_error("Native FD view failed");
    target.owned.push_back(view);
    target.bind(true, index, view);
}

static void pack(NativeGraph& graph, unsigned index, const float* values, unsigned channels, unsigned height, unsigned width) {
    const auto& attr = graph.inputs.at(index);
    auto* memory = graph.input_memory.at(index);
    auto* output = static_cast<__fp16*>(memory->virt_addr);
    if (attr.fmt == RKNN_TENSOR_NC1HWC2 && attr.n_dims == 5 && attr.dims[0] == 1 &&
        attr.dims[1] == (channels + 7) / 8 && attr.dims[2] == height && attr.dims[3] == width && attr.dims[4] == 8) {
        const unsigned stride = attr.w_stride ? attr.w_stride : width;
        if (attr.h_stride || attr.size_with_stride != ((channels + 7) / 8) * height * stride * 8 * sizeof(__fp16))
            throw std::runtime_error("Unexpected NC1HWC2 stride");
        for (unsigned channel = 0; channel < channels; ++channel)
            for (unsigned row = 0; row < height; ++row)
                for (unsigned column = 0; column < width; ++column)
                    output[(((channel / 8) * height + row) * stride + column) * 8 + channel % 8] =
                        values[(channel * height + row) * width + column];
    } else if ((attr.fmt == RKNN_TENSOR_NCHW || attr.fmt == RKNN_TENSOR_UNDEFINED) &&
               attr.n_elems == channels * height * width && attr.size_with_stride == channels * height * width * sizeof(__fp16)) {
        for (unsigned element = 0; element < channels * height * width; ++element) output[element] = values[element];
    } else if (attr.fmt == RKNN_TENSOR_NHWC && attr.n_dims == 4 && attr.dims[0] == 1 &&
               attr.dims[1] == height && attr.dims[2] == width && attr.dims[3] == channels &&
               attr.size_with_stride == channels * height * width * sizeof(__fp16)) {
        for (unsigned token = 0; token < height * width; ++token)
            for (unsigned channel = 0; channel < channels; ++channel)
                output[token * channels + channel] = values[channel * height * width + token];
    } else throw std::runtime_error("Unsupported native channel-major input");
    native_check(rknn_mem_sync(graph.context, memory, RKNN_MEMORY_SYNC_TO_DEVICE), "input sync");
}

class ConditioningProbe {
public:
    NativeGraph fused, compact_prepare, expanded_prepare, compact_warm, expanded_warm, grouped_warm,
                ready_prepare, ready_warm;
    bool prepared = false;

    ConditioningProbe(const std::string& directory, rknn_core_mask cores)
        : fused(directory + "/fused.rknn", cores),
          compact_prepare(directory + "/prepare_compact.rknn", cores),
          expanded_prepare(directory + "/prepare_expanded.rknn", cores),
          compact_warm(directory + "/warm_compact.rknn", cores),
          expanded_warm(directory + "/warm_expanded.rknn", cores),
          grouped_warm(directory + "/warm_grouped.rknn", cores),
          ready_prepare(directory + "/prepare_ready.rknn", cores),
          ready_warm(directory + "/warm_ready.rknn", cores) {
        if (fused.inputs.size() != 3 || compact_prepare.inputs.size() != 2 || expanded_prepare.inputs.size() != 2 ||
            compact_prepare.outputs.size() != 2 || expanded_prepare.outputs.size() != 2)
            throw std::runtime_error("Unexpected conditioning graph IO");
        for (unsigned index = 0; index < 2; ++index) {
            share(compact_prepare, index, fused, true, index + 1);
            share(expanded_prepare, index, fused, true, index + 1);
            share(ready_prepare, index, fused, true, index + 1);
            share(compact_warm, index + 1, compact_prepare, false, index);
            share(grouped_warm, index + 1, compact_prepare, false, index);
            share(expanded_warm, index + 1, expanded_prepare, false, index);
            share(ready_warm, index + 1, ready_prepare, false, index);
        }
        for (auto* graph : {&compact_warm, &expanded_warm, &grouped_warm, &ready_warm}) share(*graph, 0, fused, true, 0);
    }

    void set_prefix(const float* key, const float* value) {
        prepared = false;
        pack(fused, 1, key, 149, 5, 64);
        pack(fused, 2, value, 149, 5, 64);
        prepared = true;
    }

    void run(unsigned mode, const float* queries, unsigned steps, float* output, double* times) {
        if (!prepared || mode > 4 || steps < 1) throw std::runtime_error("Invalid fragment run");
        std::fill(times, times + 5, 0.0);
        const auto total = Clock::now();
        NativeGraph* consumer = mode == 0 ? &fused : mode == 1 ? &compact_warm :
                                mode == 2 ? &expanded_warm : mode == 3 ? &grouped_warm : &ready_warm;
        if (mode) {
            const auto start = Clock::now();
            (mode == 2 ? expanded_prepare : mode == 4 ? ready_prepare : compact_prepare).run();
            times[0] = milliseconds(start);
        }
        for (unsigned step = 0; step < steps; ++step) {
            auto start = Clock::now();
            pack(fused, 0, queries + step * 960 * 50, 50, 15, 64);
            times[1] += milliseconds(start);
            start = Clock::now();
            consumer->run();
            times[2] += milliseconds(start);
        }
        const auto start = Clock::now();
        const auto values = consumer->read_output();
        if (values.size() != 960 * 50) throw std::runtime_error("Unexpected attention output size");
        for (size_t index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index])) throw std::runtime_error("Non-finite attention output");
            output[index] = values[index];
        }
        times[3] = milliseconds(start);
        times[4] = milliseconds(total);
    }

    std::string information() {
        std::ostringstream output;
        output << "{";
        const char* names[] = {"fused", "prepare_compact", "prepare_expanded", "warm_compact", "warm_expanded", "warm_grouped", "prepare_ready", "warm_ready"};
        NativeGraph* graphs[] = {&fused, &compact_prepare, &expanded_prepare, &compact_warm, &expanded_warm, &grouped_warm, &ready_prepare, &ready_warm};
        for (unsigned index = 0; index < 8; ++index) {
            auto& graph = *graphs[index];
            rknn_mem_size memory{};
            native_check(rknn_query(graph.context, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory)), "memory info");
            output << (index ? "," : "") << "\"" << names[index] << "\":{\"weight_bytes\":" << memory.total_weight_size
                   << ",\"internal_bytes\":" << memory.total_internal_size << ",\"native_inputs\":[";
            for (unsigned tensor = 0; tensor < graph.inputs.size() + graph.outputs.size(); ++tensor) {
                if (tensor == graph.inputs.size()) output << "],\"native_outputs\":[";
                const bool input = tensor < graph.inputs.size();
                const unsigned ordinal = input ? tensor : tensor - graph.inputs.size();
                const auto& attr = (input ? graph.inputs : graph.outputs)[ordinal];
                output << (ordinal ? "," : "") << "{\"format\":\"" << get_format_string(attr.fmt)
                       << "\",\"bytes\":" << attr.size_with_stride << ",\"dims\":[";
                for (unsigned dim = 0; dim < attr.n_dims; ++dim) output << (dim ? "," : "") << attr.dims[dim];
                output << "]}";
            }
            output << "]}";
        }
        output << "}";
        return output.str();
    }

    void read_projection(bool expanded, float* key, float* value) {
        auto& graph = expanded ? expanded_prepare : compact_prepare;
        graph.run();
        auto outputs = graph.read_all_outputs();
        const unsigned size = (expanded ? 960 : 320) * 149;
        if (outputs.size() != 2 || outputs[0].size() != size || outputs[1].size() != size)
            throw std::runtime_error("Unexpected projection outputs");
        std::copy(outputs[0].begin(), outputs[0].end(), key);
        std::copy(outputs[1].begin(), outputs[1].end(), value);
    }
};

extern "C" {
const char* conditioning_error() { return last_error.c_str(); }
void* conditioning_create(const char* directory, unsigned core_mask) {
    try { last_error.clear(); return new ConditioningProbe(directory, static_cast<rknn_core_mask>(core_mask)); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void conditioning_destroy(void* handle) { delete static_cast<ConditioningProbe*>(handle); }
int conditioning_prefix(void* handle, const float* key, const float* value) {
    try { static_cast<ConditioningProbe*>(handle)->set_prefix(key, value); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int conditioning_run(void* handle, unsigned mode, const float* queries, unsigned steps, float* output, double* times) {
    try { static_cast<ConditioningProbe*>(handle)->run(mode, queries, steps, output, times); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
const char* conditioning_info(void* handle) {
    static thread_local std::string result;
    try { result = static_cast<ConditioningProbe*>(handle)->information(); return result.c_str(); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
int conditioning_projection(void* handle, unsigned expanded, float* key, float* value) {
    try { static_cast<ConditioningProbe*>(handle)->read_projection(expanded != 0, key, value); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
}
