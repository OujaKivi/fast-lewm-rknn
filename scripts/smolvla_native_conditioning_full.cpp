// Strong original-graph controls versus ordinary invariant K/V hoisting.
#include "rknn_native_graph.hpp"
#include <chrono>
#include <sstream>

using Clock = std::chrono::steady_clock;
static thread_local std::string last_error;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void share_input(NativeGraph& target, unsigned index, NativeGraph& producer,
                        bool from_input, unsigned source_index) {
    const auto& attr = target.inputs.at(index);
    const auto& source = (from_input ? producer.inputs : producer.outputs).at(source_index);
    if (attr.n_dims != source.n_dims || attr.fmt != source.fmt || attr.type != source.type ||
        attr.size_with_stride != source.size_with_stride || attr.h_stride != source.h_stride ||
        (attr.w_stride ? attr.w_stride : attr.dims[3]) != (source.w_stride ? source.w_stride : source.dims[3]))
        throw std::runtime_error("Incompatible native continuation layout");
    for (unsigned dim = 0; dim < attr.n_dims; ++dim)
        if (attr.dims[dim] != source.dims[dim]) throw std::runtime_error("Incompatible native continuation dimensions");
    auto* source_memory = (from_input ? producer.input_memory : producer.output_memory).at(source_index);
    auto* view = rknn_create_mem_from_fd(target.context, source_memory->fd, source_memory->virt_addr, attr.size_with_stride, 0);
    if (!view) throw std::runtime_error("Cannot create native continuation view");
    target.owned.push_back(view);
    target.bind(true, index, view);
}

class FullConditioning {
public:
    NativeGraph original, ordinary, preparation, warm;
    bool prefix_set = false, cache_ready = false;

    FullConditioning(const std::string& original_path, const std::string& directory)
        : original(original_path, RKNN_NPU_CORE_0_1_2), ordinary(original_path, RKNN_NPU_CORE_0_1_2, false),
          preparation(directory + "/prepare_full.rknn", RKNN_NPU_CORE_0_1_2, false),
          warm(directory + "/warm_full.rknn", RKNN_NPU_CORE_0_1_2, false) {
        if (original.inputs.size() != 65 || warm.inputs.size() != 65 || preparation.inputs.size() != 32 ||
            preparation.outputs.size() != 32 || original.outputs.size() != 1 || warm.outputs.size() != 1)
            throw std::runtime_error("Unexpected full conditioning IO");
        for (auto* graph : {&preparation, &warm})
            for (unsigned index = 0; index < graph->outputs.size(); ++index)
                graph->bind(false, index, graph->allocate(graph->outputs[index]));
        share_input(warm, 0, original, true, 0);
        for (unsigned layer = 0; layer < 32; ++layer) {
            for (unsigned kind = 0; kind < 2; ++kind) {
                const unsigned slot = 1 + 2 * layer + kind;
                if (layer % 2 == 0) share_input(warm, slot, original, true, slot);
                else {
                    const unsigned prepared = (layer / 2) * 2 + kind;
                    share_input(preparation, prepared, original, true, slot);
                    share_input(warm, slot, preparation, false, prepared);
                }
            }
        }
    }

    void prefix(const float* values, size_t count) {
        if (count != 64 * 149 * 5 * 64) throw std::runtime_error("Invalid canonical prefix fixture size");
        prefix_set = false;
        cache_ready = false;
        for (unsigned index = 1; index < original.inputs.size(); ++index) {
            const auto& attr = original.inputs[index];
            if (attr.fmt != RKNN_TENSOR_NC1HWC2 || attr.n_dims != 5 || attr.dims[0] != 1 ||
                attr.dims[1] != 19 || attr.dims[2] != 5 || attr.dims[3] != 64 || attr.dims[4] != 8 ||
                attr.size_with_stride != 19 * 5 * 64 * 8 * sizeof(__fp16))
                throw std::runtime_error("Unexpected original prefix native layout");
            auto* memory = original.input_memory[index];
            auto* output = static_cast<__fp16*>(memory->virt_addr);
            const auto* input = values + (index - 1) * 149 * 5 * 64;
            for (unsigned token = 0; token < 149; ++token)
                for (unsigned pixel = 0; pixel < 5 * 64; ++pixel)
                    output[((token / 8) * 5 * 64 + pixel) * 8 + token % 8] = input[token * 5 * 64 + pixel];
            native_check(rknn_mem_sync(original.context, memory, RKNN_MEMORY_SYNC_TO_DEVICE), "prefix sync");
        }
        prefix_set = true;
    }

    void begin() {
        if (!prefix_set) throw std::runtime_error("Prefix has not been set");
        cache_ready = false;
    }

    void step(unsigned mode, const float* suffix, size_t count, float* result, size_t output_count, double* times) {
        if (!prefix_set || mode > 2 || count != 50 * 480 || output_count != 50 * 32)
            throw std::runtime_error("Invalid full denoising invocation");
        std::fill(times, times + 4, 0.0);
        if (mode == 2 && !cache_ready) {
            const auto start = Clock::now();
            preparation.run();
            times[0] = milliseconds(start);
            cache_ready = true;
        }
        auto start = Clock::now();
        const auto& attr = original.inputs[0];
        if (attr.fmt != RKNN_TENSOR_UNDEFINED || attr.n_elems != count || attr.size_with_stride != count * sizeof(__fp16))
            throw std::runtime_error("Unexpected suffix native layout");
        auto* memory = original.input_memory[0];
        auto* destination = static_cast<__fp16*>(memory->virt_addr);
        for (size_t index = 0; index < count; ++index) destination[index] = suffix[index];
        native_check(rknn_mem_sync(original.context, memory, RKNN_MEMORY_SYNC_TO_DEVICE), "suffix sync");
        if (mode == 1) {
            std::vector<rknn_input> inputs(original.inputs.size());
            for (unsigned index = 0; index < inputs.size(); ++index) {
                inputs[index].index = index;
                inputs[index].buf = original.input_memory[index]->virt_addr;
                inputs[index].size = original.inputs[index].size_with_stride;
                inputs[index].type = RKNN_TENSOR_FLOAT16;
                inputs[index].fmt = original.inputs[index].fmt;
                inputs[index].pass_through = 1;
            }
            native_check(rknn_inputs_set(ordinary.context, inputs.size(), inputs.data()), "ordinary native inputs");
        }
        times[1] = milliseconds(start);
        auto* graph = mode == 0 ? &original : mode == 1 ? &ordinary : &warm;
        start = Clock::now();
        graph->run();
        times[2] = milliseconds(start);
        start = Clock::now();
        const auto values = graph->read_output();
        if (values.size() != output_count) throw std::runtime_error("Unexpected velocity size");
        for (size_t index = 0; index < output_count; ++index) {
            if (!std::isfinite(values[index])) throw std::runtime_error("Non-finite velocity");
            result[index] = values[index];
        }
        times[3] = milliseconds(start);
    }

    std::string information() {
        std::ostringstream output;
        const char* names[] = {"original_resident", "original_ordinary", "preparation", "warm"};
        NativeGraph* graphs[] = {&original, &ordinary, &preparation, &warm};
        output << "{";
        for (unsigned index = 0; index < 4; ++index) {
            auto* graph = graphs[index];
            rknn_mem_size memory{};
            native_check(rknn_query(graph->context, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory)), "memory info");
            unsigned input_bytes = 0, output_bytes = 0;
            for (const auto& attr : graph->inputs) input_bytes += attr.size_with_stride;
            for (const auto& attr : graph->outputs) output_bytes += attr.size_with_stride;
            output << (index ? "," : "") << "\"" << names[index] << "\":{\"weight_bytes\":" << memory.total_weight_size
                   << ",\"internal_bytes\":" << memory.total_internal_size << ",\"native_input_bytes\":" << input_bytes
                   << ",\"native_output_bytes\":" << output_bytes << "}";
        }
        output << "}";
        return output.str();
    }
};

extern "C" {
const char* full_conditioning_error() { return last_error.c_str(); }
void* full_conditioning_create(const char* original, const char* directory) {
    try { last_error.clear(); return new FullConditioning(original, directory); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void full_conditioning_destroy(void* handle) { delete static_cast<FullConditioning*>(handle); }
int full_conditioning_prefix(void* handle, const float* prefix, size_t count) {
    try { static_cast<FullConditioning*>(handle)->prefix(prefix, count); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int full_conditioning_begin(void* handle) {
    try { static_cast<FullConditioning*>(handle)->begin(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int full_conditioning_step(void* handle, unsigned mode, const float* suffix, size_t count,
                           float* output, size_t output_count, double* times) {
    try { static_cast<FullConditioning*>(handle)->step(mode, suffix, count, output, output_count, times); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
const char* full_conditioning_info(void* handle) {
    static thread_local std::string result;
    try { result = static_cast<FullConditioning*>(handle)->information(); return result.c_str(); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
}
