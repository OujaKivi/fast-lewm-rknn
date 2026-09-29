// Native input lifetime controls for the unchanged denoising graph.
#include "rknn_native_graph.hpp"
#include <chrono>
#include <memory>

static thread_local std::string last_error;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class DenoiseProbe {
public:
    NativeGraph legacy, native, dirty;
    std::vector<rknn_tensor_attr> logical;
    std::vector<std::vector<float>> prefix;
    std::vector<std::vector<__fp16>> prefix_half, prefix_native;
    size_t prefix_elements = 0, suffix_elements = 0, output_elements = 0;
    bool prepared = false;

    explicit DenoiseProbe(const std::string& path)
        : legacy(path, RKNN_NPU_CORE_0_1_2, false), native(path, RKNN_NPU_CORE_0_1_2),
          dirty(path, RKNN_NPU_CORE_0_1_2, true, nullptr, RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE) {
        if (legacy.inputs.size() != 65 || legacy.outputs.size() != 1)
            throw std::runtime_error("Expected the original 32-layer denoise graph");
        for (unsigned i = 0; i < legacy.inputs.size(); ++i) {
            rknn_tensor_attr attr{}; attr.index = i;
            native_check(rknn_query(legacy.context, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "logical input");
            logical.push_back(attr);
            if (i) {
                prefix.emplace_back(attr.n_elems);
                prefix_half.emplace_back(attr.n_elems);
                prefix_native.emplace_back(native.inputs.at(i).size_with_stride / sizeof(__fp16));
                prefix_elements += attr.n_elems;
            }
            else suffix_elements = attr.n_elems;
            for (const auto* graph : {&native, &dirty}) {
                const auto& target = graph->inputs.at(i);
                if (!i) {
                    if (target.fmt != RKNN_TENSOR_UNDEFINED || target.n_dims != 3 ||
                        target.n_elems != attr.n_elems || target.size_with_stride != attr.n_elems * sizeof(__fp16))
                        throw std::runtime_error("Unsupported suffix native layout");
                } else if (attr.fmt != RKNN_TENSOR_NHWC || attr.n_dims != 4 || attr.dims[0] != 1 ||
                    target.fmt != RKNN_TENSOR_NC1HWC2 || target.n_dims != 5 || target.dims[0] != 1 ||
                    target.dims[1] != (attr.dims[3] + 7) / 8 || target.dims[2] != attr.dims[1] ||
                    target.dims[3] != attr.dims[2] || target.dims[4] != 8 ||
                    (target.w_stride && target.w_stride != attr.dims[2]) || target.h_stride ||
                    target.size_with_stride != target.n_elems * sizeof(__fp16))
                    throw std::runtime_error("Unsupported prefix native layout");
            }
        }
        rknn_tensor_attr output{};
        native_check(rknn_query(legacy.context, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)), "logical output");
        output_elements = output.n_elems;
    }

    void pack(NativeGraph& graph, unsigned index, const float* values) {
        auto* destination = static_cast<__fp16*>(graph.input_memory.at(index)->virt_addr);
        const auto& attr = logical.at(index);
        if (!index) {
            for (size_t i = 0; i < attr.n_elems; ++i) destination[i] = values[i];
        } else {
            const size_t spatial = attr.dims[1] * attr.dims[2], channels = attr.dims[3];
            // NHWC -> NC1HWC2. Padding was zeroed at allocation and is never written.
            for (size_t block = 0; block < (channels + 7) / 8; ++block)
                for (size_t pixel = 0; pixel < spatial; ++pixel)
                    for (size_t lane = 0; lane < 8 && block * 8 + lane < channels; ++lane)
                        destination[(block * spatial + pixel) * 8 + lane] = values[pixel * channels + block * 8 + lane];
        }
        native_check(rknn_mem_sync(graph.context, graph.input_memory[index], RKNN_MEMORY_SYNC_TO_DEVICE), "explicit input sync");
    }

    void prepare(const float* values, size_t count, double* times) {
        if (count != prefix_elements) throw std::runtime_error("Prefix size mismatch");
        prepared = false;
        const auto start = Clock::now();
        size_t offset = 0;
        for (auto& tensor : prefix) {
            std::copy(values + offset, values + offset + tensor.size(), tensor.begin());
            offset += tensor.size();
        }
        times[0] = elapsed(start);
        unsigned slot = 1;
        for (auto* graph : {&native, &dirty}) {
            const auto pack_start = Clock::now();
            for (unsigned i = 1; i < logical.size(); ++i) pack(*graph, i, prefix[i - 1].data());
            times[slot++] = elapsed(pack_start);
        }
        const auto half_start = Clock::now();
        for (unsigned i = 0; i < prefix.size(); ++i)
            for (size_t j = 0; j < prefix[i].size(); ++j) prefix_half[i][j] = prefix[i][j];
        times[3] = elapsed(half_start);
        const auto native_copy_start = Clock::now();
        for (unsigned i = 1; i < logical.size(); ++i)
            std::memcpy(prefix_native[i - 1].data(), native.input_memory[i]->virt_addr, native.inputs[i].size_with_stride);
        times[4] = elapsed(native_copy_start);
        prepared = true;
    }

    void run(unsigned mode, const float* suffix, size_t count, float* result, size_t result_count, double* times) {
        if (!prepared || count != suffix_elements || result_count != output_elements || mode > 5)
            throw std::runtime_error("Invalid denoise invocation");
        const auto start = Clock::now();
        NativeGraph* graph = (mode == 0 || mode >= 4) ? &legacy : mode == 3 ? &dirty : &native;
        if (mode == 0 || mode >= 4) {
            std::vector<rknn_input> inputs(logical.size());
            for (unsigned i = 0; i < inputs.size(); ++i) {
                inputs[i].index = i;
                inputs[i].buf = const_cast<float*>(i ? prefix[i - 1].data() : suffix);
                inputs[i].size = logical[i].n_elems * sizeof(float);
                inputs[i].type = RKNN_TENSOR_FLOAT32;
                inputs[i].fmt = logical[i].fmt;
                if (i && mode >= 4) {
                    auto& storage = mode == 4 ? prefix_half[i - 1] : prefix_native[i - 1];
                    inputs[i].buf = storage.data();
                    inputs[i].size = storage.size() * sizeof(__fp16);
                    inputs[i].type = RKNN_TENSOR_FLOAT16;
                    if (mode == 5) {
                        inputs[i].pass_through = 1;
                        inputs[i].fmt = native.inputs[i].fmt;
                    }
                }
            }
            native_check(rknn_inputs_set(legacy.context, inputs.size(), inputs.data()), "legacy inputs");
        } else {
            if (mode == 1)
                for (unsigned i = 1; i < logical.size(); ++i) pack(*graph, i, prefix[i - 1].data());
            pack(*graph, 0, suffix);
        }
        times[0] = elapsed(start);
        const auto run_start = Clock::now();
        graph->run();
        times[1] = elapsed(run_start);
        const auto output_start = Clock::now();
        const auto output = graph->read_output();
        if (output.size() != result_count) throw std::runtime_error("Output size mismatch");
        for (size_t i = 0; i < result_count; ++i) {
            if (!std::isfinite(output[i])) throw std::runtime_error("Non-finite output");
            result[i] = output[i];
        }
        times[2] = elapsed(output_start);
    }
};

extern "C" {
const char* denoise_error() { return last_error.c_str(); }
void* denoise_create(const char* path) {
    try { last_error.clear(); return new DenoiseProbe(path); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void denoise_destroy(void* handle) { delete static_cast<DenoiseProbe*>(handle); }
int denoise_prepare(void* handle, const float* prefix, size_t count, double* times) {
    try { static_cast<DenoiseProbe*>(handle)->prepare(prefix, count, times); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int denoise_step(void* handle, unsigned mode, const float* suffix, size_t count,
                 float* output, size_t output_count, double* times) {
    try { static_cast<DenoiseProbe*>(handle)->run(mode, suffix, count, output, output_count, times); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
}
