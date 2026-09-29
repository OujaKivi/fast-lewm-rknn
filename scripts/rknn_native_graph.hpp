#pragma once
#include <rknn_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

inline void native_check(int code, const char* operation) {
    if (code < 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}

class NativeGraph {
public:
    rknn_context context = 0;
    std::vector<rknn_tensor_attr> inputs, outputs;
    std::vector<rknn_tensor_mem*> input_memory, output_memory, owned;

    NativeGraph(const std::string& path, rknn_core_mask cores, bool allocate_io = true, NativeGraph* source = nullptr,
                uint32_t init_flags = 0) {
        try {
            if (source && init_flags) throw std::runtime_error("Duplicate context inherits source flags");
            if (source) native_check(rknn_dup_context(&source->context, &context), "duplicate context");
            else native_check(rknn_init(&context, const_cast<char*>(path.c_str()), 0, init_flags, nullptr), "init");
            native_check(rknn_set_core_mask(context, cores), "core mask");
            rknn_input_output_num count{};
            native_check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "io count");
            for (unsigned index = 0; index < count.n_input + count.n_output; ++index) {
                const bool input = index < count.n_input;
                rknn_tensor_attr attr{};
                attr.index = input ? index : index - count.n_input;
                native_check(rknn_query(context, input ? RKNN_QUERY_NATIVE_INPUT_ATTR : RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                                       &attr, sizeof(attr)), "native attr");
                if (attr.type != RKNN_TENSOR_FLOAT16) throw std::runtime_error("Expected native FP16");
                attr.pass_through = 1;
                (input ? inputs : outputs).push_back(attr);
                (input ? input_memory : output_memory).push_back(nullptr);
                if (allocate_io) bind(input, attr.index, allocate(attr));
            }
        } catch (...) { release(); throw; }
    }
    NativeGraph(const NativeGraph&) = delete;
    NativeGraph& operator=(const NativeGraph&) = delete;
    ~NativeGraph() { release(); }
    void release() {
        for (auto* memory : owned) rknn_destroy_mem(context, memory);
        owned.clear();
        if (context) rknn_destroy(context);
        context = 0;
    }
    rknn_tensor_mem* allocate(const rknn_tensor_attr& attr) {
        auto* memory = rknn_create_mem(context, attr.size_with_stride);
        if (!memory) throw std::runtime_error("Native allocation failed");
        owned.push_back(memory);
        std::memset(memory->virt_addr, 0, memory->size);
        native_check(rknn_mem_sync(context, memory, RKNN_MEMORY_SYNC_TO_DEVICE), "initial sync");
        return memory;
    }
    void bind(bool input, unsigned index, rknn_tensor_mem* memory) {
        auto attr = (input ? inputs : outputs).at(index);
        if (memory->size < attr.size_with_stride) throw std::runtime_error("Native buffer too small");
        native_check(rknn_set_io_mem(context, memory, &attr), "native binding");
        (input ? input_memory : output_memory)[index] = memory;
    }
    void alias(bool input, unsigned index, rknn_tensor_mem* source,
               const rknn_tensor_attr& source_attr, unsigned partitions = 1, unsigned part = 0) {
        const auto attr = (input ? inputs : outputs).at(index);
        if (!source || partitions < 1 || part >= partitions || attr.fmt != RKNN_TENSOR_NC1HWC2 ||
            source_attr.fmt != attr.fmt || attr.n_dims != 5 || source_attr.n_dims != 5 ||
            attr.type != source_attr.type || source_attr.dims[0] != 1)
            throw std::runtime_error("Expected compatible B1 native channel views");
        for (unsigned dim = 0; dim < 5; ++dim)
            if (attr.dims[dim] * (dim == 1 ? partitions : 1) != source_attr.dims[dim])
                throw std::runtime_error("Native view dimensions mismatch");
        const unsigned width_stride = attr.w_stride ? attr.w_stride : attr.dims[3];
        const unsigned source_width_stride = source_attr.w_stride ? source_attr.w_stride : source_attr.dims[3];
        if (attr.size_with_stride * partitions != source_attr.size_with_stride ||
            width_stride != source_width_stride || source->size < source_attr.size_with_stride)
            throw std::runtime_error("Native view strides mismatch");
        auto* memory = rknn_create_mem_from_fd(context, source->fd, source->virt_addr,
                                              attr.size_with_stride, part * attr.size_with_stride);
        if (!memory) throw std::runtime_error("FD view failed");
        owned.push_back(memory);
        bind(input, index, memory);
    }
    void run() { native_check(rknn_run(context, nullptr), "run"); }

    std::vector<std::vector<float>> read_all_outputs() {
        std::vector<rknn_output> requests(outputs.size());
        std::vector<rknn_tensor_attr> attributes(outputs.size());
        for (unsigned index = 0; index < outputs.size(); ++index) {
            requests[index].index = index;
            requests[index].want_float = 1;
            attributes[index].index = index;
            native_check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &attributes[index], sizeof(attributes[index])), "output attr");
        }
        native_check(rknn_outputs_get(context, requests.size(), requests.data(), nullptr), "read all outputs");
        std::vector<std::vector<float>> result;
        for (unsigned index = 0; index < requests.size(); ++index) {
            const auto* values = static_cast<float*>(requests[index].buf);
            result.emplace_back(values, values + attributes[index].n_elems);
        }
        native_check(rknn_outputs_release(context, requests.size(), requests.data()), "release all outputs");
        return result;
    }

    void set_image(const std::vector<float>& image) {
        const auto& attr = inputs.at(0);
        if (image.size() != 3 * 512 * 512) throw std::runtime_error("Invalid image fixture size");
        auto* destination = static_cast<__fp16*>(input_memory[0]->virt_addr);
        if ((attr.fmt == RKNN_TENSOR_NCHW || attr.fmt == RKNN_TENSOR_UNDEFINED) &&
            attr.n_dims == 4 && attr.dims[0] == 1 && attr.dims[1] == 3 &&
            attr.dims[2] == 512 && attr.dims[3] == 512 && attr.size_with_stride == image.size() * sizeof(__fp16)) {
            for (size_t index = 0; index < image.size(); ++index) destination[index] = image[index];
        } else if (attr.fmt == RKNN_TENSOR_NHWC && attr.n_dims == 4 && attr.dims[0] == 1 &&
                   attr.dims[1] == 512 && attr.dims[2] == 512 && attr.dims[3] == 3 &&
                   attr.size_with_stride == image.size() * sizeof(__fp16)) {
            for (size_t pixel = 0; pixel < 512 * 512; ++pixel)
                for (size_t channel = 0; channel < 3; ++channel)
                    destination[pixel * 3 + channel] = image[channel * 512 * 512 + pixel];
        } else if (attr.fmt == RKNN_TENSOR_NC1HWC2 && attr.n_dims == 5 &&
                   attr.dims[0] == 1 && attr.dims[1] == 1 && attr.dims[2] == 512 && attr.dims[3] == 512 &&
                   attr.dims[4] == 8 && attr.size_with_stride == 512 * 512 * 8 * sizeof(__fp16)) {
            for (size_t pixel = 0; pixel < 512 * 512; ++pixel)
                for (size_t channel = 0; channel < 3; ++channel)
                    destination[pixel * 8 + channel] = image[channel * 512 * 512 + pixel];
        } else throw std::runtime_error("Unsupported native image layout: " + std::string(get_format_string(attr.fmt)) +
                                        " bytes=" + std::to_string(attr.size_with_stride));
        native_check(rknn_mem_sync(context, input_memory[0], RKNN_MEMORY_SYNC_TO_DEVICE), "image sync");
    }
    std::vector<float> read_output(bool hidden_rows = false) {
        rknn_tensor_attr attr{};
        native_check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "output attr");
        rknn_output output{};
        output.want_float = 1;
        native_check(rknn_outputs_get(context, 1, &output, nullptr), "read output");
        const auto* source = static_cast<float*>(output.buf);
        std::vector<float> result(source, source + attr.n_elems);
        native_check(rknn_outputs_release(context, 1, &output), "release output");
        if (hidden_rows) {
            if (attr.fmt != RKNN_TENSOR_NCHW || attr.n_dims != 4 || attr.dims[0] != 1 ||
                attr.dims[1] != 768 || attr.dims[2] != 1 || attr.dims[3] != 1024)
                throw std::runtime_error("Expected logical channel-major hidden output");
            std::vector<float> rows(result.size());
            for (size_t token = 0; token < 1024; ++token)
                for (size_t channel = 0; channel < 768; ++channel)
                    rows[token * 768 + channel] = result[channel * 1024 + token];
            return rows;
        }
        return result;
    }
};
