// Actual vision block: packed producer -> independent attention heads -> packed consumer.
#include <rknn_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int code, const char* operation) {
    if (code < 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}

class Graph {
public:
    rknn_context context = 0;
    std::vector<rknn_tensor_attr> inputs, outputs;
    std::vector<rknn_tensor_mem*> input_memory, output_memory, owned;

    Graph(const std::string& path, rknn_core_mask cores) {
        try {
            check(rknn_init(&context, const_cast<char*>(path.c_str()), 0, 0, nullptr), "init");
            check(rknn_set_core_mask(context, cores), "core mask");
            rknn_input_output_num count{};
            check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "io count");
            for (unsigned index = 0; index < count.n_input + count.n_output; ++index) {
                const bool input = index < count.n_input;
                rknn_tensor_attr attr{};
                attr.index = input ? index : index - count.n_input;
                check(rknn_query(context, input ? RKNN_QUERY_NATIVE_INPUT_ATTR : RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                                 &attr, sizeof(attr)), "native attr");
                if (attr.type != RKNN_TENSOR_FLOAT16) throw std::runtime_error("Expected native FP16");
                auto* memory = rknn_create_mem(context, attr.size_with_stride);
                if (!memory) throw std::runtime_error("Allocation failed");
                owned.push_back(memory);
                std::memset(memory->virt_addr, 0, memory->size);
                check(rknn_mem_sync(context, memory, RKNN_MEMORY_SYNC_TO_DEVICE), "initial sync");
                attr.pass_through = 1;
                check(rknn_set_io_mem(context, memory, &attr), "native binding");
                (input ? inputs : outputs).push_back(attr);
                (input ? input_memory : output_memory).push_back(memory);
                std::cout << path << (input ? " input " : " output ") << attr.index
                          << " " << get_format_string(attr.fmt) << " bytes=" << attr.size_with_stride << " dims=";
                for (unsigned dim = 0; dim < attr.n_dims; ++dim) std::cout << attr.dims[dim] << ',';
                std::cout << '\n';
            }
        } catch (...) { release(); throw; }
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() { release(); }
    void release() {
        for (auto* memory : owned) rknn_destroy_mem(context, memory);
        owned.clear();
        if (context) rknn_destroy(context);
        context = 0;
    }
    void run() { check(rknn_run(context, nullptr), "run"); }

    void alias(bool input, unsigned index, rknn_tensor_mem* source,
               const rknn_tensor_attr& source_attr, unsigned partitions = 1, unsigned part = 0) {
        auto attr = (input ? inputs : outputs).at(index);
        if (attr.fmt != RKNN_TENSOR_NC1HWC2 || source_attr.fmt != RKNN_TENSOR_NC1HWC2 ||
            attr.n_dims != 5 || source_attr.n_dims != 5 || attr.type != source_attr.type)
            throw std::runtime_error("Shared interface must be compatible NC1HWC2 FP16");
        for (unsigned dim = 0; dim < attr.n_dims; ++dim) {
            const unsigned factor = dim == 1 ? partitions : 1;
            if (attr.dims[dim] * factor != source_attr.dims[dim])
                throw std::runtime_error("Shared native dimensions mismatch");
        }
        if (source_attr.dims[0] != 1 || attr.size_with_stride * partitions != source_attr.size_with_stride || part >= partitions)
            throw std::runtime_error("Unsupported shared layout/stride/partition");
        const unsigned offset = part * attr.size_with_stride;
        auto* memory = rknn_create_mem_from_fd(context, source->fd, source->virt_addr, attr.size_with_stride, offset);
        if (!memory) throw std::runtime_error("FD view failed");
        owned.push_back(memory);
        check(rknn_set_io_mem(context, memory, &attr), "shared binding");
        (input ? input_memory : output_memory)[index] = memory;
    }

    void set_hidden(const std::vector<float>& values) {
        const auto& attr = inputs.at(0);
        if (attr.size_with_stride != values.size() * sizeof(__fp16))
            throw std::runtime_error("Expected unpadded FP16 hidden input");
        auto* destination = static_cast<__fp16*>(input_memory[0]->virt_addr);
        if (attr.fmt == RKNN_TENSOR_UNDEFINED && attr.n_dims == 3) {
            for (size_t index = 0; index < values.size(); ++index) destination[index] = values[index];
        } else if (attr.fmt == RKNN_TENSOR_NC1HWC2 && attr.n_dims == 5 &&
                   attr.dims[0] == 1 && attr.dims[1] == 96 && attr.dims[2] == 1 &&
                   attr.dims[3] == 1024 && attr.dims[4] == 8) {
            for (size_t token = 0; token < 1024; ++token)
                for (size_t channel = 0; channel < 768; ++channel)
                    destination[((channel / 8) * 1024 + token) * 8 + channel % 8] = values[token * 768 + channel];
        } else throw std::runtime_error("Unsupported native hidden layout");
        check(rknn_mem_sync(context, input_memory[0], RKNN_MEMORY_SYNC_TO_DEVICE), "hidden sync");
    }

    std::vector<float> read_output() {
        rknn_tensor_attr attr{};
        check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "output attr");
        rknn_output output{};
        output.want_float = 1;
        check(rknn_outputs_get(context, 1, &output, nullptr), "read output");
        auto* values = static_cast<float*>(output.buf);
        std::vector<float> result(values, values + attr.n_elems);
        check(rknn_outputs_release(context, 1, &output), "release output");
        return result;
    }
};

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: native_block MODEL_DIRECTORY REPEATS OUTPUT_JSON [PARTITIONS=2]\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + "/";
        const int repeats = std::stoi(argv[2]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        const unsigned partitions = argc == 5 ? std::stoul(argv[4]) : 2;
        if (partitions != 2 && partitions != 3) throw std::runtime_error("Use two or three partitions");
        Graph baseline(directory + "block_original_fp16.rknn", RKNN_NPU_CORE_0_1_2);
        Graph packed_baseline(directory + "block_unmasked_native_fp16.rknn", RKNN_NPU_CORE_0_1_2);
        Graph producer(directory + "qkv_native_fp16.rknn", RKNN_NPU_CORE_0_1_2);
        Graph attention(directory + "attention_native_full_fp16.rknn", RKNN_NPU_CORE_0_1_2);
        std::vector<std::unique_ptr<Graph>> parts;
        for (unsigned part = 0; part < partitions; ++part)
            parts.emplace_back(new Graph(directory + "attention_native_h" + std::to_string(12 / partitions) + "_fp16.rknn",
                                         static_cast<rknn_core_mask>(1u << part)));
        Graph consumer(directory + "suffix_native_fp16.rknn", RKNN_NPU_CORE_0_1_2);
        if (producer.outputs.size() != 3 || attention.inputs.size() != 3 || consumer.inputs.size() != 2)
            throw std::runtime_error("Unexpected split graph interface");
        for (unsigned index = 0; index < 3; ++index) {
            attention.alias(true, index, producer.output_memory[index], producer.outputs[index]);
            for (unsigned part = 0; part < partitions; ++part)
                parts[part]->alias(true, index, producer.output_memory[index], producer.outputs[index], partitions, part);
        }
        for (unsigned part = 0; part < partitions; ++part)
            parts[part]->alias(false, 0, attention.output_memory[0], attention.outputs[0], partitions, part);
        consumer.alias(true, 0, producer.input_memory[0], producer.inputs[0]);
        consumer.alias(true, 1, attention.output_memory[0], attention.outputs[0]);

        auto run = [&](int plan) {
            if (plan == 0) { baseline.run(); return; }
            if (plan == 1) { packed_baseline.run(); return; }
            producer.run();
            if (plan == 2) attention.run();
            else {
                std::vector<std::future<void>> pending;
                for (unsigned part = 0; part < partitions; ++part)
                    pending.emplace_back(std::async(std::launch::async, [&, part] { parts[part]->run(); }));
                for (auto& task : pending) task.get();
            }
            consumer.run();
        };
        std::mt19937 generator(0);
        std::uniform_real_distribution<float> random(-1.0f, 1.0f);
        std::vector<float> hidden(1024 * 768);
        std::vector<double> errors(4, 0), mean_errors(4, 0);
        double partition_error = 0;
        unsigned trials = 3;
        for (unsigned trial = 0; trial < trials; ++trial) {
            for (auto& value : hidden) value = random(generator);
            baseline.set_hidden(hidden); packed_baseline.set_hidden(hidden); producer.set_hidden(hidden);
            run(0);
            const auto reference = baseline.read_output();
            std::vector<float> full_attention_result;
            for (int plan = 1; plan < 4; ++plan) {
                run(plan);
                const auto result = plan == 1 ? packed_baseline.read_output() : consumer.read_output();
                if (reference.size() != result.size()) throw std::runtime_error("Output size mismatch");
                double total = 0;
                for (size_t index = 0; index < result.size(); ++index) {
                    if (!std::isfinite(reference[index]) || !std::isfinite(result[index]))
                        throw std::runtime_error("Non-finite block output");
                    const double error = std::abs(double(reference[index]) - result[index]);
                    errors[plan] = std::max(errors[plan], error);
                    total += error;
                    if (plan == 3)
                        partition_error = std::max(partition_error, std::abs(double(full_attention_result[index]) - result[index]));
                }
                mean_errors[plan] += total / result.size() / trials;
                if (plan == 2) full_attention_result = result;
            }
        }
        for (int warmup = 0; warmup < 3; ++warmup) for (int plan = 0; plan < 4; ++plan) run(plan);
        std::vector<double> samples[4];
        for (int iteration = 0; iteration < repeats; ++iteration) {
            for (int offset = 0; offset < 4; ++offset) {
                const int plan = (iteration + offset) % 4;
                const auto start = std::chrono::steady_clock::now();
                run(plan);
                samples[plan].push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
        }
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open output JSON");
        output << "{\"scope\":\"Actual SmolVLA vision layer-0, resident native FP16; producer/attention/consumer included; FD-offset buffer views; default SDK cache sync; excludes final host retrieval\","
               << "\"random_hidden_trials\":" << trials
               << ",\"head_partitions\":" << partitions
               << ",\"parallel_vs_full_attention_max_abs_difference\":" << partition_error << ",\"plans\":{";
        const char* names[] = {"fused_original", "native_fused_unmasked", "native_split_full_attention", "native_split_parallel_heads"};
        for (int plan = 0; plan < 4; ++plan) {
            output << (plan ? "," : "") << "\"" << names[plan] << "\":{\"median_ms\":" << median(samples[plan])
                   << ",\"fp16_max_abs_difference\":" << errors[plan] << ",\"fp16_mean_abs_difference\":" << mean_errors[plan]
                   << ",\"samples_ms\":[";
            for (size_t index = 0; index < samples[plan].size(); ++index) output << (index ? "," : "") << samples[plan][index];
            output << "]}";
        }
        output << "}}\n";
        for (int plan = 0; plan < 4; ++plan)
            std::cout << names[plan] << " median_ms=" << median(samples[plan]) << " max_diff=" << errors[plan]
                      << " mean_diff=" << mean_errors[plan] << '\n';
        std::cout << "parallel_vs_full_attention_max_diff=" << partition_error << '\n';
        if (partition_error != 0) throw std::runtime_error("Parallel heads differ from native full-head control");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
