// Resident native FP16 SDPA probe. It does not include producer/consumer graph boundaries.
#include <rknn_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int code, const char* operation) {
    if (code < 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}

class Graph {
public:
    rknn_context context = 0;
    std::vector<rknn_tensor_mem*> memory;
    std::vector<rknn_tensor_attr> input_attributes;

    Graph(const char* path, rknn_core_mask mask) {
        try {
            check(rknn_init(&context, const_cast<char*>(path), 0, 0, nullptr), "init");
            check(rknn_set_core_mask(context, mask), "core mask");
            rknn_input_output_num count{};
            check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "io count");
            if (count.n_input != 3 || count.n_output != 1) throw std::runtime_error("Expected Q/K/V -> attention");
            for (unsigned index = 0; index < count.n_input + count.n_output; ++index) {
                const bool input = index < count.n_input;
                rknn_tensor_attr attribute{};
                attribute.index = input ? index : index - count.n_input;
                check(rknn_query(context, input ? RKNN_QUERY_NATIVE_INPUT_ATTR : RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                                 &attribute, sizeof(attribute)), "native attr");
                if (attribute.type != RKNN_TENSOR_FLOAT16) throw std::runtime_error("Expected native FP16");
                auto* buffer = rknn_create_mem(context, attribute.size_with_stride);
                if (!buffer) throw std::runtime_error("native allocation failed");
                memory.push_back(buffer);
                std::memset(buffer->virt_addr, 0, buffer->size);
                // Zero Q/K and unit V make every logical output exactly one.
                if (input && attribute.index == 2) {
                    auto* values = static_cast<uint16_t*>(buffer->virt_addr);
                    std::fill(values, values + buffer->size / sizeof(uint16_t), uint16_t{0x3c00});
                }
                check(rknn_mem_sync(context, buffer, RKNN_MEMORY_SYNC_TO_DEVICE), "input sync");
                attribute.pass_through = 1;
                check(rknn_set_io_mem(context, buffer, &attribute), "native binding");
                if (input) input_attributes.push_back(attribute);
            }
        } catch (...) {
            release();
            throw;
        }
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() { release(); }

    void release() {
        for (auto* buffer : memory) rknn_destroy_mem(context, buffer);
        memory.clear();
        if (context) rknn_destroy(context);
        context = 0;
    }

    void run() { check(rknn_run(context, nullptr), "native run"); }

    double error() {
        rknn_tensor_attr attribute{};
        check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &attribute, sizeof(attribute)), "output attr");
        rknn_output output{};
        output.want_float = 1;
        check(rknn_outputs_get(context, 1, &output, nullptr), "validation output");
        double maximum = 0;
        const auto* values = static_cast<float*>(output.buf);
        for (unsigned index = 0; index < attribute.n_elems; ++index) {
            if (!std::isfinite(values[index])) {
                rknn_outputs_release(context, 1, &output);
                throw std::runtime_error("Non-finite native output");
            }
            maximum = std::max(maximum, std::abs(double(values[index]) - 1.0));
        }
        check(rknn_outputs_release(context, 1, &output), "release validation output");
        return maximum;
    }
};

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: native_attention FULL12_MODEL HALF6_MODEL REPEATS OUTPUT_JSON\n";
        return 2;
    }
    try {
        const int repeats = std::stoi(argv[3]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        Graph full(argv[1], RKNN_NPU_CORE_0_1_2);
        Graph first(argv[2], RKNN_NPU_CORE_0);
        Graph second(argv[2], RKNN_NPU_CORE_1);
        for (int warmup = 0; warmup < 3; ++warmup) { full.run(); first.run(); second.run(); }
        std::vector<double> samples[3];
        for (int iteration = 0; iteration < repeats; ++iteration) {
            for (int offset = 0; offset < 3; ++offset) {
                const int plan = (iteration + offset) % 3;
                const auto start = std::chrono::steady_clock::now();
                if (plan == 0) full.run();
                else if (plan == 1) { first.run(); second.run(); }
                else {
                    auto left = std::async(std::launch::async, [&] { first.run(); });
                    auto right = std::async(std::launch::async, [&] { second.run(); });
                    left.get();
                    right.get();
                }
                samples[plan].push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
        }
        const double maximum = std::max({full.error(), first.error(), second.error()});
        std::ofstream output(argv[4]);
        if (!output) throw std::runtime_error("Cannot open output");
        output << "{\"scope\":\"Resident native FP16 standalone SDPA; default SDK cache synchronization; no host repacking/output retrieval per call; excludes producer/consumer graph transitions\","
               << "\"constant_fixture_max_abs_error\":" << maximum << ",\"plans\":{";
        const char* names[] = {"full_three_cores", "serial_head_halves", "parallel_head_halves"};
        for (int plan = 0; plan < 3; ++plan) {
            output << (plan ? "," : "") << "\"" << names[plan] << "\":{\"median_ms\":" << median(samples[plan]) << ",\"samples_ms\":[";
            for (size_t index = 0; index < samples[plan].size(); ++index) output << (index ? "," : "") << samples[plan][index];
            output << "]}";
        }
        output << "},\"full_native_inputs\":[";
        for (size_t index = 0; index < full.input_attributes.size(); ++index) {
            const auto& attribute = full.input_attributes[index];
            output << (index ? "," : "") << "{\"format\":\"" << get_format_string(attribute.fmt)
                   << "\",\"size_with_stride\":" << attribute.size_with_stride << ",\"dims\":[";
            for (unsigned dimension = 0; dimension < attribute.n_dims; ++dimension) output << (dimension ? "," : "") << attribute.dims[dimension];
            output << "]}";
        }
        output << "]}\n";
        output.close();
        std::cout << "full_ms=" << median(samples[0]) << " serial_halves_ms=" << median(samples[1])
                  << " parallel_halves_ms=" << median(samples[2]) << " fixture_error=" << maximum << '\n';
        if (maximum != 0) throw std::runtime_error("Constant fixture validation failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
