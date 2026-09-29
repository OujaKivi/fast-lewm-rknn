// Fixed-zero-input graph probe. Profiling timings are not production latencies.
#include <rknn_api.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(int result, const char* operation) {
    if (result < 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: probe MODEL CORE_MASK PROFILE REPEATS REPORT_PATH\n";
        return 2;
    }
    rknn_context context = 0;
    try {
        const bool profile = std::stoi(argv[3]) != 0;
        const int repeats = std::stoi(argv[4]);
        if (repeats < 1) throw std::runtime_error("repeats must be positive");
        check(rknn_init(&context, argv[1], 0, profile ? RKNN_FLAG_COLLECT_PERF_MASK : 0, nullptr), "init");
        check(rknn_set_core_mask(context, static_cast<rknn_core_mask>(std::stoi(argv[2]))), "core mask");
        rknn_input_output_num count{};
        check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "io query");
        rknn_mem_size memory{};
        check(rknn_query(context, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory)), "memory query");
        std::vector<rknn_input> inputs(count.n_input);
        std::vector<std::vector<float>> storage(count.n_input);
        size_t input_bytes = 0;
        for (unsigned i = 0; i < count.n_input; ++i) {
            rknn_tensor_attr attr{};
            attr.index = i;
            check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "input attr");
            storage[i].resize(attr.n_elems, 0.0f);
            inputs[i].index = i;
            inputs[i].buf = storage[i].data();
            inputs[i].size = storage[i].size() * sizeof(float);
            inputs[i].type = RKNN_TENSOR_FLOAT32;
            inputs[i].fmt = attr.fmt;
            input_bytes += inputs[i].size;
        }
        check(rknn_inputs_set(context, count.n_input, inputs.data()), "inputs");
        std::vector<double> milliseconds;
        for (int iteration = -2; iteration < repeats; ++iteration) {
            if (iteration == 0) std::cout << "PROBE_MEASURE_START" << std::endl;
            std::vector<rknn_output> outputs(count.n_output);
            for (unsigned i = 0; i < count.n_output; ++i) outputs[i].index = i;
            const auto start = std::chrono::steady_clock::now();
            check(rknn_run(context, nullptr), "run");
            check(rknn_outputs_get(context, count.n_output, outputs.data(), nullptr), "outputs");
            const auto stop = std::chrono::steady_clock::now();
            if (iteration >= 0) milliseconds.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
            if (profile && iteration == repeats - 1) {
                rknn_perf_detail detail{};
                check(rknn_query(context, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)), "perf");
                std::ofstream file(argv[5]);
                if (!file) throw std::runtime_error("cannot open report");
                file.write(detail.perf_data, detail.data_len);
            }
            check(rknn_outputs_release(context, count.n_output, outputs.data()), "release outputs");
        }
        std::cout << "PROBE_MEASURE_END" << std::endl;
        std::sort(milliseconds.begin(), milliseconds.end());
        const size_t n = milliseconds.size();
        const double median = (milliseconds[(n - 1) / 2] + milliseconds[n / 2]) / 2;
        std::cout << "{\"profile\":" << (profile ? "true" : "false")
                  << ",\"core_mask\":" << argv[2] << ",\"input_bytes\":" << input_bytes
                  << ",\"weight_bytes\":" << memory.total_weight_size
                  << ",\"internal_bytes\":" << memory.total_internal_size
                  << ",\"dma_bytes\":" << memory.total_dma_allocated_size
                  << ",\"median_ms\":" << median << ",\"samples_ms\":[";
        for (size_t i = 0; i < n; ++i) std::cout << (i ? "," : "") << milliseconds[i];
        std::cout << "]}\n";
        rknn_destroy(context);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (context) rknn_destroy(context);
        return 1;
    }
}
