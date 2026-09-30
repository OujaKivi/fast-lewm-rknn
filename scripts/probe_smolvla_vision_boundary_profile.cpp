// Profile actual final-layer inputs after an ordinary full visual pass; diagnostic only.
#include "smolvla_packed_vision.hpp"
#include <fstream>
#include <iostream>

static std::vector<float> image_from(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != 3 * 512 * 512 * sizeof(float))
        throw std::runtime_error("Expected B1 prepared FP32 camera");
    std::vector<float> image(3 * 512 * 512);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
    if (!input || !std::all_of(image.begin(), image.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Invalid image");
    return image;
}

static void write_attrs(std::ostream& output, const std::vector<rknn_tensor_attr>& attrs) {
    output << '[';
    for (size_t index = 0; index < attrs.size(); ++index) {
        const auto& attr = attrs[index];
        output << (index ? "," : "") << "{\"index\":" << attr.index << ",\"format\":\"" << get_format_string(attr.fmt)
               << "\",\"size_with_stride\":" << attr.size_with_stride << ",\"w_stride\":" << attr.w_stride << ",\"dims\":[";
        for (unsigned dim = 0; dim < attr.n_dims; ++dim) output << (dim ? "," : "") << attr.dims[dim];
        output << "]}";
    }
    output << ']';
}

static void write_error(std::ostream& output, const std::vector<float>& values, const std::vector<float>& anchor) {
    if (values.empty() || values.size() != anchor.size()) throw std::runtime_error("Audit output size mismatch");
    double maximum = 0, mean = 0;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]) || !std::isfinite(anchor[index])) throw std::runtime_error("Non-finite audit output");
        const double error = std::abs(double(values[index]) - anchor[index]);
        maximum = std::max(maximum, error);
        mean += error / values.size();
    }
    output << "{\"max_abs\":" << maximum << ",\"mae\":" << mean << ",\"bitwise_equal\":"
           << (std::memcmp(values.data(), anchor.data(), values.size() * sizeof(float)) == 0 ? "true" : "false") << '}';
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: vision_boundary_profile MODEL_DIR CAMERA_FP32 OUTPUT_PREFIX\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + '/';
        const std::string prefix = argv[3];
        PackedVision vision(directory);
        const auto image = image_from(argv[2]);
        vision.set_image(image);
        std::vector<std::vector<float>> layers;
        vision.run(true, &layers);
        NativeGraph fused(directory + "vision_unmasked.rknn", RKNN_NPU_CORE_0_1_2);
        NativeGraph auditor(directory + "vision_audit.rknn", RKNN_NPU_CORE_0_1_2);
        fused.set_image(image); auditor.set_image(image);
        fused.run(); auditor.run();
        const auto audited = auditor.read_all_outputs();
        if (audited.size() != 13 || layers.size() != 12) throw std::runtime_error("Missing audit layers");
        vision.attention.run();
        // The two recycled hidden buffers still contain the final layer's input/output.
        // Earlier layers' inputs are no longer preserved and must not be called actual inputs.
        std::array<NativeGraph*, 3> sources = {vision.producers[11].get(), &vision.attention, vision.consumers[11].get()};
        const char* names[] = {"qkv", "attention_full", "consumer"};
        const char* files[] = {"layer11_qkv.rknn", "attention_full.rknn", "layer11_suffix.rknn"};
        std::ofstream report(prefix + ".json");
        if (!report) throw std::runtime_error("Cannot open report");
        report << std::setprecision(10) << "{\"scope\":\"SDK diagnostic on actual last-layer resident inputs; independent profiled graphs, not production latency or physical DDR measurements\","
               << "\"layer\":11,\"repeats\":3,\"fused_audit_connector_vs_fused_control\":";
        write_error(report, audited.back(), fused.read_output());
        report << ",\"packed_connector_vs_fused_control\":";
        write_error(report, vision.read_output(), fused.read_output());
        report << ",\"packed_layers_vs_fused_audit\":[";
        for (unsigned layer = 0; layer < 12; ++layer) {
            if (layer) report << ',';
            write_error(report, layers[layer], audited[layer]);
        }
        report << "],\"audit_limitation\":\"Extra layer outputs may change internal fusion; even matching final bits does not prove all internal execution identical to the single-output graph\",\"plans\":{";
        for (unsigned stage = 0; stage < sources.size(); ++stage) {
            auto& source = *sources[stage];
            const auto expected = source.read_all_outputs();
            NativeGraph profiler(directory + files[stage], RKNN_NPU_CORE_0_1_2, false, nullptr, RKNN_FLAG_COLLECT_PERF_MASK);
            for (unsigned index = 0; index < source.inputs.size(); ++index)
                profiler.alias(true, index, source.input_memory[index], source.inputs[index]);
            for (unsigned index = 0; index < profiler.outputs.size(); ++index)
                profiler.bind(false, index, profiler.allocate(profiler.outputs[index]));
            profiler.run();
            std::vector<double> samples;
            for (unsigned repeat = 0; repeat < 3; ++repeat) {
                const auto start = std::chrono::steady_clock::now();
                profiler.run();
                samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                const auto outputs = profiler.read_all_outputs();
                if (outputs.size() != expected.size()) throw std::runtime_error("Missing diagnostic output");
                for (unsigned index = 0; index < outputs.size(); ++index)
                    if (outputs[index].size() != expected[index].size() ||
                        std::memcmp(outputs[index].data(), expected[index].data(), outputs[index].size() * sizeof(float)) != 0)
                        throw std::runtime_error("Profiled graph differs from actual native anchor");
            }
            rknn_perf_detail detail{};
            native_check(rknn_query(profiler.context, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)), "perf detail");
            std::ofstream profile(prefix + '.' + names[stage] + ".perf.txt");
            profile.write(detail.perf_data, detail.data_len);
            if (!profile) throw std::runtime_error("Cannot save profile");
            rknn_mem_size memory{};
            native_check(rknn_query(profiler.context, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory)), "mem size");
            report << (stage ? "," : "") << '"' << names[stage] << "\":{\"tested_output_bitwise_equal\":true,\"profiled_run_samples_ms\":[";
            for (size_t index = 0; index < samples.size(); ++index) report << (index ? "," : "") << samples[index];
            report << "],\"sdk_allocation_fields\":{\"weight_bytes\":" << memory.total_weight_size
                   << ",\"internal_bytes\":" << memory.total_internal_size << ",\"dma_bytes\":" << memory.total_dma_allocated_size
                   << "},\"native_inputs\":";
            write_attrs(report, profiler.inputs);
            report << ",\"native_outputs\":";
            write_attrs(report, profiler.outputs);
            report << '}';
            std::cout << names[stage] << " profiled anchor bit checks passed\n";
        }
        report << "}}\n";
        if (!report) throw std::runtime_error("Report write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
