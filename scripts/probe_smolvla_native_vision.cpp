// Complete vision + connector. Two persistent hidden buffers and shared Q/K/V/head buffers.
#include "smolvla_packed_vision.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot read " + path);
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(float)) throw std::runtime_error("Invalid FP32 fixture size");
    std::vector<float> result(static_cast<size_t>(bytes) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!file || !std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Invalid FP32 fixture data");
    return result;
}

struct Error {
    double maximum = 0, mean = 0;
};
static Error compare(const std::vector<float>& result, const std::vector<float>& reference) {
    if (result.empty() || result.size() != reference.size()) throw std::runtime_error("Output size mismatch");
    Error error;
    for (size_t index = 0; index < result.size(); ++index) {
        if (!std::isfinite(result[index]) || !std::isfinite(reference[index])) throw std::runtime_error("Non-finite result");
        const double difference = std::abs(double(result[index]) - reference[index]);
        error.maximum = std::max(error.maximum, difference);
        error.mean += difference / result.size();
    }
    return error;
}
static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}
static void write_error(std::ostream& stream, const Error& error) {
    stream << "{\"max_abs\":" << error.maximum << ",\"mae\":" << error.mean << "}";
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: native_vision MODEL_DIRECTORY REPEATS OUTPUT_JSON\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + "/";
        const int repeats = std::stoi(argv[2]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        const auto image = read_floats(directory + "image_fp32.bin");
        const auto reference = read_floats(directory + "embedding_fp32.bin");
        const auto layer_reference = read_floats(directory + "layers_fp32.bin");
        if (layer_reference.size() != 12 * 1024 * 768) throw std::runtime_error("Invalid layer reference fixture");
        NativeGraph original(directory + "vision_original.rknn", RKNN_NPU_CORE_0_1_2);
        NativeGraph repaired(directory + "vision_unmasked.rknn", RKNN_NPU_CORE_0_1_2);
        NativeGraph auditor(directory + "vision_audit.rknn", RKNN_NPU_CORE_0_1_2);
        PackedVision pipeline(directory);
        original.set_image(image); repaired.set_image(image); auditor.set_image(image); pipeline.set_image(image);

        std::vector<std::vector<float>> snapshots[2];
        auto run = [&](int plan, bool capture = false, std::array<double, 5>* stage_times = nullptr) {
            if (plan == 0) { original.run(); return; }
            if (plan == 1) { repaired.run(); return; }
            pipeline.run(plan == 3, capture ? &snapshots[plan - 2] : nullptr, stage_times);
        };

        std::vector<std::vector<float>> results;
        for (int plan = 0; plan < 4; ++plan) {
            run(plan, plan >= 2);
            results.push_back(plan == 0 ? original.read_output() : plan == 1 ? repaired.read_output() : pipeline.read_output());
        }
        const auto partition_error = compare(results[3], results[2]);
        if (partition_error.maximum != 0) throw std::runtime_error("Full/parallel attention connector results differ");
        auditor.run();
        const auto audited = auditor.read_all_outputs();
        if (audited.size() != 13) throw std::runtime_error("Expected 12 layer outputs and connector in audit graph");
        const auto audit_embedding_error = compare(audited.back(), results[0]);
        std::vector<Error> layer_errors[2];
        std::vector<Error> audit_reference_errors, pipeline_audit_errors;
        for (unsigned layer = 0; layer < 12; ++layer) {
            const auto parity = compare(snapshots[1][layer], snapshots[0][layer]);
            if (parity.maximum != 0) throw std::runtime_error("Full/parallel attention layer differs: " + std::to_string(layer));
            const std::vector<float> expected(layer_reference.begin() + layer * 1024 * 768,
                                               layer_reference.begin() + (layer + 1) * 1024 * 768);
            for (int plan = 0; plan < 2; ++plan) layer_errors[plan].push_back(compare(snapshots[plan][layer], expected));
            audit_reference_errors.push_back(compare(audited[layer], expected));
            pipeline_audit_errors.push_back(compare(snapshots[1][layer], audited[layer]));
        }
        std::vector<std::array<Error, 3>> additional_errors;
        for (int fixture = 0; fixture < 2; ++fixture) {
            auto changed = image;
            for (size_t channel = 0; channel < 3; ++channel)
                for (size_t y = 0; y < 512; ++y)
                    for (size_t x = 0; x < 512; ++x) {
                        const size_t source_y = fixture == 1 ? 511 - y : y;
                        const size_t source_x = fixture == 0 ? 511 - x : x;
                        changed[channel * 512 * 512 + y * 512 + x] = image[channel * 512 * 512 + source_y * 512 + source_x];
                    }
            original.set_image(changed); repaired.set_image(changed); pipeline.set_image(changed);
            std::vector<std::vector<float>> outputs;
            for (int plan = 0; plan < 4; ++plan) {
                run(plan);
                outputs.push_back(plan == 0 ? original.read_output() : plan == 1 ? repaired.read_output() : pipeline.read_output());
            }
            std::array<Error, 3> errors = {compare(outputs[1], outputs[0]), compare(outputs[3], outputs[2]), compare(outputs[3], outputs[0])};
            if (errors[0].maximum != 0 || errors[1].maximum != 0)
                throw std::runtime_error("Equivalent control differs after image switch");
            additional_errors.push_back(errors);
        }
        original.set_image(image); repaired.set_image(image); pipeline.set_image(image);
        for (int warmup = 0; warmup < 2; ++warmup) for (int plan = 0; plan < 4; ++plan) run(plan);
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
        // One more run after the timing loop detects stale buffer/cache state across plans.
        for (int plan = 2; plan < 4; ++plan) {
            run(plan);
            if (compare(pipeline.read_output(), results[plan]).maximum != 0) throw std::runtime_error("Repeated pipeline output changed");
        }
        std::vector<double> stage_samples[2][5];
        for (int iteration = 0; iteration < 3; ++iteration) {
            for (int plan = 2; plan < 4; ++plan) {
                std::array<double, 5> times{};
                run(plan, false, &times);
                for (unsigned stage = 0; stage < 5; ++stage) stage_samples[plan - 2][stage].push_back(times[stage]);
            }
        }
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open output JSON");
        output << "{\"scope\":\"Complete patched vision+connector B1; resident FP16 I/O; 12-layer graph dispatch/cache sync included; no layer host repacking; excludes input preparation and final host output retrieval; one synthetic image fixture\","
               << "\"hidden_buffer_count\":2,\"shared_qkv_buffers\":3,\"shared_attention_output_buffers\":1,"
               << "\"parallel_vs_full_attention_max_abs_difference\":" << partition_error.maximum << ",\"plans\":{";
        const char* names[] = {"patched_fused_original", "patched_fused_unmasked", "native_pipeline_full_attention", "native_pipeline_parallel_h4"};
        for (int plan = 0; plan < 4; ++plan) {
            output << (plan ? "," : "") << "\"" << names[plan] << "\":{\"median_ms\":" << median(samples[plan]) << ",\"embedding_vs_fp32\":";
            write_error(output, compare(results[plan], reference));
            output << ",\"embedding_vs_original_fp16\":";
            write_error(output, compare(results[plan], results[0]));
            output << ",\"samples_ms\":[";
            for (size_t index = 0; index < samples[plan].size(); ++index) output << (index ? "," : "") << samples[plan][index];
            output << "]";
            if (plan >= 2) {
                output << ",\"supplemental_stage_medians_ms\":{";
                const char* stages[] = {"patch_embedding", "qkv_producers", "attention", "projection_mlp_consumers", "postnorm_connector"};
                for (unsigned stage = 0; stage < 5; ++stage)
                    output << (stage ? "," : "") << "\"" << stages[stage] << "\":" << median(stage_samples[plan - 2][stage]);
                output << "}";
                output << ",\"layers_vs_fp32\":[";
                for (unsigned layer = 0; layer < 12; ++layer) {
                    if (layer) output << ',';
                    write_error(output, layer_errors[plan - 2][layer]);
                }
                output << "]";
            }
            output << "}";
            const auto error = compare(results[plan], results[0]);
            std::cout << names[plan] << " median_ms=" << median(samples[plan])
                      << " max_diff=" << error.maximum << " mae=" << error.mean << '\n';
        }
        output << "},\"audit_connector_vs_original\":";
        write_error(output, audit_embedding_error);
        output << ",\"fused_audit_layers_vs_fp32\":[";
        for (unsigned layer = 0; layer < 12; ++layer) {
            if (layer) output << ',';
            write_error(output, audit_reference_errors[layer]);
        }
        output << "],\"pipeline_layers_vs_fused_audit\":[";
        for (unsigned layer = 0; layer < 12; ++layer) {
            if (layer) output << ',';
            write_error(output, pipeline_audit_errors[layer]);
        }
        output << "],\"additional_image_trials\":[";
        for (size_t fixture = 0; fixture < additional_errors.size(); ++fixture) {
            output << (fixture ? "," : "") << "{\"kind\":\"" << (fixture == 0 ? "horizontal_flip" : "vertical_flip") << "\",\"mask_repair_vs_original\":";
            write_error(output, additional_errors[fixture][0]);
            output << ",\"parallel_vs_full_attention\":";
            write_error(output, additional_errors[fixture][1]);
            output << ",\"pipeline_vs_fused_original\":";
            write_error(output, additional_errors[fixture][2]);
            output << "}";
        }
        output << "]}\n";
        std::cout << "parallel_vs_full_attention_max_diff=" << partition_error.maximum << '\n';
        std::cout << "audit_connector_vs_original_max_diff=" << audit_embedding_error.maximum << '\n';
        std::cout << "last_layer_vs_fused_audit max_diff=" << pipeline_audit_errors.back().maximum
                  << " mae=" << pipeline_audit_errors.back().mean << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
