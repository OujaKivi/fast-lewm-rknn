// Same two images, fused versus packed/head-parallel execution, resident and host-I/O boundaries.
#include "smolvla_packed_vision.hpp"
#include <fstream>
#include <iostream>
#include <numeric>

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot read " + path);
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(float)) throw std::runtime_error("Invalid fixture size");
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file || !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Invalid fixture data");
    return values;
}
struct Error { double maximum = 0, mean = 0; };
static Error compare(const std::vector<float>& values, const std::vector<float>& reference) {
    if (values.empty() || values.size() != reference.size()) throw std::runtime_error("Output size mismatch");
    Error error;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]) || !std::isfinite(reference[index])) throw std::runtime_error("Non-finite output");
        const double difference = std::abs(double(values[index]) - reference[index]);
        error.maximum = std::max(error.maximum, difference);
        error.mean += difference / values.size();
    }
    return error;
}
static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}
static void write_error(std::ostream& output, const Error& error) {
    output << "{\"max_abs\":" << error.maximum << ",\"mae\":" << error.mean << "}";
}
template<class First, class Second>
static void together(First&& first, Second&& second) {
    auto left = std::async(std::launch::async, first);
    auto right = std::async(std::launch::async, second);
    left.get(); right.get();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: native_dual_view MODEL_DIRECTORY REPEATS OUTPUT_JSON\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + "/";
        const int repeats = std::stoi(argv[2]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        std::vector<float> images[2] = {read_floats(directory + "image_fp32.bin"), {}};
        if (images[0].size() != 3 * 512 * 512) throw std::runtime_error("Expected B1 512x512 image");
        images[1] = images[0];
        for (size_t channel = 0; channel < 3; ++channel)
            for (size_t y = 0; y < 512; ++y)
                for (size_t x = 0; x < 512; ++x)
                    images[1][channel * 512 * 512 + y * 512 + x] = images[0][channel * 512 * 512 + y * 512 + 511 - x];
        NativeGraph fused_left(directory + "vision_unmasked.rknn", RKNN_NPU_CORE_0_1_2);
        NativeGraph fused_right(directory + "vision_unmasked.rknn", RKNN_NPU_CORE_0_1_2, true, &fused_left);
        PackedVision packed_left(directory);
        PackedVision packed_right(directory, &packed_left);
        auto prepare = [&](bool packed, bool swapped = false) {
            const int first = swapped ? 1 : 0;
            const int second = swapped ? 0 : 1;
            if (packed) { packed_left.set_image(images[first]); packed_right.set_image(images[second]); }
            else { fused_left.set_image(images[first]); fused_right.set_image(images[second]); }
        };
        auto outputs = [&](bool packed) {
            return std::array<std::vector<float>, 2>{packed ? packed_left.read_output() : fused_left.read_output(),
                                                   packed ? packed_right.read_output() : fused_right.read_output()};
        };
        constexpr int plans = 7;
        const char* names[] = {"fused_serial_mask7", "fused_parallel_mask1_mask2", "fused_parallel_mask4_mask3",
                               "fused_parallel_mask3_mask4", "fused_parallel_mask7_mask7",
                               "packed_h4_serial_views", "packed_h4_parallel_views"};
        const rknn_core_mask masks[5][2] = {
            {RKNN_NPU_CORE_0_1_2, RKNN_NPU_CORE_0_1_2},
            {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1},
            {RKNN_NPU_CORE_2, RKNN_NPU_CORE_0_1},
            {RKNN_NPU_CORE_0_1, RKNN_NPU_CORE_2},
            {RKNN_NPU_CORE_0_1_2, RKNN_NPU_CORE_0_1_2}};
        auto configure = [&](int plan) {
            if (plan < 5) {
                native_check(rknn_set_core_mask(fused_left.context, static_cast<rknn_core_mask>(masks[plan][0])), "left core mask");
                native_check(rknn_set_core_mask(fused_right.context, static_cast<rknn_core_mask>(masks[plan][1])), "right core mask");
            }
        };
        auto run = [&](int plan) {
            if (plan == 0) { fused_left.run(); fused_right.run(); }
            else if (plan < 5) together([&] { fused_left.run(); }, [&] { fused_right.run(); });
            else if (plan == 5) { packed_left.run(); packed_right.run(); }
            else together([&] { packed_left.run(); }, [&] { packed_right.run(); });
        };
        prepare(false); configure(0); run(0);
        const auto fused_reference = outputs(false);
        prepare(true);
        packed_left.run(false); packed_right.run(false);
        const auto packed_reference = outputs(true);
        std::array<Error, 2> errors[plans];
        for (int plan = 0; plan < plans; ++plan) {
            configure(plan); prepare(plan >= 5); run(plan);
            const auto values = outputs(plan >= 5);
            for (int camera = 0; camera < 2; ++camera) {
                errors[plan][camera] = compare(values[camera], fused_reference[camera]);
                const auto parity = compare(values[camera], plan >= 5 ? packed_reference[camera] : fused_reference[camera]);
                if (parity.maximum != 0) throw std::runtime_error(std::string("Schedule changed output: ") + names[plan]);
            }
        }
        // Context duplication and graph concurrency must not alias the two cameras' state.
        for (int plan = 0; plan < plans; ++plan) {
            configure(plan); prepare(plan >= 5, true); run(plan);
            const auto values = outputs(plan >= 5);
            for (int camera = 0; camera < 2; ++camera)
                if (compare(values[camera], plan >= 5 ? packed_reference[1 - camera] : fused_reference[1 - camera]).maximum != 0)
                    throw std::runtime_error(std::string("Swapped-camera state changed: ") + names[plan]);
        }
        prepare(false); prepare(true);
        for (int warmup = 0; warmup < 2; ++warmup)
            for (int plan = 0; plan < plans; ++plan) { configure(plan); run(plan); }

        std::vector<double> samples[plans][2];
        std::cout << "PROBE_MEASURE_START" << std::endl;
        for (int iteration = 0; iteration < repeats; ++iteration) {
            for (int offset = 0; offset < plans; ++offset) {
                const int plan = (iteration + offset) % plans;
                configure(plan);
                for (int mode_offset = 0; mode_offset < 2; ++mode_offset) {
                    const int mode = (iteration + offset + mode_offset) % 2;
                    std::cout << "PROBE_PLAN_START " << names[plan] << (mode ? ":input_compute_output" : ":resident_compute") << std::endl;
                    const auto start = std::chrono::steady_clock::now();
                    if (mode) prepare(plan >= 5);
                    run(plan);
                    if (mode) {
                        const auto values = outputs(plan >= 5);
                        (void)values;
                    }
                    samples[plan][mode].push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count());
                    std::cout << "PROBE_PLAN_END" << std::endl;
                }
            }
        }
        std::cout << "PROBE_MEASURE_END" << std::endl;
        for (int plan = 0; plan < plans; ++plan) {
            configure(plan); run(plan);
            const auto values = outputs(plan >= 5);
            for (int camera = 0; camera < 2; ++camera)
                if (compare(values[camera], plan >= 5 ? packed_reference[camera] : fused_reference[camera]).maximum != 0)
                    throw std::runtime_error(std::string("Repeated schedule changed output: ") + names[plan]);
        }
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open output JSON");
        output << "{\"scope\":\"Two synthetic B1 images (fixture and horizontal flip), patched/unmasked full vision+connector; identical resident FP16 I/O boundaries; not VLA rollout\","
               << "\"second_camera_contexts\":\"rknn_dup_context with separately allocated/bound I/O; no claim of measured physical weight sharing\","
               << "\"repeats\":" << repeats << ",\"swapped_camera_and_repeated_schedule_checks\":true,\"plans\":{";
        for (int plan = 0; plan < plans; ++plan) {
            output << (plan ? "," : "") << "\"" << names[plan] << "\":{\"errors_vs_fused_fp16\":[";
            write_error(output, errors[plan][0]); output << ','; write_error(output, errors[plan][1]);
            output << "],\"timings\":{";
            const char* modes[] = {"resident_compute", "input_compute_output"};
            for (int mode = 0; mode < 2; ++mode) {
                auto ordered = samples[plan][mode];
                std::sort(ordered.begin(), ordered.end());
                output << (mode ? "," : "") << "\"" << modes[mode] << "\":{\"median_ms\":" << median(ordered)
                       << ",\"p95_nearest_rank_ms\":" << ordered[static_cast<size_t>(std::ceil(0.95 * ordered.size())) - 1]
                       << ",\"samples_ms\":[";
                for (size_t index = 0; index < samples[plan][mode].size(); ++index)
                    output << (index ? "," : "") << samples[plan][mode][index];
                output << "]}";
            }
            output << "}}";
            std::cout << names[plan] << " compute_ms=" << median(samples[plan][0])
                      << " input_compute_output_ms=" << median(samples[plan][1]) << '\n';
        }
        output << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
