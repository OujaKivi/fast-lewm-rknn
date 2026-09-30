// Actual camera branches, matched whole-vision boundaries, and separate stage diagnostics.
#include "smolvla_packed_vision.hpp"
#include <fstream>
#include <iostream>

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot read " + path);
    const auto bytes = file.tellg();
    if (bytes != 3 * 512 * 512 * sizeof(float)) throw std::runtime_error("Expected B1 FP32 512x512 RGB image");
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file || !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Invalid fixture data");
    return values;
}

struct Error { double maximum = 0, mean = 0; bool bitwise = false; };
static Error compare(const std::vector<float>& values, const std::vector<float>& reference) {
    if (values.empty() || values.size() != reference.size()) throw std::runtime_error("Output size mismatch");
    Error error;
    error.bitwise = std::memcmp(values.data(), reference.data(), values.size() * sizeof(float)) == 0;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]) || !std::isfinite(reference[index])) throw std::runtime_error("Non-finite output");
        const double difference = std::abs(double(values[index]) - reference[index]);
        error.maximum = std::max(error.maximum, difference);
        error.mean += difference / values.size();
    }
    return error;
}
static void write_error(std::ostream& output, const Error& error) {
    output << "{\"max_abs\":" << error.maximum << ",\"mae\":" << error.mean
           << ",\"bitwise_equal\":" << (error.bitwise ? "true" : "false") << '}';
}
static double median(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("No samples");
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}
static void write_samples(std::ostream& output, const std::vector<double>& values) {
    output << "{\"median_ms\":" << median(values) << ",\"samples_ms\":[";
    for (size_t index = 0; index < values.size(); ++index) output << (index ? "," : "") << values[index];
    output << "]}";
}
static void save_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!output) throw std::runtime_error("Cannot write " + path);
}
template<class First, class Second>
static void together(First&& first, Second&& second) {
    auto left = std::async(std::launch::async, first);
    auto right = std::async(std::launch::async, second);
    left.get(); right.get();
}

struct Plan {
    const char* name;
    bool packed, parallel_heads, parallel_cameras;
    int camera;  // -1 means both camera branches; 0/1 is a service measurement only.
};

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: vision_service_matrix MODEL_DIR REPEATS OUTPUT_JSON CAMERA0_FP32 CAMERA1_FP32\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + "/";
        const int repeats = std::stoi(argv[2]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        const std::array<std::vector<float>, 2> images = {read_floats(argv[4]), read_floats(argv[5])};
        NativeGraph fused_left(directory + "vision_unmasked.rknn", RKNN_NPU_CORE_0_1_2);
        NativeGraph fused_right(directory + "vision_unmasked.rknn", RKNN_NPU_CORE_0_1_2, true, &fused_left);
        PackedVision packed_left(directory);
        PackedVision packed_right(directory, &packed_left);
        std::array<NativeGraph*, 2> fused = {&fused_left, &fused_right};
        std::array<PackedVision*, 2> packed = {&packed_left, &packed_right};
        const std::array<Plan, 10> plans = {{
            {"fused_camera0_mask7", false, false, false, 0},
            {"fused_camera1_mask7", false, false, false, 1},
            {"packed_full_camera0", true, false, false, 0},
            {"packed_full_camera1", true, false, false, 1},
            {"packed_h4_camera0", true, true, false, 0},
            {"packed_h4_camera1", true, true, false, 1},
            {"fused_dual_serial_mask7", false, false, false, -1},
            {"fused_dual_parallel_mask1_mask2", false, false, true, -1},
            {"packed_h4_dual_serial", true, true, false, -1},
            {"packed_h4_dual_parallel", true, true, true, -1}
        }};
        auto cameras = [](const Plan& plan) {
            return plan.camera < 0 ? std::vector<int>{0, 1} : std::vector<int>{plan.camera};
        };
        auto configure = [&](const Plan& plan) {
            if (plan.packed) return;
            for (int camera : cameras(plan))
                native_check(rknn_set_core_mask(fused[camera]->context, plan.parallel_cameras ?
                    (camera == 0 ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_1) : RKNN_NPU_CORE_0_1_2), "fused core mask");
        };
        auto prepare = [&](const Plan& plan, bool swapped = false) {
            for (int camera : cameras(plan)) {
                const auto& image = images[swapped ? 1 - camera : camera];
                if (plan.packed) packed[camera]->set_image(image);
                else fused[camera]->set_image(image);
            }
        };
        auto run_camera = [&](const Plan& plan, int camera) {
            if (plan.packed) packed[camera]->run(plan.parallel_heads);
            else fused[camera]->run();
        };
        auto run = [&](const Plan& plan) {
            if (plan.parallel_cameras) together([&] { run_camera(plan, 0); }, [&] { run_camera(plan, 1); });
            else for (int camera : cameras(plan)) run_camera(plan, camera);
        };
        auto read = [&](const Plan& plan, int camera) {
            return plan.packed ? packed[camera]->read_output() : fused[camera]->read_output();
        };
        std::array<std::vector<float>, 2> fused_reference, packed_reference;
        std::array<std::vector<std::vector<float>>, 2> layer_reference;
        for (int camera = 0; camera < 2; ++camera) {
            configure(plans[camera]); prepare(plans[camera]); run(plans[camera]);
            fused_reference[camera] = read(plans[camera], camera);
            packed[camera]->set_image(images[camera]);
            packed[camera]->run(false, &layer_reference[camera]);
            packed_reference[camera] = packed[camera]->read_output();
            save_floats(std::string(argv[3]) + ".fused_camera" + std::to_string(camera) + ".bin", fused_reference[camera]);
            save_floats(std::string(argv[3]) + ".packed_camera" + std::to_string(camera) + ".bin", packed_reference[camera]);
        }
        std::array<std::vector<Error>, 2> layer_parity;
        for (int camera = 0; camera < 2; ++camera) {
            std::vector<std::vector<float>> snapshots;
            packed[camera]->run(true, &snapshots);
            if (snapshots.size() != 12 || layer_reference[camera].size() != 12) throw std::runtime_error("Missing layer outputs");
            for (unsigned layer = 0; layer < 12; ++layer) {
                layer_parity[camera].push_back(compare(snapshots[layer], layer_reference[camera][layer]));
                if (!layer_parity[camera].back().bitwise) throw std::runtime_error("Head/full layer bits differ");
            }
        }
        std::array<std::vector<Error>, plans.size()> errors;
        auto audit = [&](const Plan& plan, bool swapped) {
            for (int camera : cameras(plan)) {
                const int source = swapped ? 1 - camera : camera;
                const auto values = read(plan, camera);
                if (!compare(values, plan.packed ? packed_reference[source] : fused_reference[source]).bitwise)
                    throw std::runtime_error(std::string("Schedule/swap/restore changed bits: ") + plan.name);
            }
        };
        for (size_t index = 0; index < plans.size(); ++index) {
            const auto& plan = plans[index];
            configure(plan); prepare(plan); run(plan); audit(plan, false);
            for (int camera : cameras(plan)) errors[index].push_back(compare(read(plan, camera), fused_reference[camera]));
            prepare(plan, true); run(plan); audit(plan, true);
            prepare(plan); run(plan); audit(plan, false);
        }
        for (int warmup = 0; warmup < 2; ++warmup)
            for (const auto& plan : plans) { configure(plan); prepare(plan); run(plan); }
        std::array<std::array<std::vector<double>, 2>, plans.size()> samples;
        std::cout << "PROBE_MEASURE_START" << std::endl;
        for (int iteration = 0; iteration < repeats; ++iteration) {
            for (size_t offset = 0; offset < plans.size(); ++offset) {
                const size_t index = (iteration + offset) % plans.size();
                const auto& plan = plans[index];
                configure(plan);
                for (int mode_offset = 0; mode_offset < 2; ++mode_offset) {
                    const int mode = (iteration + offset + mode_offset) % 2;
                    std::cout << "PROBE_PLAN_START " << plan.name << (mode ? ":input_compute_output" : ":resident_compute") << std::endl;
                    const auto start = std::chrono::steady_clock::now();
                    if (mode) prepare(plan);
                    run(plan);
                    if (mode) for (int camera : cameras(plan)) (void)read(plan, camera);
                    samples[index][mode].push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count());
                    std::cout << "PROBE_PLAN_END" << std::endl;
                }
            }
        }
        std::cout << "PROBE_MEASURE_END" << std::endl;
        for (const auto& plan : plans) { configure(plan); run(plan); audit(plan, false); }
        std::array<std::array<std::array<std::vector<double>, 5>, 2>, 2> stages;
        std::cout << "PROBE_STAGE_DIAGNOSTIC_START" << std::endl;
        for (int repeat = 0; repeat < 3; ++repeat)
            for (int camera = 0; camera < 2; ++camera)
                for (int mode_offset = 0; mode_offset < 2; ++mode_offset) {
                    const int mode = (repeat + mode_offset) % 2;
                    std::array<double, 5> times{};
                    packed[camera]->run(mode == 1, nullptr, &times);
                    for (unsigned stage = 0; stage < 5; ++stage) stages[camera][mode][stage].push_back(times[stage]);
                    if (!compare(packed[camera]->read_output(), packed_reference[camera]).bitwise)
                        throw std::runtime_error("Stage instrumentation changed output");
                }
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open report");
        output << std::setprecision(10);
        output << "{\"scope\":\"Actual recorded camera branches; full 12-layer vision and connector; not complete policy, closed loop or edge-cloud service\","
               << "\"repeats\":" << repeats << ",\"contexts\":\"Both fused and packed camera contexts coexist; duplicate contexts have independent I/O; not isolated RSS\","
               << "\"schedule_swap_restore_and_post_timing_bitwise_checks\":true,\"head_vs_full_layer_parity\":[";
        for (int camera = 0; camera < 2; ++camera) {
            output << (camera ? "," : "") << '[';
            for (unsigned layer = 0; layer < 12; ++layer) { if (layer) output << ','; write_error(output, layer_parity[camera][layer]); }
            output << ']';
        }
        output << "],\"plans\":{";
        for (size_t index = 0; index < plans.size(); ++index) {
            const auto& plan = plans[index];
            output << (index ? "," : "") << '"' << plan.name << "\":{\"camera_count\":" << cameras(plan).size()
                   << ",\"anchor\":\"" << (plan.packed ? "packed_full" : "fused") << "\",\"errors_vs_fused\":[";
            for (size_t camera = 0; camera < errors[index].size(); ++camera) { if (camera) output << ','; write_error(output, errors[index][camera]); }
            output << "],\"timings\":{";
            for (int mode = 0; mode < 2; ++mode) {
                output << (mode ? "," : "") << '"' << (mode ? "input_compute_output" : "resident_compute") << "\":";
                write_samples(output, samples[index][mode]);
            }
            output << "}}";
            std::cout << plan.name << " compute_ms=" << median(samples[index][0]) << " io_ms=" << median(samples[index][1]) << '\n';
        }
        output << "},\"supplemental_stage_diagnostics\":{\"repeats\":3,\"not_pooled_with_formal_timings\":true,\"cameras\":[";
        const char* stage_names[] = {"patch", "qkv", "attention", "projection_mlp", "connector"};
        for (int camera = 0; camera < 2; ++camera) {
            output << (camera ? "," : "") << '{';
            for (int mode = 0; mode < 2; ++mode) {
                output << (mode ? "," : "") << '"' << (mode ? "packed_h4" : "packed_full") << "\":{";
                for (unsigned stage = 0; stage < 5; ++stage) {
                    output << (stage ? "," : "") << '"' << stage_names[stage] << "\":";
                    write_samples(output, stages[camera][mode][stage]);
                }
                output << '}';
            }
            output << '}';
        }
        output << "]}}\n";
        if (!output) throw std::runtime_error("Report write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
