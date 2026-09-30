// Equivalent spatial shapes over the same native bits; isolated actual final-layer consumer.
#include "smolvla_packed_vision.hpp"
#include <fstream>
#include <iostream>

static std::vector<float> read_image(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != 3 * 512 * 512 * sizeof(float)) throw std::runtime_error("Expected prepared B1 image");
    std::vector<float> image(3 * 512 * 512);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
    if (!input || !std::all_of(image.begin(), image.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Invalid image");
    return image;
}

static void require_dense_spatial(const rknn_tensor_attr& attr, bool allow_extra_allocation = false) {
    if (attr.fmt != RKNN_TENSOR_NC1HWC2 || attr.type != RKNN_TENSOR_FLOAT16 || attr.n_dims != 5 ||
        attr.dims[0] != 1 || attr.dims[1] != 96 || attr.dims[4] != 8 || attr.dims[2] * attr.dims[3] != 1024 ||
        (attr.w_stride && attr.w_stride != attr.dims[3]) || attr.size_with_stride < 768 * 1024 * sizeof(__fp16) ||
        (!allow_extra_allocation && attr.size_with_stride != 768 * 1024 * sizeof(__fp16)))
    {
        std::ostringstream detail;
        detail << "Spatial view requires identical dense C8 byte order and no row padding: name=" << attr.name
               << " format=" << attr.fmt << " type=" << attr.type << " dims=";
        for (unsigned index = 0; index < attr.n_dims; ++index) detail << attr.dims[index] << ',';
        detail << " stride=" << attr.w_stride << " bytes=" << attr.size_with_stride;
        throw std::runtime_error(detail.str());
    }
}
static void bind_inputs(NativeGraph& target, NativeGraph& source, bool from_outputs = false) {
    const auto& source_attrs = from_outputs ? source.outputs : source.inputs;
    const auto& source_memory = from_outputs ? source.output_memory : source.input_memory;
    if (target.inputs.size() != 2 || source_attrs.size() != 2 || target.outputs.empty() || target.outputs.size() > 2)
        throw std::runtime_error("Unexpected consumer interface");
    for (unsigned index = 0; index < 2; ++index) {
        require_dense_spatial(target.inputs[index]);
        require_dense_spatial(source_attrs[index], from_outputs);
        auto* original = source_memory[index];
        if (!original || original->size < source_attrs[index].size_with_stride) throw std::runtime_error("Missing source allocation");
        auto* view = rknn_create_mem_from_fd(target.context, original->fd, original->virt_addr,
                                            target.inputs[index].size_with_stride, 0);
        if (!view) throw std::runtime_error("Cannot import spatial FD view");
        target.owned.push_back(view);
        target.bind(true, index, view);
    }
    for (unsigned index = 0; index < target.outputs.size(); ++index) {
        require_dense_spatial(target.outputs[index], target.outputs.size() == 2);
        target.bind(false, index, target.allocate(target.outputs[index]));
    }
}
static void check_prefix_dense_bits(NativeGraph& prefix) {
    std::vector<std::vector<float>> decoded(2, std::vector<float>(768 * 1024));
    for (unsigned output = 0; output < 2; ++output) {
        native_check(rknn_mem_sync(prefix.context, prefix.output_memory[output], RKNN_MEMORY_SYNC_FROM_DEVICE), "prefix output sync");
        const auto* native = static_cast<const __fp16*>(prefix.output_memory[output]->virt_addr);
        for (unsigned channel = 0; channel < 768; ++channel)
            for (unsigned token = 0; token < 1024; ++token)
                decoded[output][channel * 1024 + token] = native[(channel / 8 * 1024 + token) * 8 + channel % 8];
    }
    const auto logical = prefix.read_all_outputs();
    for (unsigned output = 0; output < 2; ++output)
        if (decoded[output].size() != logical[output].size() ||
            std::memcmp(decoded[output].data(), logical[output].data(), logical[output].size() * sizeof(float)))
            throw std::runtime_error("Prefix output is not a verified dense C8 payload; no zero-copy reinterpretation allowed");
}
static bool same_bits(const std::vector<float>& values, const std::vector<float>& anchor) {
    if (values.size() != anchor.size() || values.empty()) throw std::runtime_error("Output size mismatch");
    return std::memcmp(values.data(), anchor.data(), values.size() * sizeof(float)) == 0;
}
static void write_error(std::ostream& output, const std::vector<float>& values, const std::vector<float>& anchor) {
    const bool bits = same_bits(values, anchor);
    double maximum = 0, mean = 0;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]) || !std::isfinite(anchor[index])) throw std::runtime_error("Non-finite output");
        const double error = std::abs(double(values[index]) - anchor[index]);
        maximum = std::max(maximum, error);
        mean += error / values.size();
    }
    output << "{\"max_abs\":" << maximum << ",\"mae\":" << mean << ",\"bitwise_equal\":" << (bits ? "true" : "false") << '}';
}
static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9) {
        std::cerr << "usage: spatial_consumer VISION_DIR CANDIDATE_DIR CAMERA0 CAMERA1 REPEATS OUTPUT_JSON [SPLIT_DIR [STAGE_DIR]]\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + '/';
        const std::string candidates = std::string(argv[2]) + '/';
        const int repeats = std::stoi(argv[5]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        const auto images = std::array<std::vector<float>, 2>{read_image(argv[3]), read_image(argv[4])};
        PackedVision vision(directory);
        auto& original = *vision.consumers[11];
        std::vector<std::string> names = {"original_native", "original_recompiled", "spatial_1x1024", "spatial_16x64", "spatial_32x32", "spatial_64x16"};
        std::vector<std::string> paths = {directory + "layer11_suffix.rknn", candidates + "consumer_original_recompiled.rknn",
            candidates + "consumer_spatial_1x1024.rknn", candidates + "consumer_spatial_16x64.rknn",
            candidates + "consumer_spatial_32x32.rknn", candidates + "consumer_spatial_64x16.rknn"};
        std::unique_ptr<NativeGraph> split_prefix;
        if (argc >= 8) {
            const std::string split = std::string(argv[7]) + '/';
            split_prefix.reset(new NativeGraph(split + "consumer_prefix.rknn", RKNN_NPU_CORE_0_1_2, false));
            bind_inputs(*split_prefix, original);
            for (const std::string shape : {"original", "spatial_1x1024", "spatial_16x64", "spatial_32x32", "spatial_64x16"}) {
                names.push_back("split_mlp_" + shape);
                paths.push_back(split + "mlp_" + shape + ".rknn");
            }
        }
        if (argc == 9) {
            for (const std::string shape : {"1x1024", "16x64", "32x32", "64x16"}) {
                names.push_back("stage_spatial_" + shape);
                paths.push_back(std::string(argv[8]) + "/consumer_stage_spatial_" + shape + ".rknn");
            }
        }
        auto is_split = [](unsigned plan) { return plan >= 6 && plan < 11; };
        const unsigned count = names.size();
        std::vector<std::unique_ptr<NativeGraph>> owners;
        std::vector<NativeGraph*> graphs(count);
        graphs[0] = &original;
        for (unsigned plan = 1; plan < count; ++plan) {
            owners.emplace_back(new NativeGraph(paths[plan], RKNN_NPU_CORE_0_1_2, false));
            graphs[plan] = owners.back().get();
            bind_inputs(*graphs[plan], is_split(plan) ? *split_prefix : original, is_split(plan));
        }
        auto run_plan = [&](unsigned plan) {
            if (is_split(plan)) split_prefix->run();
            graphs[plan]->run();
        };
        std::array<std::vector<std::vector<double>>, 2> samples;
        std::array<std::vector<std::vector<float>>, 2> anchors;
        std::array<std::vector<rknn_mem_size>, 2> memory;
        for (unsigned camera = 0; camera < 2; ++camera) {
            samples[camera].resize(count); anchors[camera].resize(count); memory[camera].resize(count);
        }
        for (unsigned camera = 0; camera < 2; ++camera) {
            vision.set_image(images[camera]); vision.run();
            if (split_prefix) { split_prefix->run(); check_prefix_dense_bits(*split_prefix); }
            for (unsigned plan = 0; plan < count; ++plan) {
                run_plan(plan);
                anchors[camera][plan] = graphs[plan]->read_output();
                if (!std::all_of(anchors[camera][plan].begin(), anchors[camera][plan].end(), [](float value) { return std::isfinite(value); }))
                    throw std::runtime_error("Non-finite candidate");
            }
            for (unsigned warmup = 0; warmup < 2; ++warmup) for (unsigned plan = 0; plan < count; ++plan) run_plan(plan);
            std::cout << "PROBE_MEASURE_START" << std::endl;
            for (int repeat = 0; repeat < repeats; ++repeat)
                for (unsigned offset = 0; offset < count; ++offset) {
                    const unsigned plan = (repeat + offset) % count;
                    std::cout << "PROBE_PLAN_START camera" << camera << ':' << names[plan] << std::endl;
                    const auto start = std::chrono::steady_clock::now();
                    run_plan(plan);
                    samples[camera][plan].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                    std::cout << "PROBE_PLAN_END" << std::endl;
                }
            std::cout << "PROBE_MEASURE_END" << std::endl;
            if (split_prefix) { split_prefix->run(); check_prefix_dense_bits(*split_prefix); }
            for (unsigned plan = 0; plan < count; ++plan) {
                run_plan(plan);
                if (!same_bits(graphs[plan]->read_output(), anchors[camera][plan])) throw std::runtime_error("Candidate output changed after timing");
                NativeGraph profiler(paths[plan], RKNN_NPU_CORE_0_1_2, false, nullptr, RKNN_FLAG_COLLECT_PERF_MASK);
                bind_inputs(profiler, is_split(plan) ? *split_prefix : original, is_split(plan));
                profiler.run();
                if (!same_bits(profiler.read_output(), anchors[camera][plan])) throw std::runtime_error("Profile flags changed candidate bits");
                rknn_perf_detail detail{};
                native_check(rknn_query(profiler.context, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)), "perf detail");
                std::ofstream profile(std::string(argv[6]) + ".camera" + std::to_string(camera) + '.' + names[plan] + ".perf.txt");
                profile.write(detail.perf_data, detail.data_len);
                if (!profile) throw std::runtime_error("Cannot write profile");
                native_check(rknn_query(profiler.context, RKNN_QUERY_MEM_SIZE, &memory[camera][plan], sizeof(rknn_mem_size)), "memory query");
            }
        }
        vision.set_image(images[0]); vision.run();
        if (split_prefix) { split_prefix->run(); check_prefix_dense_bits(*split_prefix); }
        for (unsigned plan = 0; plan < count; ++plan) {
            run_plan(plan);
            if (!same_bits(graphs[plan]->read_output(), anchors[0][plan])) throw std::runtime_error("Restored camera state mismatch");
        }
        std::ofstream output(argv[6]);
        if (!output) throw std::runtime_error("Cannot open report");
        output << std::setprecision(10) << "{\"scope\":\"Isolated last-layer consumer on actual two-camera inputs from packed vision; all 1024 tokens and original parameters; dense native FD view checked; not whole vision/policy or fused-control bit parity\","
               << "\"repeats\":" << repeats << ",\"post_timing_and_restored_camera_checks\":true";
        if (split_prefix)
            output << ",\"split_prefix_dense_payload_checks\":true,\"split_prefix_native_output_allocation_bytes\":["
                   << split_prefix->outputs[0].size_with_stride << ',' << split_prefix->outputs[1].size_with_stride << ']';
        output << ",\"cameras\":[";
        for (unsigned camera = 0; camera < 2; ++camera) {
            output << (camera ? "," : "") << "{\"plans\":{";
            for (unsigned plan = 0; plan < count; ++plan) {
                output << (plan ? "," : "") << '"' << names[plan] << "\":{\"median_ms\":" << median(samples[camera][plan])
                       << ",\"errors_vs_original_native\":";
                write_error(output, anchors[camera][plan], anchors[camera][0]);
                if (is_split(plan)) {
                    output << ",\"errors_vs_split_original\":";
                    write_error(output, anchors[camera][plan], anchors[camera][6]);
                }
                const auto& attr = graphs[plan]->inputs[0];
                output << ",\"native_input_spatial\":[" << attr.dims[2] << ',' << attr.dims[3]
                       << "],\"native_input_bytes\":" << attr.size_with_stride
                       << ",\"timing_includes_prefix_and_mlp\":" << (is_split(plan) ? "true" : "false")
                       << ",\"sdk_profile_and_allocation_exclude_prefix\":" << (is_split(plan) ? "true" : "false")
                       << ",\"sdk_allocation_fields\":{\"weight_bytes\":"
                       << memory[camera][plan].total_weight_size << ",\"internal_bytes\":" << memory[camera][plan].total_internal_size
                       << ",\"dma_bytes\":" << memory[camera][plan].total_dma_allocated_size << "},\"samples_ms\":[";
                for (size_t index = 0; index < samples[camera][plan].size(); ++index) output << (index ? "," : "") << samples[camera][plan][index];
                output << "]}";
                std::cout << "camera" << camera << ' ' << names[plan] << " median_ms=" << median(samples[camera][plan])
                          << " bits_equal=" << same_bits(anchors[camera][plan], anchors[camera][0]) << '\n';
            }
            output << "}}";
        }
        output << "]}\n";
        if (!output) throw std::runtime_error("Report write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
