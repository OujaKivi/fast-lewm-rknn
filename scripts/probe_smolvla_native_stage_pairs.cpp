// Isolated final-layer stage contention; not a complete two-view scheduling result.
#include "smolvla_packed_vision.hpp"
#include <fstream>
#include <iostream>

template<class First, class Second>
static void together(First&& first, Second&& second) {
    auto a = std::async(std::launch::async, first);
    auto b = std::async(std::launch::async, second);
    a.get(); b.get();
}
static void heads(PackedVision& vision) {
    std::vector<std::future<void>> pending;
    for (unsigned part = 0; part < 3; ++part)
        pending.emplace_back(std::async(std::launch::async, [&, part] { vision.parts[part]->run(); }));
    for (auto& task : pending) task.get();
}
static double max_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) throw std::runtime_error("Output size mismatch");
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("Non-finite output");
        error = std::max(error, std::abs(double(a[i]) - b[i]));
    }
    return error;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: native_stage_pairs MODEL_DIRECTORY REPEATS OUTPUT_JSON\n";
        return 2;
    }
    try {
        const std::string directory = std::string(argv[1]) + "/";
        const int repeats = std::stoi(argv[2]);
        if (repeats < 1) throw std::runtime_error("Positive repeats required");
        std::vector<float> image(3 * 512 * 512);
        std::ifstream fixture(directory + "image_fp32.bin", std::ios::binary);
        fixture.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
        if (!fixture) throw std::runtime_error("Cannot read image fixture");
        PackedVision left(directory), right(directory, &left);
        left.set_image(image);
        for (size_t c = 0; c < 3; ++c)
            for (size_t y = 0; y < 512; ++y)
                std::reverse(image.begin() + c * 512 * 512 + y * 512,
                             image.begin() + c * 512 * 512 + (y + 1) * 512);
        right.set_image(image);
        left.run(); right.run();
        // Layer 11 reads hidden[1] and writes hidden[0]; its repeated execution
        // leaves the real producer input intact. Q/K/V and attention stay resident.
        const auto left_reference = left.consumers[11]->read_output();
        const auto right_reference = right.consumers[11]->read_output();
        constexpr unsigned count = 13;
        const char* names[count] = {"qkv_serial_mask7", "qkv_parallel_mask7_mask7", "qkv_parallel_mask1_mask2",
            "attention_h4_serial_views", "attention_h4_parallel_views", "consumer_serial_mask7",
            "consumer_parallel_mask7_mask7", "consumer_parallel_mask1_mask2", "consumer_parallel_mask3_mask4",
            "mixed_attention_consumer_serial_mask7", "mixed_attention_consumer_parallel_mask7",
            "mixed_attention_consumer_parallel_mask1", "mixed_attention_consumer_parallel_mask3"};
        auto configure = [&](unsigned plan) {
            auto& a = *(plan < 3 ? left.producers[11] : left.consumers[11]);
            auto& b = *(plan < 3 ? right.producers[11] : right.consumers[11]);
            const auto mask_a = (plan == 2 || plan == 7 || plan == 11) ? RKNN_NPU_CORE_0 :
                (plan == 8 || plan == 12) ? RKNN_NPU_CORE_0_1 : RKNN_NPU_CORE_0_1_2;
            const auto mask_b = (plan == 2 || plan == 7) ? RKNN_NPU_CORE_1 :
                plan == 8 ? RKNN_NPU_CORE_2 : RKNN_NPU_CORE_0_1_2;
            native_check(rknn_set_core_mask(a.context, mask_a), "stage left mask");
            native_check(rknn_set_core_mask(b.context, mask_b), "stage right mask");
        };
        auto run = [&](unsigned plan) {
            if (plan < 3) {
                if (plan == 0) { left.producers[11]->run(); right.producers[11]->run(); }
                else together([&] { left.producers[11]->run(); }, [&] { right.producers[11]->run(); });
            } else if (plan < 5) {
                if (plan == 3) { heads(left); heads(right); }
                else together([&] { heads(left); }, [&] { heads(right); });
            } else if (plan < 9) {
                if (plan == 5) { left.consumers[11]->run(); right.consumers[11]->run(); }
                else together([&] { left.consumers[11]->run(); }, [&] { right.consumers[11]->run(); });
            } else if (plan == 9) { left.consumers[11]->run(); heads(right); }
            else together([&] { left.consumers[11]->run(); }, [&] { heads(right); });
        };
        auto verify = [&] {
            if (max_error(left.consumers[11]->read_output(), left_reference) != 0 ||
                max_error(right.consumers[11]->read_output(), right_reference) != 0)
                throw std::runtime_error("Stage schedule changed output");
        };
        for (unsigned plan = 0; plan < count; ++plan) {
            configure(plan); run(plan);
            if (plan < 3) { run(3); configure(5); run(5); }
            else if (plan < 5) { configure(5); run(5); }
            verify();
        }
        std::vector<double> samples[count];
        std::cout << "PROBE_MEASURE_START" << std::endl;
        for (int iteration = 0; iteration < repeats; ++iteration)
            for (unsigned offset = 0; offset < count; ++offset) {
                const unsigned plan = (iteration + offset) % count;
                configure(plan);
                std::cout << "PROBE_PLAN_START " << names[plan] << std::endl;
                const auto start = std::chrono::steady_clock::now();
                run(plan);
                samples[plan].push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
                std::cout << "PROBE_PLAN_END" << std::endl;
            }
        std::cout << "PROBE_MEASURE_END" << std::endl;
        verify();
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open output JSON");
        output << "{\"scope\":\"Isolated real final-layer stages of two synthetic views, resident native FP16; not end-to-end scheduling\","
               << "\"output_parity\":true,\"repeats\":" << repeats << ",\"plans\":{";
        for (unsigned plan = 0; plan < count; ++plan) {
            auto ordered = samples[plan];
            std::sort(ordered.begin(), ordered.end());
            const double median = (ordered[(ordered.size() - 1) / 2] + ordered[ordered.size() / 2]) / 2;
            output << (plan ? "," : "") << '"' << names[plan] << "\":{\"median_ms\":" << median
                   << ",\"samples_ms\":[";
            for (size_t i = 0; i < samples[plan].size(); ++i) output << (i ? "," : "") << samples[plan][i];
            output << "]}";
            std::cout << names[plan] << " median_ms=" << median << '\n';
        }
        output << "}}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
