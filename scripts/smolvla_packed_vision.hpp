#pragma once
#include "rknn_native_graph.hpp"
#include <array>
#include <chrono>
#include <future>
#include <iomanip>
#include <memory>
#include <sstream>

class PackedVision {
public:
    // Allocation owner precedes and outlives all imported views.
    NativeGraph stem, attention;
    std::vector<std::unique_ptr<NativeGraph>> parts, producers, consumers;
    std::unique_ptr<NativeGraph> tail;

    explicit PackedVision(const std::string& directory, PackedVision* source = nullptr)
        : stem(directory + "stem_native.rknn", RKNN_NPU_CORE_0_1_2, true, source ? &source->stem : nullptr),
          attention(directory + "attention_full.rknn", RKNN_NPU_CORE_0_1_2, false, source ? &source->attention : nullptr) {
        for (unsigned part = 0; part < 3; ++part)
            parts.emplace_back(new NativeGraph(directory + "attention_h4.rknn", static_cast<rknn_core_mask>(1u << part),
                                                false, source ? source->parts.at(part).get() : nullptr));
        rknn_tensor_mem* hidden[] = {stem.output_memory.at(0), stem.allocate(stem.outputs.at(0))};
        std::vector<rknn_tensor_mem*> qkv;
        for (const auto& attr : attention.inputs) qkv.push_back(stem.allocate(attr));
        auto* attention_output = stem.allocate(attention.outputs.at(0));
        for (unsigned index = 0; index < 3; ++index) {
            attention.alias(true, index, qkv.at(index), attention.inputs.at(index));
            for (unsigned part = 0; part < 3; ++part)
                parts[part]->alias(true, index, qkv.at(index), attention.inputs.at(index), 3, part);
        }
        attention.alias(false, 0, attention_output, attention.outputs.at(0));
        for (unsigned part = 0; part < 3; ++part)
            parts[part]->alias(false, 0, attention_output, attention.outputs.at(0), 3, part);
        for (unsigned layer = 0; layer < 12; ++layer) {
            std::ostringstream prefix;
            prefix << directory << "layer" << std::setw(2) << std::setfill('0') << layer;
            producers.emplace_back(new NativeGraph(prefix.str() + "_qkv.rknn", RKNN_NPU_CORE_0_1_2, false,
                                                     source ? source->producers.at(layer).get() : nullptr));
            consumers.emplace_back(new NativeGraph(prefix.str() + "_suffix.rknn", RKNN_NPU_CORE_0_1_2, false,
                                                     source ? source->consumers.at(layer).get() : nullptr));
            auto& producer = *producers.back();
            auto& consumer = *consumers.back();
            producer.alias(true, 0, hidden[layer % 2], stem.outputs.at(0));
            for (unsigned index = 0; index < 3; ++index)
                producer.alias(false, index, qkv.at(index), attention.inputs.at(index));
            consumer.alias(true, 0, hidden[layer % 2], stem.outputs.at(0));
            consumer.alias(true, 1, attention_output, attention.outputs.at(0));
            consumer.alias(false, 0, hidden[(layer + 1) % 2], stem.outputs.at(0));
        }
        tail.reset(new NativeGraph(directory + "tail_native.rknn", RKNN_NPU_CORE_0_1_2, false,
                                   source ? source->tail.get() : nullptr));
        tail->alias(true, 0, hidden[0], stem.outputs.at(0));
        tail->bind(false, 0, tail->allocate(tail->outputs.at(0)));
    }

    void set_image(const std::vector<float>& image) { stem.set_image(image); }
    std::vector<float> read_output() { return tail->read_output(); }

    void run(bool parallel_heads = true, std::vector<std::vector<float>>* snapshots = nullptr,
             std::array<double, 5>* stage_times = nullptr) {
        auto timed = [&](unsigned stage, auto&& operation) {
            if (!stage_times) { operation(); return; }
            const auto start = std::chrono::steady_clock::now();
            operation();
            (*stage_times)[stage] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        };
        timed(0, [&] { stem.run(); });
        for (unsigned layer = 0; layer < 12; ++layer) {
            timed(1, [&] { producers[layer]->run(); });
            timed(2, [&] {
                if (!parallel_heads) { attention.run(); return; }
                std::vector<std::future<void>> pending;
                for (unsigned part = 0; part < 3; ++part)
                    pending.emplace_back(std::async(std::launch::async, [&, part] { parts[part]->run(); }));
                for (auto& task : pending) task.get();
            });
            timed(3, [&] { consumers[layer]->run(); });
            if (snapshots) snapshots->push_back(consumers[layer]->read_output(true));
        }
        timed(4, [&] { tail->run(); });
    }
};
