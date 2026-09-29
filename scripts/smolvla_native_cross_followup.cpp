// Reuse the measured boundary protocol; add numerical ablations and head controls.
#include "smolvla_native_cross_boundary.cpp"
#include <array>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

static std::string follow_path(const std::string& directory, const std::string& name, bool controlled = false) {
    return directory + "/" + name + (controlled && std::getenv("SMOLVLA_FOLLOWUP_OPT3") ? "_opt3" : "") + ".rknn";
}

static void view(NativeGraph& target_graph, bool target_input, unsigned target_index,
                 NativeGraph& source_graph, bool source_input, unsigned source_index,
                 unsigned divisions = 1, unsigned part = 0, unsigned divided_dimension = 1) {
    const auto& target = (target_input ? target_graph.inputs : target_graph.outputs).at(target_index);
    const auto& source = (source_input ? source_graph.inputs : source_graph.outputs).at(source_index);
    if (!divisions || part >= divisions || target.n_dims != source.n_dims || target.type != source.type ||
        target.fmt != source.fmt || target.h_stride != source.h_stride ||
        target.size_with_stride * divisions != source.size_with_stride ||
        (target.w_stride ? target.w_stride : target.dims[3]) != (source.w_stride ? source.w_stride : source.dims[3]))
    {
        std::ostringstream error;
        error << "Head view stride/layout mismatch target=" << target.name << " fmt=" << target.fmt
              << " bytes=" << target.size_with_stride << " ws=" << target.w_stride << " hs=" << target.h_stride
              << " source=" << source.name << " fmt=" << source.fmt << " bytes=" << source.size_with_stride
              << " ws=" << source.w_stride << " hs=" << source.h_stride << " divisions=" << divisions;
        throw std::runtime_error(error.str());
    }
    for (unsigned dim = 0; dim < target.n_dims; ++dim)
        if (target.dims[dim] * (dim == divided_dimension ? divisions : 1) != source.dims[dim])
            throw std::runtime_error("Head view shape mismatch");
    auto* buffer = (source_input ? source_graph.input_memory : source_graph.output_memory).at(source_index);
    auto* memory = rknn_create_mem_from_fd(target_graph.context, buffer->fd, buffer->virt_addr,
                                         target.size_with_stride, part * target.size_with_stride);
    if (!memory) throw std::runtime_error("Head FD view failed");
    target_graph.owned.push_back(memory);
    target_graph.bind(target_input, target_index, memory);
}

static void group_view(NativeGraph& target_graph, bool target_input, unsigned target_index,
                       NativeGraph& source_graph, bool source_input, unsigned source_index,
                       unsigned groups, unsigned first, unsigned dimension) {
    const auto& target = (target_input ? target_graph.inputs : target_graph.outputs).at(target_index);
    const auto& source = (source_input ? source_graph.inputs : source_graph.outputs).at(source_index);
    if (!groups || first + groups > 5 || target.fmt != RKNN_TENSOR_NC1HWC2 || source.fmt != target.fmt ||
        target.n_dims != 5 || source.n_dims != 5 || target.type != source.type ||
        target.h_stride != source.h_stride || source.size_with_stride % 5 ||
        target.size_with_stride != source.size_with_stride / 5 * groups ||
        (target.w_stride ? target.w_stride : target.dims[3]) != (source.w_stride ? source.w_stride : source.dims[3]))
        throw std::runtime_error("Compact group view layout mismatch");
    for (unsigned dim = 0; dim < 5; ++dim)
        if (dim == dimension ? source.dims[dim] % 5 || target.dims[dim] != source.dims[dim] / 5 * groups :
                               target.dims[dim] != source.dims[dim])
            throw std::runtime_error("Compact group view shape mismatch");
    auto* buffer = (source_input ? source_graph.input_memory : source_graph.output_memory).at(source_index);
    auto* memory = rknn_create_mem_from_fd(target_graph.context, buffer->fd, buffer->virt_addr,
                                         target.size_with_stride, source.size_with_stride / 5 * first);
    if (!memory) throw std::runtime_error("Compact group FD view failed");
    target_graph.owned.push_back(memory);
    target_graph.bind(target_input, target_index, memory);
}

class HeadWorkers {
    std::array<std::thread, 3> threads;
    std::array<std::exception_ptr, 3> errors{};
    std::mutex mutex;
    std::condition_variable available, finished;
    unsigned generation = 0, completed = 0;
    bool stop = false;
    const bool trace = std::getenv("SMOLVLA_CROSS_TRACE") != nullptr;
    Clock::time_point dispatch;
public:
    std::array<double, 3> wake_ms{}, run_ms{};
    explicit HeadWorkers(const std::array<std::unique_ptr<NativeGraph>, 3>& graphs) {
        try {
            for (unsigned index = 0; index < 3; ++index) {
                auto* graph = graphs[index].get();
                threads[index] = std::thread([this, graph, index] {
                    unsigned seen = 0;
                    std::unique_lock<std::mutex> lock(mutex);
                    while (true) {
                        available.wait(lock, [&] { return stop || generation != seen; });
                        if (stop) return;
                        seen = generation;
                        const auto dispatched = dispatch;
                        lock.unlock();
                        const auto started = trace ? Clock::now() : Clock::time_point{};
                        std::exception_ptr error;
                        try { graph->run(); } catch (...) { error = std::current_exception(); }
                        const auto ended = trace ? Clock::now() : Clock::time_point{};
                        lock.lock();
                        if (trace) {
                            wake_ms[index] = std::chrono::duration<double, std::milli>(started - dispatched).count();
                            run_ms[index] = std::chrono::duration<double, std::milli>(ended - started).count();
                        }
                        errors[index] = error;
                        ++completed;
                        finished.notify_one();
                    }
                });
            }
        } catch (...) { shutdown(); throw; }
    }
    void shutdown() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        available.notify_all();
        for (auto& thread : threads) if (thread.joinable()) thread.join();
    }
    ~HeadWorkers() { shutdown(); }
    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        completed = 0;
        ++generation;
        if (trace) dispatch = Clock::now();
        available.notify_all();
        finished.wait(lock, [&] { return completed == 3; });
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
    }
};

class CrossFollowup {
public:
    const bool tracing = std::getenv("SMOLVLA_CROSS_TRACE") != nullptr;
    std::vector<std::array<double, 11>> stage_trace;
    CrossBoundary base;
    NativeGraph restore_o, original_q, both, producer, consumer, full;
    std::array<std::unique_ptr<NativeGraph>, 3> heads;
    std::unique_ptr<HeadWorkers> workers;
    NativeGraph group_producer, group_consumer, group_full;
    std::array<std::unique_ptr<NativeGraph>, 3> group_heads;
    std::unique_ptr<HeadWorkers> group_workers;
    NativeGraph original_order_full;
    std::array<std::unique_ptr<NativeGraph>, 3> original_order_heads;
    std::unique_ptr<HeadWorkers> original_order_workers;

    CrossFollowup(const std::string& directory, const std::string& original)
        : base(original, RKNN_NPU_CORE_0_1_2),
          restore_o(directory + "/restore_o.rknn", RKNN_NPU_CORE_0_1_2),
          original_q(directory + "/original_q.rknn", RKNN_NPU_CORE_0_1_2),
          both(directory + "/original_q_restore_o.rknn", RKNN_NPU_CORE_0_1_2),
          producer(follow_path(directory, "head_producer", true), RKNN_NPU_CORE_0_1_2),
          consumer(follow_path(directory, "head_consumer", true), RKNN_NPU_CORE_0_1_2),
          full(follow_path(directory, "head_attention_15", true), RKNN_NPU_CORE_0_1_2),
          group_producer(follow_path(directory, "group_producer", true), RKNN_NPU_CORE_0_1_2),
          group_consumer(directory + "/group_consumer.rknn", RKNN_NPU_CORE_0_1_2),
          group_full(follow_path(directory, "group_attention_5", true), RKNN_NPU_CORE_0_1_2),
          original_order_full(directory + "/original_order_attention_5.rknn", RKNN_NPU_CORE_0_1_2) {
        for (auto* graph : {&restore_o, &original_q, &both, &producer, &consumer}) share(*graph, 0, base.original, true, 0);
        for (unsigned index = 0; index < 2; ++index) {
            for (auto* graph : {&restore_o, &original_q, &both}) share(*graph, index + 1, base.group_prepare, false, index);
            view(full, true, index + 1, base.prep_ready, false, index);
        }
        view(full, true, 0, producer, false, 0);
        view(full, false, 0, consumer, true, 1);
        for (unsigned part = 0; part < 3; ++part) {
            heads[part] = std::make_unique<NativeGraph>(follow_path(directory, "head_attention_5", true),
                                                       static_cast<rknn_core_mask>(1u << part));
            view(*heads[part], true, 0, producer, false, 0, 3, part, 1);
            view(*heads[part], false, 0, consumer, true, 1, 3, part, 1);
            for (unsigned index = 0; index < 2; ++index)
                view(*heads[part], true, index + 1, base.prep_ready, false, index, 3, part, 0);
        }
        workers = std::make_unique<HeadWorkers>(heads);
        share(group_producer, 0, base.original, true, 0);
        share(group_consumer, 0, base.original, true, 0);
        view(group_full, true, 0, group_producer, false, 0);
        view(group_full, false, 0, group_consumer, true, 1);
        for (unsigned index = 0; index < 2; ++index) view(group_full, true, index + 1, base.group_prepare, false, index);
        for (unsigned part = 0; part < 3; ++part) {
            const unsigned groups = part < 2 ? 2 : 1;
            group_heads[part] = std::make_unique<NativeGraph>(follow_path(directory, "group_attention_" + std::to_string(groups), true),
                                                            static_cast<rknn_core_mask>(1u << part));
            group_view(*group_heads[part], true, 0, group_producer, false, 0, groups, part * 2, 1);
            group_view(*group_heads[part], false, 0, group_consumer, true, 1, groups, part * 2, 1);
            for (unsigned index = 0; index < 2; ++index)
                group_view(*group_heads[part], true, index + 1, base.group_prepare, false, index, groups, part * 2, 0);
        }
        group_workers = std::make_unique<HeadWorkers>(group_heads);
        view(original_order_full, true, 0, producer, false, 0);
        view(original_order_full, false, 0, consumer, true, 1);
        for (unsigned index = 0; index < 2; ++index) view(original_order_full, true, index + 1, base.group_prepare, false, index);
        for (unsigned part = 0; part < 3; ++part) {
            const unsigned groups = part < 2 ? 2 : 1;
            original_order_heads[part] = std::make_unique<NativeGraph>(directory + "/original_order_attention_" + std::to_string(groups) + ".rknn",
                                                                     static_cast<rknn_core_mask>(1u << part));
            group_view(*original_order_heads[part], true, 0, producer, false, 0, groups, part * 2, 1);
            group_view(*original_order_heads[part], false, 0, consumer, true, 1, groups, part * 2, 1);
            for (unsigned index = 0; index < 2; ++index)
                group_view(*original_order_heads[part], true, index + 1, base.group_prepare, false, index, groups, part * 2, 0);
        }
        original_order_workers = std::make_unique<HeadWorkers>(original_order_heads);
    }

    void run(unsigned mode, const float* hidden, unsigned count, float* output, double* times) {
        if (mode > 18 || !count || !base.prefix_ready) throw std::runtime_error("Invalid followup run");
        stage_trace.clear();
        if (mode == 8) { base.run(0, hidden, count, output, times); return; }
        if (mode < 2) { base.run(mode ? 7 : 1, hidden, count, output, times); return; }
        if (tracing) stage_trace.resize(count);
        std::fill(times, times + 8, 0.0);
        const auto start = Clock::now();
        auto tick = Clock::now();
        (mode < 5 || mode >= 9 ? base.group_prepare : base.prep_ready).run();
        times[0] = elapsed(tick);
        auto* graph = mode == 2 ? &restore_o : mode == 3 ? &original_q : &both;
        const bool bit_shuffle = mode >= 15;
        const bool bit_pack = mode >= 17;
        const bool propagated_group = (mode >= 9 && mode <= 11) || (bit_shuffle && !bit_pack);
        for (unsigned index = 0; index < count; ++index) {
            tick = Clock::now();
            pack(base.original, 0, hidden + index * 50 * 480, 50, 1, 480);
            times[1] += elapsed(tick);
            if (mode < 5) {
                tick = Clock::now(); graph->run(); times[2] += elapsed(tick);
            } else {
                tick = Clock::now(); (propagated_group ? group_producer : producer).run(); times[3] += elapsed(tick);
                if (tracing) stage_trace[index][0] = elapsed(tick);
                if (bit_pack) {
                    tick = Clock::now(); copy_native_tiles(true); times[2] += elapsed(tick);
                    if (tracing) stage_trace[index][1] = elapsed(tick);
                }
                tick = Clock::now();
                if (mode == 5) full.run();
                else if (mode == 6) for (auto& head : heads) head->run();
                else if (mode == 7) workers->run();
                else if (mode == 9) group_full.run();
                else if (mode == 10) for (auto& head : group_heads) head->run();
                else if (mode == 11) group_workers->run();
                else if (mode == 12) original_order_full.run();
                else if (mode == 13) for (auto& head : original_order_heads) head->run();
                else if (mode == 14) original_order_workers->run();
                else if (mode == 15) group_workers->run();
                else if (mode == 16) group_full.run();
                else if (mode == 17) group_workers->run();
                else group_full.run();
                times[4] += elapsed(tick);
                if (tracing) {
                    stage_trace[index][2] = elapsed(tick);
                    auto* traced_workers = mode == 7 ? workers.get() : (mode == 11 || mode == 15 || mode == 17) ? group_workers.get() : nullptr;
                    if (traced_workers) for (unsigned head = 0; head < 3; ++head) {
                        stage_trace[index][5 + head] = traced_workers->wake_ms[head];
                        stage_trace[index][8 + head] = traced_workers->run_ms[head];
                    }
                }
                if (bit_shuffle) {
                    tick = Clock::now(); copy_native_tiles(false); times[2] += elapsed(tick);
                    if (tracing) stage_trace[index][3] = elapsed(tick);
                }
                tick = Clock::now(); (propagated_group && !bit_shuffle ? group_consumer : consumer).run(); times[5] += elapsed(tick);
                if (tracing) stage_trace[index][4] = elapsed(tick);
            }
        }
        tick = Clock::now();
        const auto values = (mode < 5 ? *graph : propagated_group && !bit_shuffle ? group_consumer : consumer).read_output();
        if (values.size() != 50 * 480) throw std::runtime_error("Invalid followup output");
        for (unsigned index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index])) throw std::runtime_error("Nonfinite followup output");
            output[index] = values[index];
        }
        times[6] = elapsed(tick);
        times[7] = elapsed(start);
    }

    void copy_native_tiles(bool pack_groups) {
        const auto& source = pack_groups ? group_producer.outputs.at(0) : group_consumer.inputs.at(1);
        const auto& target = pack_groups ? producer.outputs.at(0) : consumer.inputs.at(1);
        const unsigned source_dims[] = {1, 40, 1, 152, 8}, target_dims[] = {1, 120, 1, 52, 8};
        if (source.fmt != RKNN_TENSOR_NC1HWC2 || target.fmt != source.fmt || source.n_dims != 5 || target.n_dims != 5 ||
            source.type != RKNN_TENSOR_FLOAT16 || target.type != source.type || source.h_stride || target.h_stride ||
            (source.w_stride && source.w_stride != 152) || (target.w_stride && target.w_stride != 52) ||
            source.size_with_stride != 5 * 64 * 152 * sizeof(__fp16) ||
            target.size_with_stride != 15 * 64 * 52 * sizeof(__fp16))
            throw std::runtime_error("Unsupported exact native shuffle format");
        for (unsigned dim = 0; dim < 5; ++dim)
            if (source.dims[dim] != source_dims[dim] || target.dims[dim] != target_dims[dim])
                throw std::runtime_error("Unsupported exact native shuffle shape");
        auto* group_memory = pack_groups ? group_producer.output_memory.at(0) : group_consumer.input_memory.at(1);
        auto* head_memory = pack_groups ? producer.output_memory.at(0) : consumer.input_memory.at(1);
        auto* from = pack_groups ? head_memory : group_memory;
        auto* to = pack_groups ? group_memory : head_memory;
        const auto from_context = pack_groups ? producer.context : group_consumer.context;
        const auto to_context = pack_groups ? group_producer.context : consumer.context;
        native_check(rknn_mem_sync(from_context, from, RKNN_MEMORY_SYNC_FROM_DEVICE), "shuffle source sync");
        const auto* input = static_cast<const char*>(from->virt_addr);
        auto* output = static_cast<char*>(to->virt_addr);
        // C8 lanes already agree. Move 120 contiguous 50-row tiles, without arithmetic or FP conversion.
        for (unsigned group = 0; group < 5; ++group)
            for (unsigned block = 0; block < 8; ++block)
                for (unsigned head = 0; head < 3; ++head) {
                    const unsigned start = ((group * 8 + block) * 152 + head * 50) * 8 * sizeof(__fp16);
                    const unsigned finish = ((group * 3 + head) * 8 + block) * 52 * 8 * sizeof(__fp16);
                    std::memcpy(output + (pack_groups ? start : finish), input + (pack_groups ? finish : start),
                                50 * 8 * sizeof(__fp16));
                }
        native_check(rknn_mem_sync(to_context, to, RKNN_MEMORY_SYNC_TO_DEVICE), "shuffle target sync");
    }

    std::string information() {
        std::ostringstream result;
        result << "{\"base\":" << base.information() << ",\"new_graphs\":{";
        const char* names[] = {"restore_o", "original_q", "both", "head_producer", "head_consumer", "head_full", "head_0", "head_1", "head_2",
                              "group_producer", "group_consumer", "group_full", "group_0", "group_1", "group_2",
                              "original_order_full", "original_order_0", "original_order_1", "original_order_2"};
        NativeGraph* graphs[] = {&restore_o, &original_q, &both, &producer, &consumer, &full, heads[0].get(), heads[1].get(), heads[2].get(),
                                &group_producer, &group_consumer, &group_full, group_heads[0].get(), group_heads[1].get(), group_heads[2].get(),
                                &original_order_full, original_order_heads[0].get(), original_order_heads[1].get(), original_order_heads[2].get()};
        for (unsigned ordinal = 0; ordinal < 19; ++ordinal) {
            auto& graph = *graphs[ordinal];
            rknn_mem_size size{};
            native_check(rknn_query(graph.context, RKNN_QUERY_MEM_SIZE, &size, sizeof(size)), "followup memory query");
            result << (ordinal ? "," : "") << "\"" << names[ordinal] << "\":{\"weight_bytes\":" << size.total_weight_size
                   << ",\"internal_bytes\":" << size.total_internal_size << ",\"edges\":[";
            bool first = true;
            for (bool input : {true, false}) for (const auto& attr : (input ? graph.inputs : graph.outputs)) {
                result << (first ? "" : ",") << "{\"input\":" << (input ? "true" : "false") << ",\"index\":" << attr.index
                       << ",\"bytes\":" << attr.size_with_stride << ",\"w_stride\":" << attr.w_stride
                       << ",\"h_stride\":" << attr.h_stride << ",\"format\":\"" << get_format_string(attr.fmt) << "\",\"dims\":[";
                for (unsigned dim = 0; dim < attr.n_dims; ++dim) result << (dim ? "," : "") << attr.dims[dim];
                result << "]}";
                first = false;
            }
            result << "]}";
        }
        result << "}}";
        return result.str();
    }
};

extern "C" {
void* cross_followup_create(const char* directory, const char* base) {
    try { last_error.clear(); return new CrossFollowup(directory, base); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void cross_followup_destroy(void* handle) { delete static_cast<CrossFollowup*>(handle); }
int cross_followup_prefix(void* handle, const float* key, const float* value) {
    try { static_cast<CrossFollowup*>(handle)->base.set_prefix(key, value); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int cross_followup_run(void* handle, unsigned mode, const float* hidden, unsigned count, float* output, double* times) {
    try { static_cast<CrossFollowup*>(handle)->run(mode, hidden, count, output, times); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
const char* cross_followup_info(void* handle) {
    static thread_local std::string value;
    try { value = static_cast<CrossFollowup*>(handle)->information(); return value.c_str(); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
const char* cross_followup_trace(void* handle) {
    static thread_local std::string value;
    std::ostringstream result;
    result << "[";
    bool first = true;
    for (const auto& row : static_cast<CrossFollowup*>(handle)->stage_trace) {
        result << (first ? "" : ",") << "[";
        for (unsigned index = 0; index < row.size(); ++index) result << (index ? "," : "") << row[index];
        result << "]";
        first = false;
    }
    result << "]";
    value = result.str();
    return value.c_str();
}
}
