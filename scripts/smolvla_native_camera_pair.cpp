// Expose the already-tested fused camera-parallel control to the full policy.
#include "rknn_native_graph.hpp"
#include <chrono>
#include <future>
#include <memory>

using PairClock = std::chrono::steady_clock;
static thread_local std::string pair_error;
static double pair_elapsed(PairClock::time_point start) {
    return std::chrono::duration<double, std::milli>(PairClock::now() - start).count();
}

struct CameraPair {
    NativeGraph left, right;
    explicit CameraPair(const std::string& path)
        : left(path, RKNN_NPU_CORE_0), right(path, RKNN_NPU_CORE_1, true, &left) {}

    void run(unsigned parallel, const float* images, size_t count, float* outputs, size_t output_count, double* times) {
        if (parallel > 1 || count != 2 * 3 * 512 * 512 || output_count != 2 * 64 * 960)
            throw std::runtime_error("Expected two complete camera inputs and connector outputs");
        auto started = PairClock::now();
        const size_t image_size = count / 2;
        left.set_image(std::vector<float>(images, images + image_size));
        right.set_image(std::vector<float>(images + image_size, images + count));
        times[0] = pair_elapsed(started);
        started = PairClock::now();
        if (parallel) {
            auto future = std::async(std::launch::async, [&] { left.run(); });
            right.run();
            future.get();
        } else {
            left.run(); right.run();
        }
        times[1] = pair_elapsed(started);
        started = PairClock::now();
        const auto first = left.read_output(), second = right.read_output();
        if (first.size() != output_count / 2 || second.size() != output_count / 2)
            throw std::runtime_error("Unexpected connector output size");
        std::copy(first.begin(), first.end(), outputs);
        std::copy(second.begin(), second.end(), outputs + first.size());
        times[2] = pair_elapsed(started);
    }
};

extern "C" {
void* camera_pair_create(const char* path) {
    try { return new CameraPair(path); }
    catch (const std::exception& error) { pair_error = error.what(); return nullptr; }
}
void camera_pair_destroy(void* handle) { delete static_cast<CameraPair*>(handle); }
const char* camera_pair_error() { return pair_error.c_str(); }
int camera_pair_run(void* handle, unsigned parallel, const float* images, size_t count, float* outputs, size_t output_count, double* times) {
    try { static_cast<CameraPair*>(handle)->run(parallel, images, count, outputs, output_count, times); return 0; }
    catch (const std::exception& error) { pair_error = error.what(); return -1; }
}
}
