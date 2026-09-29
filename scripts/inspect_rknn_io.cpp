#include "rknn_native_graph.hpp"
#include <iostream>

static void print_attr(const rknn_tensor_attr& attr) {
    std::cout << "{\"name\":\"" << attr.name << "\",\"format\":\"" << get_format_string(attr.fmt)
              << "\",\"dtype\":\"" << get_type_string(attr.type) << "\",\"dims\":[";
    for (unsigned dim = 0; dim < attr.n_dims; ++dim) std::cout << (dim ? "," : "") << attr.dims[dim];
    std::cout << "],\"size\":" << attr.size << ",\"size_with_stride\":" << attr.size_with_stride
              << ",\"w_stride\":" << attr.w_stride << ",\"h_stride\":" << attr.h_stride << "}";
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        NativeGraph graph(argv[1], RKNN_NPU_CORE_0_1_2, false);
        std::cout << "{\"inputs\":[";
        for (unsigned i = 0; i < graph.inputs.size(); ++i) {
            rknn_tensor_attr attr{}; attr.index = i;
            native_check(rknn_query(graph.context, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "logical input");
            std::cout << (i ? "," : "") << "{\"logical\":"; print_attr(attr);
            std::cout << ",\"native\":"; print_attr(graph.inputs[i]); std::cout << "}";
        }
        std::cout << "],\"outputs\":[";
        for (unsigned i = 0; i < graph.outputs.size(); ++i) {
            rknn_tensor_attr attr{}; attr.index = i;
            native_check(rknn_query(graph.context, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "logical output");
            std::cout << (i ? "," : "") << "{\"logical\":"; print_attr(attr);
            std::cout << ",\"native\":"; print_attr(graph.outputs[i]); std::cout << "}";
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
