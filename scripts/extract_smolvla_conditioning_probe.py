#!/usr/bin/env python3
"""Extract real layer-1 conditioning graphs and equivalent GQA controls."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor


QUERY = "/Cast_16_output_0"
COMPACT = ["/Reshape_12_output_0", "/Reshape_14_output_0"]
EXPANDED = ["/Reshape_17_output_0", "/Reshape_18_output_0"]
OUTPUT = "/Reshape_19_output_0"
READY = ["/Transpose_5_output_0", "/Transpose_6_output_0"]


def dimensions(info):
    return [dim.dim_value for dim in info.type.tensor_type.shape.dim]


def native_interface(model, ready_names=READY, flat_widths=(), ready_heads=15):
    """Use token-as-channel, 64-wide rows to match producer/consumer padding."""
    result = copy.deepcopy(model)
    before, after = [], []
    for is_input, values, nodes in ((True, result.graph.input, before),
                                    (False, result.graph.output, after)):
        for index, info in enumerate(values):
            shape = dimensions(info)
            if len(shape) == 3 and (info.name == "suffix" or shape[-1] == 32):
                continue
            if len(shape) not in (3, 4) or shape[0] != 1:
                raise ValueError(f"Unsupported token-major interface: {shape}")
            length, width = shape[1], int(np.prod(shape[2:]))
            original = info.name
            name = f"native_{'input' if is_input else 'output'}_{index}"
            original_shape, native_shape = name + "_shape", name + "_4d"
            if original in ready_names:
                # Output width pads 149 -> 152. Expose that physical width on the
                # consumer and slice the padding before the unchanged attention.
                native_dimensions = [ready_heads, 64, 1, 152 if is_input else 149]
                restore = [1, ready_heads, 64, 149]
                result.graph.initializer.extend([
                    numpy_helper.from_array(np.array(restore, np.int64), original_shape),
                    numpy_helper.from_array(np.array(native_dimensions, np.int64), native_shape)])
                if is_input:
                    for label, value in (("starts", [0]), ("ends", [149]), ("axes", [3]), ("steps", [1])):
                        result.graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), name + "_" + label))
                    nodes.append(helper.make_node("Slice", [name, *[name + "_" + label for label in ("starts", "ends", "axes", "steps")]],
                                                  [name + "_trimmed"], name=name + "_trim"))
                    nodes.append(helper.make_node("Reshape", [name + "_trimmed", original_shape],
                                                  [original if ready_names.index(original) % 2 == 0 else name + "_kv"], name=name + "_restore"))
                    if ready_names.index(original) % 2 == 1:
                        nodes.append(helper.make_node("Transpose", [name + "_kv"], [original], perm=[0, 1, 3, 2], name=name + "_value_order"))
                else:
                    value = original
                    if ready_names.index(original) % 2 == 1:
                        value = name + "_kv"
                        nodes.append(helper.make_node("Transpose", [original], [value], perm=[0, 1, 3, 2], name=name + "_value_order"))
                    nodes.append(helper.make_node("Reshape", [value, native_shape], [name], name=name + "_restore"))
                info.name = name
                info.type.tensor_type.shape.ClearField("dim")
                for size in native_dimensions:
                    info.type.tensor_type.shape.dim.add().dim_value = size
                continue
            if width not in flat_widths and width % 64:
                raise ValueError("Expected 64-wide attention heads")
            native_dimensions = [1, length, 1, width] if width in flat_widths else [1, length, width // 64, 64]
            for constant_name, constant in ((original_shape, shape), (native_shape, native_dimensions)):
                result.graph.initializer.append(numpy_helper.from_array(np.array(constant, np.int64), constant_name))
            if is_input:
                nodes.append(helper.make_node("Reshape", [name, original_shape], [original], name=name + "_restore"))
            else:
                nodes.append(helper.make_node("Reshape", [original, native_shape], [name], name=name + "_restore"))
            info.name = name
            info.type.tensor_type.shape.ClearField("dim")
            for size in native_dimensions:
                info.type.tensor_type.shape.dim.add().dim_value = size
    old = list(result.graph.node)
    result.graph.ClearField("node")
    result.graph.node.extend(before + old + after)
    result.graph.ClearField("value_info")
    result = onnx.shape_inference.infer_shapes(result)
    onnx.checker.check_model(result)
    return result


def grouped_control(source, token_interleaved=False):
    """Known GQA packing control: concatenate shared queries along GEMM rows."""
    initializers = []
    nodes = []

    def const(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def node(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], name="grouped_" + output, **attributes))
        return output

    q = node("Reshape", [QUERY, const("q_shape", np.array([1, 50, 5, 3, 64], np.int64))], "q_split")
    q = node("Transpose", [q], "q_group_order", perm=[0, 2, 1, 3, 4] if token_interleaved else [0, 2, 3, 1, 4])
    q = node("Reshape", [q, const("q_packed_shape", np.array([5, 150, 64], np.int64))], "q_packed")
    transformed = []
    for name, original in zip(("k", "v"), COMPACT):
        value = node("Transpose", [original], name + "_group_order", perm=[0, 2, 1, 3])
        value = node("Reshape", [value, const(name + "_shape", np.array([5, 149, 64], np.int64))], name + "_grouped")
        transformed.append(value)
    key = node("Transpose", [transformed[0]], "key_transposed", perm=[0, 2, 1])
    scores = node("MatMul", [q, key], "scores")
    scores = node("Mul", [scores, const("scale", np.float32(0.125))], "scaled_scores")
    probabilities = node("Softmax", [scores], "probabilities", axis=-1)
    attended = node("MatMul", [probabilities, transformed[1]], "attended")
    attended = node("Reshape", [attended, const("attended_shape", np.array([1, 5, 50, 3, 64] if token_interleaved else [1, 5, 3, 50, 64], np.int64))], "attended_split")
    attended = node("Transpose", [attended], "attended_order", perm=[0, 2, 1, 3, 4] if token_interleaved else [0, 3, 1, 2, 4])
    node("Reshape", [attended, const("output_shape", np.array([1, 50, 960], np.int64))], OUTPUT)
    graph = helper.make_graph(nodes, "known_grouped_gqa_control",
                              [helper.make_tensor_value_info(QUERY, onnx.TensorProto.FLOAT, [1, 50, 15, 64]),
                               *[helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 149, 5, 64]) for name in COMPACT]],
                              [helper.make_tensor_value_info(OUTPUT, onnx.TensorProto.FLOAT, [1, 50, 960])], initializers)
    model = helper.make_model(graph, opset_imports=list(source.opset_import))
    model.ir_version = source.ir_version
    return onnx.shape_inference.infer_shapes(model)


def as_native(value):
    batch, length = value.shape[:2]
    return value.reshape(batch, length, -1, 64).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    variants = {
        "fused": ([QUERY, "key_1", "value_1"], [OUTPUT]),
        "prepare_compact": (["key_1", "value_1"], COMPACT),
        "prepare_expanded": (["key_1", "value_1"], EXPANDED),
        "warm_compact": ([QUERY, *COMPACT], [OUTPUT]),
        "warm_expanded": ([QUERY, *EXPANDED], [OUTPUT]),
        "prepare_ready": (["key_1", "value_1"], READY),
        "warm_ready": ([QUERY, *READY], [OUTPUT]),
    }
    from onnx.reference import ReferenceEvaluator

    scale_graph = Extractor(source).extract_model([], ["/Constant_207_output_0"])
    scale = ReferenceEvaluator(scale_graph).run(None, {})[0]
    mask = next(value for value in source.graph.initializer if value.name == "smolvla_additive_attention_mask_1")
    if not np.array_equal(scale, np.float32(0.125)) or np.any(numpy_helper.to_array(mask) != 0):
        raise ValueError("Grouped control requires the actual 1/8 scale and fully-visible cross mask")
    models = {name: Extractor(source).extract_model(inputs, outputs) for name, (inputs, outputs) in variants.items()}
    models["warm_grouped"] = grouped_control(source)
    prep = ReferenceEvaluator(models["prepare_compact"])
    raw = {name: np.load(args.fixtures / (name + ".npy")) for name in ("key_1", "value_1")}
    compact = prep.run(None, raw)
    expanded = ReferenceEvaluator(models["prepare_expanded"]).run(None, raw)
    ready = ReferenceEvaluator(models["prepare_ready"]).run(None, raw)
    report = {"scope": "Actual layer-1 ONNX subgraphs, checkpoint activations from a synthetic exported prefix; not a complete denoising loop",
              "source_onnx": str(args.onnx), "scale": float(scale), "cross_mask_all_zero": True,
              "expanded_vs_repeat_compact": [], "models": {}, "reference_controls": {}}
    for a, b in zip(compact, expanded):
        repeated = np.repeat(a, 3, axis=2)
        report["expanded_vs_repeat_compact"].append({"bitwise_equal": bool(np.array_equal(repeated.view(np.uint8), b.view(np.uint8))),
                                                    "max_abs": float(np.max(np.abs(repeated - b)))})
    for name, model in models.items():
        native = native_interface(model)
        onnx.save(native, args.output_dir / (name + ".onnx"))
        report["models"][name] = {"inputs": [dimensions(value) for value in native.graph.input],
                                  "outputs": [dimensions(value) for value in native.graph.output],
                                  "initializer_bytes": sum(len(value.raw_data) for value in model.graph.initializer)}
    for variant in range(2):
        query = np.load(args.fixtures / f"query_{variant}.npy")
        feeds = {QUERY: query, **raw, **dict(zip(COMPACT, compact)), **dict(zip(EXPANDED, expanded)), **dict(zip(READY, ready))}
        expected = ReferenceEvaluator(models["fused"]).run(None, {name: feeds[name] for name in variants["fused"][0]})[0]
        np.save(args.output_dir / f"query_{variant}.npy", as_native(query))
        np.save(args.output_dir / f"reference_{variant}.npy", as_native(expected))
        for name in ("fused", "warm_compact", "warm_expanded", "warm_grouped", "warm_ready"):
            needed = [value.name for value in models[name].graph.input]
            actual = ReferenceEvaluator(models[name]).run(None, {key: feeds[key] for key in needed})[0]
            native = onnx.load(args.output_dir / (name + ".onnx"))
            native_feed = {}
            for value, original in zip(native.graph.input, needed):
                if original in READY:
                    prepared = feeds[original] if original == READY[0] else feeds[original].transpose(0, 1, 3, 2)
                    padded = np.zeros((15, 64, 1, 152), np.float32)
                    padded[..., :149] = prepared.reshape(15, 64, 1, 149)
                    native_feed[value.name] = padded
                else:
                    native_feed[value.name] = as_native(feeds[original])
            native_result = ReferenceEvaluator(native).run(None, native_feed)[0]
            report["reference_controls"][f"{name}_query{variant}"] = {
                "vs_fused_max_abs": float(np.max(np.abs(actual - expected))),
                "native_wrapper_max_abs": float(np.max(np.abs(native_result - as_native(actual))))}
            if not np.allclose(actual, expected, atol=3e-6, rtol=3e-6) or not np.array_equal(native_result, as_native(actual)):
                raise ValueError(f"Reference control failed: {name}")
    for name, value in raw.items():
        np.save(args.output_dir / (name + ".npy"), as_native(value))
    for index, (a, b) in enumerate(zip(compact, expanded)):
        np.save(args.output_dir / f"compact_{index}.npy", as_native(a))
        np.save(args.output_dir / f"expanded_{index}.npy", as_native(b))
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
