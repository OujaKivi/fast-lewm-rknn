#!/usr/bin/env python3
"""Change only MLP token geometry inside one consumer graph, without new I/O cuts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def stage_spatial_consumer(source, height, width):
    if height < 1 or width < 1 or height * width != 1024:
        raise ValueError("Must retain all 1024 positions")
    model = copy.deepcopy(source)
    normalizations = [node for node in model.graph.node if node.op_type == "LayerNormalization"]
    fc2_biases = [node for node in model.graph.node if node.name.endswith("/mlp/fc2/Add")]
    if len(normalizations) != 1 or len(fc2_biases) != 1:
        raise ValueError("Expected one normalization and final MLP bias")
    norm_output, mlp_output = normalizations[0].output[0], fc2_biases[0].output[0]
    axis = next((attr.i for attr in normalizations[0].attribute if attr.name == "axis"), -1)
    norm_consumers = [node for node in model.graph.node if norm_output in node.input]
    mlp_consumers = [node for node in model.graph.node if mlp_output in node.input]
    if axis != -1 or len(norm_consumers) != 1 or not norm_consumers[0].name.endswith("/mlp/fc1/MatMul"):
        raise ValueError("Expected channel normalization consumed only by FC1")
    if len(mlp_consumers) != 1 or mlp_consumers[0].op_type != "Add":
        raise ValueError("Expected one final residual consumer")
    spatial_norm, flat_mlp = "__stage_spatial_norm", "__stage_flat_mlp"
    if any(value.name.startswith("__stage_") for value in model.graph.initializer):
        raise ValueError("Stage geometry was already rewritten")
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([1, height, width, 768], dtype=np.int64), "__stage_spatial_shape"),
        numpy_helper.from_array(np.array([1, 1024, 768], dtype=np.int64), "__stage_flat_shape"),
    ])
    nodes = []
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == norm_output:
                node.input[index] = spatial_norm
            elif name == mlp_output:
                node.input[index] = flat_mlp
        nodes.append(node)
        if node is normalizations[0]:
            nodes.append(helper.make_node("Reshape", [norm_output, "__stage_spatial_shape"], [spatial_norm],
                                          name="__stage_norm_to_spatial"))
        if node is fc2_biases[0]:
            nodes.append(helper.make_node("Reshape", [mlp_output, "__stage_flat_shape"], [flat_mlp],
                                          name="__stage_mlp_to_flat"))
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.ClearField("value_info")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.load(args.onnx)
    rng = np.random.default_rng(20260930)
    inputs = [rng.normal(0, 0.3, (1, 768, 1, 1024)).astype(np.float32) for _ in source.graph.input]
    expected = ReferenceEvaluator(source).run(None, dict(zip([value.name for value in source.graph.input], inputs)))[0]
    if not np.isfinite(expected).all():
        raise ValueError("Invalid original reference")
    report = {"scope": "One graph; original inputs, normalization and residual geometry; only position-separable MLP geometry changes; no new intermediate storage precision",
              "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(), "plans": {}}
    for height, width in ((1, 1024), (16, 64), (32, 32), (64, 16)):
        graph = stage_spatial_consumer(source, height, width)
        output = ReferenceEvaluator(graph).run(None, dict(zip([value.name for value in graph.graph.input], inputs)))[0]
        if not np.isfinite(output).all() or not np.array_equal(output.view(np.uint32), expected.view(np.uint32)):
            raise ValueError("FP32 in-graph geometry does not match this reference")
        name = f"consumer_stage_spatial_{height}x{width}"
        path = args.output_dir / f"{name}.onnx"
        onnx.save(graph, path)
        report["plans"][name] = {"fp32_bitwise_equal": True, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
