#!/usr/bin/env python3
"""Refactor only independent token axes of a native visual consumer."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from extract_smolvla_vision_probes import set_shape


def spatial_consumer(source, height, width):
    if height < 1 or width < 1 or height * width != 1024:
        raise ValueError("Must retain all 1024 positions")
    model = copy.deepcopy(source)
    if len(model.graph.input) != 2 or len(model.graph.output) != 1:
        raise ValueError("Expected hidden/attention input and one output")
    for value in [*model.graph.input, *model.graph.output]:
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if len(dims) != 4 or dims[0] not in (0, 1) or dims[1:] != [768, 1, 1024]:
            raise ValueError("Unexpected consumer shape")
        set_shape(value, [1, 768, height, width])
    nodes = []
    changed = 0
    for node in model.graph.node:
        if node.name in ("__native_in_0_squeeze", "__native_in_1_squeeze", "__native_out_0_unsqueeze"):
            continue
        if node.name in ("__native_in_0_transpose", "__native_in_1_transpose"):
            index = int(node.name.split("_")[-2])
            node.input[0] = model.graph.input[index].name
            node.attribute[0].ints[:] = [0, 2, 3, 1]
            changed += 1
        elif node.name == "__native_out_0_transpose":
            node.output[0] = model.graph.output[0].name
            node.attribute[0].ints[:] = [0, 3, 1, 2]
            changed += 1
        elif node.op_type not in {"Add", "Mul", "Constant", "MatMul", "LayerNormalization", "Tanh"}:
            raise ValueError(f"Not a proven position-separable operation: {node.op_type}")
        if node.op_type == "LayerNormalization":
            axis = next(attr.i for attr in node.attribute if attr.name == "axis")
            if axis != -1:
                raise ValueError("Normalization must stay on the unchanged channel axis")
        nodes.append(node)
    if changed != 3:
        raise ValueError("Native interface structure changed")
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.ClearField("value_info")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    if [value.SerializeToString() for value in model.graph.initializer] != [value.SerializeToString() for value in source.graph.initializer]:
        raise ValueError("Initializer changed")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.load(args.onnx)
    onnx.save(source, args.output_dir / "consumer_original_recompiled.onnx")
    generator = np.random.default_rng(20260930)
    inputs = [generator.normal(0, 0.3, (1, 768, 1, 1024)).astype(np.float32) for _ in source.graph.input]
    original = ReferenceEvaluator(source).run(None, {value.name: array for value, array in zip(source.graph.input, inputs)})[0]
    if original.shape != (1, 768, 1, 1024) or not np.isfinite(original).all():
        raise ValueError("Invalid original FP32 reference")
    report = {"scope": "Same parameters, 1024 independent positions, channel reductions and nonlinearities; only native spatial factoring changes; not an NPU speed or quality claim",
              "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
              "initializer_bytes": sum(len(value.raw_data) for value in source.graph.initializer), "plans": {}}
    for height, width in ((1, 1024), (16, 64), (32, 32), (64, 16)):
        candidate = spatial_consumer(source, height, width)
        values = {value.name: array.reshape(1, 768, height, width) for value, array in zip(candidate.graph.input, inputs)}
        output = ReferenceEvaluator(candidate).run(None, values)[0].reshape(original.shape)
        difference = np.abs(original - output)
        if not np.isfinite(output).all() or not np.allclose(original, output, atol=1e-5, rtol=1e-5):
            raise ValueError("FP32 spatial factoring failed")
        name = f"consumer_spatial_{height}x{width}"
        path = args.output_dir / f"{name}.onnx"
        onnx.save(candidate, path)
        report["plans"][name] = {"shape": [1, 768, height, width], "fp32_max_abs": float(difference.max()),
                                  "fp32_mae": float(difference.mean()), "fp32_bitwise_equal": bool(np.array_equal(original.view(np.uint32), output.view(np.uint32))),
                                  "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
