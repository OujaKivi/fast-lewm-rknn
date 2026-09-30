#!/usr/bin/env python3
"""Keep normalization flat and test spatial geometry only after that boundary."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.utils import Extractor

from export_smolvla_spatial_consumer import spatial_consumer
from extract_smolvla_vision_probes import native_interfaces, set_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    norm_nodes = [node for node in source.graph.node if node.op_type == "LayerNormalization"]
    if len(norm_nodes) != 1 or len(norm_nodes[0].output) != 1:
        raise ValueError("Expected exactly one normalization boundary")
    normalized, residual = norm_nodes[0].output[0], norm_nodes[0].input[0]
    prefix = Extractor(source).extract_model([value.name for value in source.graph.input], [normalized, residual])
    for value in prefix.graph.output:
        set_shape(value, [1, 1024, 768])
    prefix = native_interfaces(prefix, [], [0, 1])
    mlp = Extractor(source).extract_model([normalized, residual], [source.graph.output[0].name])
    for value in mlp.graph.input:
        set_shape(value, [1, 1024, 768])
    mlp = native_interfaces(mlp, [0, 1], [])
    onnx.save(prefix, args.output_dir / "consumer_prefix.onnx")
    onnx.save(mlp, args.output_dir / "mlp_original.onnx")
    rng = np.random.default_rng(20260930)
    inputs = [rng.normal(0, 0.3, (1, 768, 1, 1024)).astype(np.float32) for _ in source.graph.input]
    expected = ReferenceEvaluator(source).run(None, {value.name: array for value, array in zip(source.graph.input, inputs)})[0]
    prepared = ReferenceEvaluator(prefix).run(None, {value.name: array for value, array in zip(prefix.graph.input, inputs)})
    controls = [("mlp_original", mlp, 1, 1024)]
    controls.extend((f"mlp_spatial_{height}x{width}", spatial_consumer(mlp, height, width), height, width)
                    for height, width in ((1, 1024), (16, 64), (32, 32), (64, 16)))
    report = {"scope": "Same position/channel arithmetic, normalization stays rank-3/flat; prefix and MLP split then native spatial MLP; no NPU speed claim",
              "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
              "normalization_tensor": normalized, "residual_tensor": residual, "plans": {}}
    for name, graph, height, width in controls:
        output = ReferenceEvaluator(graph).run(None, {value.name: array.reshape(1, 768, height, width)
                                                     for value, array in zip(graph.graph.input, prepared)})[0].reshape(expected.shape)
        if not np.isfinite(output).all() or not np.array_equal(output.view(np.uint32), expected.view(np.uint32)):
            raise ValueError("FP32 split/geometry does not match this reference")
        path = args.output_dir / f"{name}.onnx"
        onnx.save(graph, path)
        report["plans"][name] = {"fp32_bitwise_equal": True, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
