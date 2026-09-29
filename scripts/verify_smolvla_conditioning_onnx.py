#!/usr/bin/env python3
"""Check the full split graph in FP32 independently of RKNN compilation."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-onnx", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["reference", "onnxruntime"], default="reference")
    args = parser.parse_args()
    if args.backend == "onnxruntime":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        def evaluator(path):
            return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    else:
        evaluator = lambda path: ReferenceEvaluator(str(path))
    source = evaluator(args.original_onnx)
    prep_model = onnx.load(args.directory / "prepare_full.onnx")
    warm_model = onnx.load(args.directory / "warm_full.onnx")
    prep, warm = evaluator(args.directory / "prepare_full.onnx"), evaluator(args.directory / "warm_full.onnx")
    manifest = json.loads((args.directory / "manifest.json").read_text())
    raw = {f"{kind}_{layer}": np.load(args.fixture_dir / f"{kind}_{layer}.npy")
           for layer in range(32) for kind in ("key", "value")}
    suffix = np.load(args.fixture_dir / "suffix.npy")
    random = np.random.default_rng(812)
    reports = {}
    for variant in ("original", "other_suffix", "changed_prefix"):
        prefix = {key: (-value[:, :, ::-1].copy() if variant == "changed_prefix" and key.startswith("value") else value)
                  for key, value in raw.items()}
        action = suffix + np.float32(0.1) * random.standard_normal(suffix.shape, dtype=np.float32) if variant == "other_suffix" else suffix
        original = source.run(None, {"suffix": action, **prefix})[0]
        prepared = prep.run(None, {value.name: prefix[f"{kind}_{layer}"]
                                  for value, (layer, kind) in zip(prep_model.graph.input,
                                                                [(layer, kind) for layer in range(1, 32, 2) for kind in ("key", "value")])})
        ready = {}
        for name, value in zip(manifest["ready_source_tensors"], prepared):
            padded = np.zeros((15, 64, 1, 152), np.float32)
            padded[..., :149] = value
            ready[name] = padded
        feeds = {"suffix": action, **prefix, **ready}
        split = warm.run(None, {value.name: feeds[source_name]
                               for value, source_name in zip(warm_model.graph.input, manifest["warm_source_input_order"])})[0]
        if not np.isfinite(original).all() or not np.isfinite(split).all():
            raise ValueError("Non-finite FP32 reference")
        reports[variant] = {"max_abs": float(np.max(np.abs(original - split))),
                            "mean_abs": float(np.mean(np.abs(original - split))),
                            "bitwise_equal": bool(np.array_equal(original.view(np.uint32), split.view(np.uint32)))}
        if not np.allclose(original, split, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Full split graph mismatch: {variant}: {reports[variant]}")
    record = {"scope": "Full original vs split ONNX FP32 reference, three synthetic fixtures; no RKNN precision or performance inference",
              "backend": args.backend, "checks": reports}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
