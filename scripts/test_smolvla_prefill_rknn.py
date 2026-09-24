#!/usr/bin/env python3
"""Check all RK3588 prefill K/V outputs against a PyTorch fixture."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def metrics(reference, actual):
    left = reference.astype(np.float64).ravel()
    right = actual.astype(np.float64).ravel()
    difference = np.abs(left - right)
    return {
        "cosine": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
    }


def match_layout(output, reference):
    if output.shape == reference.shape:
        return output, "native"
    for axes, label in (((0, 2, 3, 1), "nchw_to_nhwc"), ((0, 3, 1, 2), "nhwc_to_nchw")):
        candidate = output.transpose(axes)
        if candidate.shape == reference.shape:
            return candidate, label
    raise RuntimeError(f"Unexpected cache shape {output.shape}, expected {reference.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    prefix = np.load(data_dir / "prefix.npy").astype(np.float32)
    names = [f"{kind}_{layer}" for layer in range(args.layers) for kind in ("key", "value")]
    references = [np.load(data_dir / f"{name}_reference.npy") for name in names]
    runtime = RKNNLite()
    if runtime.load_rknn(args.model) != 0:
        raise RuntimeError("RKNN load failed")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("RKNN runtime initialization failed")
    try:
        for _ in range(2):
            runtime.inference(inputs=[prefix], data_format=None)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            outputs = runtime.inference(inputs=[prefix], data_format=None)
            samples.append((time.perf_counter() - started) * 1000)
    finally:
        runtime.release()
    if len(outputs) != len(references):
        raise RuntimeError(f"Expected {len(references)} cache outputs, got {len(outputs)}")
    per_output = {}
    for name, output, reference in zip(names, outputs, references, strict=True):
        output, layout = match_layout(output, reference)
        per_output[name] = {**metrics(reference, output), "layout": layout}
        np.save(data_dir / f"{name}_rknn.npy", output)
    result = {
        "layers": args.layers,
        "outputs": len(outputs),
        "minimum_cosine": min(item["cosine"] for item in per_output.values()),
        "maximum_mae": max(item["mae"] for item in per_output.values()),
        "maximum_abs": max(item["max_abs"] for item in per_output.values()),
        "latency_ms_median": float(np.median(samples)),
        "latency_ms_samples": samples,
        "per_output": per_output,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_output"}, indent=2))
    if result["minimum_cosine"] < 0.999 or result["maximum_mae"] > 0.01:
        raise RuntimeError("RKNN prefill cache parity failed")


if __name__ == "__main__":
    main()
