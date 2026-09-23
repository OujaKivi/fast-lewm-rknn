#!/usr/bin/env python3
"""Validate one fixed-shape SmolVLA denoising step on RK3588 NPU."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def metrics(reference, actual):
    reference = reference.astype(np.float64).ravel()
    actual = actual.astype(np.float64).ravel()
    difference = np.abs(reference - actual)
    return {
        "cosine": float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual))),
        "mae": float(difference.mean()),
        "p99_abs_error": float(np.percentile(difference, 99)),
        "max_abs_error": float(difference.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    names = ["suffix"] + [
        f"{name}_{index}" for index in range(args.layers) for name in ("key", "value")
    ]
    inputs = [np.load(data_dir / f"{name}.npy").astype(np.float32) for name in names]
    data_formats = ["nchw"] * len(inputs)
    if args.layout == "nhwc":
        for index, value in enumerate(inputs):
            if value.ndim == 4:
                inputs[index] = value.transpose(0, 2, 3, 1).copy()
        data_formats = None
    converted_reference = np.load(data_dir / "reference.npy")
    original_reference = np.load(data_dir / "original_reference.npy")
    runtime = RKNNLite()
    if runtime.load_rknn(args.model) != 0:
        raise RuntimeError("RKNN load failed")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("RKNN runtime initialization failed")
    try:
        def infer():
            return runtime.inference(
                inputs=inputs, data_format=data_formats
            )[0]

        for _ in range(3):
            infer()
        elapsed = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            actual = infer()
            elapsed.append((time.perf_counter() - started) * 1e3)
    finally:
        runtime.release()

    if actual.shape != converted_reference.shape:
        raise RuntimeError(f"shape mismatch: {actual.shape} != {converted_reference.shape}")
    result = {
        "model": args.model,
        "input_count": len(inputs),
        "layout": args.layout,
        "layers": args.layers,
        "output_shape": list(actual.shape),
        "against_float32_reference": metrics(converted_reference, actual),
        "against_original_reference": metrics(original_reference, actual),
        "latency_ms_median": float(np.median(elapsed)),
        "latency_ms_p95": float(np.percentile(elapsed, 95)),
        "latency_ms_samples": [round(value, 3) for value in elapsed],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
