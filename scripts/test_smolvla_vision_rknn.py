#!/usr/bin/env python3
"""Compare a SmolVLA vision RKNN graph with the exported PyTorch reference."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    image = np.load(args.input).astype(np.float32)
    reference = np.load(args.reference).astype(np.float32)
    runtime = RKNNLite()
    if runtime.load_rknn(args.model) != 0:
        raise RuntimeError("RKNN load failed")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("RKNN runtime initialization failed")
    try:
        def infer():
            return runtime.inference(inputs=[image], data_format=["nchw"])[0]

        for _ in range(3):
            infer()
        times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            actual = infer()
            times.append((time.perf_counter() - started) * 1e3)
    finally:
        runtime.release()

    if actual.shape != reference.shape:
        raise RuntimeError(f"shape mismatch: {actual.shape} != {reference.shape}")
    diff = np.abs(actual.astype(np.float32) - reference)
    flat_actual = actual.astype(np.float64).ravel()
    flat_reference = reference.astype(np.float64).ravel()
    cosine = np.dot(flat_actual, flat_reference) / (
        np.linalg.norm(flat_actual) * np.linalg.norm(flat_reference)
    )
    result = {
        "model": args.model,
        "input_shape": list(image.shape),
        "output_shape": list(actual.shape),
        "cosine": float(cosine),
        "mae": float(diff.mean()),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "max_abs_error": float(diff.max()),
        "latency_ms_median": float(np.median(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "latency_ms_samples": [round(value, 3) for value in times],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
