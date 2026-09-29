#!/usr/bin/env python3
"""Measure host KV repacking, not cache residency inside the RKNN runtime."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = np.load(args.data_dir / "suffix.npy")
    cache = [np.load(args.data_dir / f"{name}_{layer}.npy")
             for layer in range(args.layers) for name in ("key", "value")]

    def pack():
        return [value.transpose(0, 2, 3, 1).copy() for value in cache]

    packed = pack()
    assert all(np.array_equal(a, b) for a, b in zip(packed, pack()))
    pack_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        repacked = pack()
        pack_ms.append((time.perf_counter() - start) * 1000)
    result = {
        "scope": "fixed exported observation and suffix; not a closed-loop rollout",
        "cache_shape": list(cache[0].shape),
        "cache_bytes": sum(value.nbytes for value in cache),
        "suffix_bytes": suffix.nbytes,
        "pack_median_ms": statistics.median(pack_ms),
        "pack_p95_ms": float(np.percentile(pack_ms, 95)),
        "packing_bitwise_equal": True,
    }
    if args.model:
        from rknnlite.api import RKNNLite

        runtime = RKNNLite()
        try:
            assert runtime.load_rknn(str(args.model)) == 0
            assert runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
            for _ in range(3):
                runtime.inference(inputs=[suffix, *packed], data_format=None)
            times = {"repack": [], "reuse_host_arrays": []}
            max_error = 0.0
            outputs = {}
            for iteration in range(args.repeats):
                # Alternate order to limit thermal/order bias.
                order = list(times) if iteration % 2 == 0 else list(reversed(times))
                for mode in order:
                    start = time.perf_counter()
                    inputs = [suffix, *(pack() if mode == "repack" else packed)]
                    out = runtime.inference(inputs=inputs, data_format=None)[0]
                    times[mode].append((time.perf_counter() - start) * 1000)
                    outputs[mode] = out.copy()
                max_error = max(max_error, float(np.max(np.abs(
                    outputs["repack"] - outputs["reuse_host_arrays"]))))
            result["runtime"] = {
                mode: {"median_ms": statistics.median(samples),
                       "p95_ms": float(np.percentile(samples, 95))}
                for mode, samples in times.items()
            }
            result["output_max_abs_difference"] = max_error
            result["note"] = "Host reuse does not prove zero-copy or NPU KV residency."
        finally:
            runtime.release()
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
