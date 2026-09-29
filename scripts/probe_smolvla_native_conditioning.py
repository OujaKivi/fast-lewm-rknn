#!/usr/bin/env python3
"""Paired native conditioning-fragment audit; not a full flow benchmark."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np


def compare(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid output")
    delta = np.abs(actual - expected)
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "bitwise_equal": bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--core-mask", type=int, choices=[1, 3, 7], default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    library = ctypes.CDLL(str(args.library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    timings = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.conditioning_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    library.conditioning_create.restype = ctypes.c_void_p
    library.conditioning_destroy.argtypes = [ctypes.c_void_p]
    library.conditioning_error.restype = ctypes.c_char_p
    library.conditioning_info.argtypes = [ctypes.c_void_p]
    library.conditioning_info.restype = ctypes.c_char_p
    library.conditioning_prefix.argtypes = [ctypes.c_void_p, array, array]
    library.conditioning_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_uint, array, timings]
    library.conditioning_projection.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, array]
    handle = library.conditioning_create(str(args.directory).encode(), args.core_mask)
    if not handle:
        raise RuntimeError(library.conditioning_error().decode())

    def check(code):
        if code < 0:
            raise RuntimeError(library.conditioning_error().decode())

    def run(mode, queries):
        result = np.empty((1, 50, 15, 64), np.float32)
        times = np.zeros(5, np.float64)
        check(library.conditioning_run(handle, mode, np.ascontiguousarray(queries), len(queries), result, times))
        return result, times

    names = ("fused_recompute", "cache_compact_repeat", "cache_expanded", "cache_compact_grouped", "cache_consumer_ready")
    queries = [np.load(args.directory / f"query_{index}.npy") for index in range(2)]
    key, value = [np.load(args.directory / (name + "_1.npy")) for name in ("key", "value")]
    report = {"scope": "Actual layer-1 fragment, native FP16 and native FD preparation-to-consumer continuation; synthetic prefix, real checkpoint queries; 10 alternating independent queries are NOT dependent full-model flow steps",
              "core_mask": args.core_mask, "affinity": sorted(os.sched_getaffinity(0)),
              "timing_columns": ["one_time_preparation_ms", "query_pack_sync_ms", "consumer_run_ms", "final_output_read_ms", "total_ms"],
              "fixture_validation": {}, "plans": {},
              "limitations": ["All eight graph contexts remain loaded; memory query is per-context, not peak RSS",
                              "No visual/prefill/action-output projection/closed-loop measurements",
                              "No continuous CPU frequency or thermal lock; modes rotated",
                              "Queries share the same observation but are not a dependent ten-step latent trajectory",
                              "Native graph default cache sync retained; output read charged once equally to all fragment plans"]}
    try:
        info = library.conditioning_info(handle)
        if not info:
            raise RuntimeError(library.conditioning_error().decode())
        report["graphs"] = json.loads(info)
        check(library.conditioning_prefix(handle, key, value))
        projected = []
        for expanded in (0, 1):
            shape = (1, 149, 15 if expanded else 5, 64)
            prepared = [np.empty(shape, np.float32) for _ in range(2)]
            check(library.conditioning_projection(handle, expanded, *prepared))
            projected.append(prepared)
        report["projection_expansion"] = []
        for compact, expanded in zip(*projected):
            repeated = np.repeat(compact, 3, axis=2)
            report["projection_expansion"].append(compare(expanded, repeated))
        for query_index, query in enumerate(queries):
            reference, _ = run(0, query[None])
            report["fixture_validation"][f"query_{query_index}"] = {
                "fused_vs_fp32_onnx": compare(reference, np.load(args.directory / f"reference_{query_index}.npy")),
                "plans_vs_fused": {name: compare(run(mode, query[None])[0], reference)
                                   for mode, name in enumerate(names)}}
        for steps in (1, 10):
            sequence = np.stack([queries[index % 2] for index in range(steps)])
            expected = run(0, sequence)[0]
            for _ in range(4):
                for mode in range(len(names)):
                    run(mode, sequence)
            print("PROBE_MEASURE_START", flush=True)
            samples = [[] for _ in names]
            for iteration in range(args.repeats):
                for offset in range(len(names)):
                    mode = (iteration + offset) % len(names)
                    print(f"PROBE_PLAN_START steps{steps}_{names[mode]}", flush=True)
                    result, times = run(mode, sequence)
                    print("PROBE_PLAN_END", flush=True)
                    if not compare(result, expected)["bitwise_equal"] and mode == 0:
                        raise ValueError("Fused anchor became nondeterministic")
                    samples[mode].append(times.tolist())
            print("PROBE_MEASURE_END", flush=True)
            report["plans"][str(steps)] = {name: {"median_ms": np.median(values, axis=0).tolist(),
                                                 "p95_ms": np.percentile(values, 95, axis=0).tolist(),
                                                 "samples_ms": values}
                                          for name, values in zip(names, samples)}
        changed_value = np.ascontiguousarray(-value[:, ::-1])
        check(library.conditioning_prefix(handle, key, changed_value))
        reference, _ = run(0, queries[0][None])
        report["changed_prefix_validation"] = {name: compare(run(mode, queries[0][None])[0], reference)
                                               for mode, name in enumerate(names)}
        check(library.conditioning_prefix(handle, key, value))
        restored, _ = run(0, queries[0][None])
        report["changed_prefix_final_difference"] = compare(reference, restored)
        report["restored_prefix_validation"] = {name: compare(run(mode, queries[0][None])[0], restored)
                                                for mode, name in enumerate(names)}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"projection_expansion": report["projection_expansion"],
                          "plans": {steps: {name: entry["median_ms"] for name, entry in values.items()}
                                    for steps, values in report["plans"].items()},
                          "parity": report["fixture_validation"]}, indent=2))
    finally:
        library.conditioning_destroy(handle)


if __name__ == "__main__":
    main()
