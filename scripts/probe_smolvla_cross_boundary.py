#!/usr/bin/env python3
"""Rotated block controls, real ONNX activations and checked native continuation."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from probe_smolvla_native_conditioning import compare as basic_compare


def compare(actual, expected):
    result = basic_compare(actual, expected)
    result["different_elements"] = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--core-mask", type=int, choices=(1, 3, 7), default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    library = ctypes.CDLL(str(args.library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    timings = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.cross_boundary_error.restype = ctypes.c_char_p
    library.cross_boundary_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    library.cross_boundary_create.restype = ctypes.c_void_p
    library.cross_boundary_destroy.argtypes = [ctypes.c_void_p]
    library.cross_boundary_prefix.argtypes = [ctypes.c_void_p, array, array]
    library.cross_boundary_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_uint, array, timings]
    library.cross_boundary_info.argtypes = [ctypes.c_void_p]
    library.cross_boundary_info.restype = ctypes.c_char_p
    handle = library.cross_boundary_create(str(args.directory / "models").encode(), args.core_mask)
    if not handle:
        raise RuntimeError(library.cross_boundary_error().decode())

    def check(code):
        if code < 0:
            raise RuntimeError(library.cross_boundary_error().decode())

    def run(mode, hidden):
        output = np.empty((1, 50, 480), np.float32)
        times = np.zeros(8, np.float64)
        check(library.cross_boundary_run(handle, mode, np.ascontiguousarray(hidden), len(hidden), output, times))
        return output, times

    names = ("original_recompute", "ready_joint", "compact_repeat_joint", "compact_grouped_joint", "compact_grouped_split",
             "compact_grouped_interleaved", "compact_grouped_propagated", "compact_grouped_propagated_ready", "compact_grouped_ready_control")
    report = {"scope": "Actual layer-1 RMSNorm/Q/RoPE/attention/output projection/residual; CPU ONNX checkpoint activations on a synthetic prefix; NOT full transformer MLP, full flow, real images or closed-loop",
              "core_mask": args.core_mask, "affinity": sorted(os.sched_getaffinity(0)),
              "compiler_controls_opt3": "SMOLVLA_CROSS_OPT3" in os.environ,
              "control_graph_files": {"ready": "block_ready_opt3.rknn" if "SMOLVLA_CROSS_OPT3" in os.environ else "block_ready.rknn",
                                      "grouped_ready": "block_grouped_joint_ready_opt3.rknn" if "SMOLVLA_CROSS_OPT3" in os.environ else "block_grouped_joint_ready.rknn"},
              "timing_columns": ["preparation_ms", "hidden_pack_sync_ms", "joint_run_ms", "split_producer_ms", "split_attention_ms", "split_consumer_ms", "final_read_ms", "total_ms"],
              "model_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((args.directory / "models").glob("*.rknn"))},
              "validation": {}, "plans": {},
              "limitations": ["Fourteen contexts coexist; graph memory is not deployment peak RSS",
                              "All original numerical work retained; backend bit parity is tested, not assumed",
                              "Ten independent alternating hidden activations, not dependent Euler steps",
                              "One preparation per sequence charged; common raw-prefix pack is reported separately",
                              "Final read once per sequence equally for all plans; default SDK synchronization retained",
                              "CPU affinity fixed; no CPU frequency/thermal lock or device-general inference"]}
    try:
        info = library.cross_boundary_info(handle)
        if not info:
            raise RuntimeError(library.cross_boundary_error().decode())
        report["graphs"] = json.loads(info)
        anchors = {}
        for case in ("original", "other_suffix", "changed_prefix", "restored"):
            fixture = "original" if case == "restored" else case
            hidden = np.load(args.directory / f"{fixture}_hidden.npy")
            key, value = [np.ascontiguousarray(np.load(args.directory / f"{fixture}_{name}.npy")) for name in ("key", "value")]
            check(library.cross_boundary_prefix(handle, key, value))
            anchor = run(0, hidden[None])[0]
            anchors[case] = anchor
            outputs = {name: compare(run(mode, hidden[None])[0], anchor) for mode, name in enumerate(names)}
            report["validation"][case] = {
                "plans_vs_original": outputs,
                "original_vs_fp32": compare(anchor, np.load(args.directory / f"{fixture}_reference.npy"))}
        report["changed_prefix_effect"] = compare(anchors["changed_prefix"], anchors["original"])
        report["restored_vs_original"] = compare(anchors["restored"], anchors["original"])
        if not report["restored_vs_original"]["bitwise_equal"] or report["changed_prefix_effect"]["max_abs"] == 0:
            raise ValueError("Invalid conditioning switch/restore")
        key, value = [np.ascontiguousarray(np.load(args.directory / f"original_{name}.npy")) for name in ("key", "value")]
        random = np.random.default_rng(929)
        original_hidden = np.load(args.directory / "original_hidden.npy")
        stress = []
        for _ in range(12):
            altered = [np.ascontiguousarray(array + np.float32(0.05) * random.standard_normal(array.shape, dtype=np.float32))
                       for array in (original_hidden, key, value)]
            check(library.cross_boundary_prefix(handle, altered[1], altered[2]))
            anchor = run(0, altered[0][None])[0]
            stress.append({name: compare(run(mode, altered[0][None])[0], anchor) for mode, name in enumerate(names)})
        report["synthetic_numerical_stress"] = {
            "scope": "12 deterministic hidden/prefix perturbations, seed 929, not robot observations or a quality test",
            "cases": stress,
            "summary": {name: {"bitwise_equal_cases": sum(case[name]["bitwise_equal"] for case in stress),
                               "max_abs": max(case[name]["max_abs"] for case in stress),
                               "max_different_elements": max(case[name]["different_elements"] for case in stress)} for name in names}}
        import time
        tick = time.perf_counter()
        check(library.cross_boundary_prefix(handle, key, value))
        report["common_prefix_pack_sync_ms"] = (time.perf_counter() - tick) * 1000
        hidden = [np.load(args.directory / f"{case}_hidden.npy") for case in ("original", "other_suffix")]
        for count in (1, 10):
            sequence = np.stack([hidden[index % 2] for index in range(count)])
            own_anchors = [run(mode, sequence)[0] for mode in range(len(names))]
            for _ in range(4):
                for mode in range(len(names)):
                    run(mode, sequence)
            samples = [[] for _ in names]
            print("PROBE_MEASURE_START", flush=True)
            for iteration in range(args.repeats):
                for offset in range(len(names)):
                    mode = (iteration + offset) % len(names)
                    print(f"PROBE_PLAN_START count{count}_{names[mode]}", flush=True)
                    output, times = run(mode, sequence)
                    print("PROBE_PLAN_END", flush=True)
                    if not compare(output, own_anchors[mode])["bitwise_equal"]:
                        raise ValueError(f"Nondeterministic mode {names[mode]}")
                    samples[mode].append(times.tolist())
            print("PROBE_MEASURE_END", flush=True)
            report["plans"][str(count)] = {name: {"median_ms": np.median(values, axis=0).tolist(),
                                                  "p95_ms": np.percentile(values, 95, axis=0).tolist(),
                                                  "samples_ms": values}
                                           for name, values in zip(names, samples)}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"validation": report["validation"],
                          "plans": {count: {name: values["median_ms"] for name, values in plans.items()}
                                    for count, plans in report["plans"].items()}}, indent=2))
    finally:
        library.cross_boundary_destroy(handle)


if __name__ == "__main__":
    main()
