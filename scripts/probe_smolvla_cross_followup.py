#!/usr/bin/env python3
"""Same-boundary numerical and head-parallel audit, with all preparation charged."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from probe_smolvla_cross_boundary import compare


NAMES = ("ready_joint", "propagated_ready", "restore_o", "original_q", "original_q_restore_o",
         "split_full_heads", "split_serial_heads", "split_parallel_heads", "original_recompute",
         "compact_split_full", "compact_split_serial", "compact_split_parallel",
         "original_order_compact_full", "original_order_compact_serial", "original_order_compact_parallel",
         "compact_parallel_cpu_restore", "compact_full_cpu_restore",
         "compact_parallel_cpu_both", "compact_full_cpu_both")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--stress-cases", type=int, default=64)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--call-telemetry", action="store_true")
    parser.add_argument("--native-trace", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.stress_cases < 1:
        parser.error("Positive repeat/stress counts required")
    if any(count < 1 for count in args.counts) or len(set(args.modes)) != len(args.modes):
        parser.error("Positive counts and unique modes required")
    selected = [NAMES.index(name) for name in args.modes]
    if args.native_trace and (not args.call_telemetry or "SMOLVLA_CROSS_TRACE" not in os.environ):
        parser.error("Native tracing requires --call-telemetry and SMOLVLA_CROSS_TRACE")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    library = ctypes.CDLL(str(args.library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    times_array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.cross_boundary_error.restype = ctypes.c_char_p
    library.cross_followup_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.cross_followup_create.restype = ctypes.c_void_p
    library.cross_followup_destroy.argtypes = [ctypes.c_void_p]
    library.cross_followup_prefix.argtypes = [ctypes.c_void_p, array, array]
    library.cross_followup_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_uint, array, times_array]
    library.cross_followup_info.argtypes = [ctypes.c_void_p]
    library.cross_followup_info.restype = ctypes.c_char_p
    if args.native_trace:
        library.cross_followup_trace.argtypes = [ctypes.c_void_p]
        library.cross_followup_trace.restype = ctypes.c_char_p
    handle = library.cross_followup_create(str(args.directory / "models").encode(), str(args.base / "models").encode())
    if not handle:
        raise RuntimeError(library.cross_boundary_error().decode())

    def check(code):
        if code < 0:
            raise RuntimeError(library.cross_boundary_error().decode())

    def prefix(key, value):
        check(library.cross_followup_prefix(handle, np.ascontiguousarray(key), np.ascontiguousarray(value)))

    def run(mode, hidden):
        output = np.empty((1, 50, 480), np.float32)
        times = np.zeros(8, np.float64)
        check(library.cross_followup_run(handle, mode, np.ascontiguousarray(hidden), len(hidden), output, times))
        return output, times

    report = {"scope": "Actual layer-1 Q/RoPE/attention/O/residual, no MLP, synthetic checkpoint activations; not full flow or closed loop",
              "affinity": sorted(os.sched_getaffinity(0)), "head_core_masks": [1, 2, 4],
              "compiler_controls_opt3": "SMOLVLA_CROSS_OPT3" in os.environ,
              "followup_opt3": {"enabled": "SMOLVLA_FOLLOWUP_OPT3" in os.environ,
                                "selected_graphs": ["head_producer", "head_consumer", "head_attention_5", "head_attention_15",
                                                    "group_producer", "group_attention_1", "group_attention_2", "group_attention_5"]},
              "timing_columns": ["preparation_ms", "hidden_pack_sync_ms", "joint_run_or_bit_shuffle_ms", "split_producer_ms",
                                 "split_attention_ms", "split_consumer_ms", "final_read_ms", "total_ms"],
              "model_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((args.directory / "models").glob("*.rknn"))},
              "base_model_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted((args.base / "models").glob("*.rknn"))},
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "validation": {}, "plans": {},
              "compact_group_core_split": [2, 2, 1],
              "selected_timing_modes": args.modes,
              "call_telemetry": {},
              "native_trace_columns_ms": ["producer", "cpu_pack", "attention_and_barrier", "cpu_restore", "consumer",
                                           "wake0", "wake1", "wake2", "run0", "run1", "run2"] if args.native_trace else None,
              "limitations": ["33 contexts coexist; no isolated peak memory, thermal or energy claim",
                              "Persistent three-worker head dispatch; dispatch/barrier charged",
                              "Strict checked FD-offset views, no CPU Q/K/V round trip, default SDK sync retained",
                              "Preparation once per sequence charged; common raw prefix pack excluded equally",
                              "CPU native bridges copy FP16 storage bits; FROM/TO sync and 120 tiles per direction charged",
                              "Ten independent hidden inputs, not dependent Euler steps; final read once equally",
                              "Numerical controls are compiled independently; compiler lowering can also change"]}
    try:
        information = library.cross_followup_info(handle)
        if not information:
            raise RuntimeError(library.cross_boundary_error().decode())
        report["graphs"] = json.loads(information)
        anchors = {}
        for case in ("original", "other_suffix", "changed_prefix", "restored"):
            fixture = "original" if case == "restored" else case
            hidden = np.load(args.base / f"{fixture}_hidden.npy")
            key, value = [np.load(args.base / f"{fixture}_{n}.npy") for n in ("key", "value")]
            prefix(key, value)
            anchor = run(8, hidden[None])[0]
            anchors[case] = anchor
            report["validation"][case] = {name: compare(run(mode, hidden[None])[0], anchor) for mode, name in enumerate(NAMES)}
        report["restore_check"] = compare(anchors["restored"], anchors["original"])
        report["prefix_effect"] = compare(anchors["changed_prefix"], anchors["original"])
        if not report["restore_check"]["bitwise_equal"] or not report["prefix_effect"]["max_abs"]:
            raise ValueError("Invalid prefix switch/restore")
        rng = np.random.default_rng(929)
        hidden = np.load(args.base / "original_hidden.npy")
        key, value = [np.load(args.base / f"original_{n}.npy") for n in ("key", "value")]
        stress = []
        for _ in range(args.stress_cases):
            altered = [np.ascontiguousarray(a + np.float32(.05) * rng.standard_normal(a.shape, dtype=np.float32))
                       for a in (hidden, key, value)]
            prefix(altered[1], altered[2])
            anchor = run(8, altered[0][None])[0]
            stress.append({name: compare(run(mode, altered[0][None])[0], anchor) for mode, name in enumerate(NAMES)})
        report["stress"] = {"seed": 929, "cases": stress, "summary": {
            name: {"bitwise_equal_cases": sum(c[name]["bitwise_equal"] for c in stress),
                   "max_abs": max(c[name]["max_abs"] for c in stress),
                   "max_different_elements": max(c[name]["different_elements"] for c in stress)} for name in NAMES}}
        prefix(key, value)
        alternate = np.load(args.base / "other_suffix_hidden.npy")
        for count in (() if args.validation_only else args.counts):
            sequence = np.stack([hidden if index % 2 == 0 else alternate for index in range(count)])
            own_anchors = {mode: run(mode, sequence)[0] for mode in selected}
            for _ in range(4):
                for mode in selected:
                    run(mode, sequence)
            samples = [[] for _ in NAMES]
            telemetry = {NAMES[mode]: [] for mode in selected}
            print("PROBE_MEASURE_START", flush=True)
            for iteration in range(args.repeats):
                for offset in range(len(selected)):
                    mode = selected[(iteration + offset) % len(selected)]
                    print(f"PROBE_PLAN_START count{count}_{NAMES[mode]}", flush=True)
                    if args.call_telemetry:
                        before = resource.getrusage(resource.RUSAGE_SELF)
                        started_ns = time.monotonic_ns()
                    result, times = run(mode, sequence)
                    if args.call_telemetry:
                        ended_ns = time.monotonic_ns()
                        after = resource.getrusage(resource.RUSAGE_SELF)
                        telemetry[NAMES[mode]].append({
                            "start_monotonic_ns": started_ns, "end_monotonic_ns": ended_ns,
                            "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
                            "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
                            "process_cpu_ms": 1000 * (after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime),
                        })
                        if args.native_trace:
                            telemetry[NAMES[mode]][-1]["native_steps_ms"] = json.loads(library.cross_followup_trace(handle))
                    print("PROBE_PLAN_END", flush=True)
                    if not compare(result, own_anchors[mode])["bitwise_equal"]:
                        raise ValueError(f"Nondeterministic {NAMES[mode]}")
                    samples[mode].append(times.tolist())
            print("PROBE_MEASURE_END", flush=True)
            report["plans"][str(count)] = {name: {"median_ms": np.median(values, axis=0).tolist(),
                                                  "p95_ms": np.percentile(values, 95, axis=0).tolist(), "samples_ms": values}
                                           for name, values in zip(NAMES, samples) if values}
            if args.call_telemetry:
                report["call_telemetry"][str(count)] = telemetry
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stress": report["stress"]["summary"],
                          "medians": {count: {n: p["median_ms"] for n, p in plans.items()} for count, plans in report["plans"].items()}}, indent=2))
    finally:
        library.cross_followup_destroy(handle)


if __name__ == "__main__":
    main()
