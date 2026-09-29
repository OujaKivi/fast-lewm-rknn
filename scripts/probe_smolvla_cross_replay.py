#!/usr/bin/env python3
"""Replay C17 on prefixes/suffixes captured from complete recorded policy calls."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from probe_smolvla_cross_boundary import compare


def reconstruct(args):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    source = args.base / "reference" / "prefix_to_hidden.onnx"
    session = ort.InferenceSession(str(source), options)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    manifest = {"scope": "Recorded NPU prefill K/V and suffixes from ten dependent target steps; layer-0 hidden reconstructed using CPU FP32 ORT, NOT a captured NPU internal hidden or an integrated C17 trajectory",
                "upstream_onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": {}}
    for path in sorted(args.captures.glob("*_inputs.npz")):
        with np.load(path) as values:
            hidden = np.stack([session.run(None, {"suffix": suffix, "key_0": values["key_0"], "value_0": values["value_0"]})[0]
                               for suffix in values["suffix"]])
            if hidden.shape != (10, 1, 50, 480) or not np.isfinite(hidden).all():
                raise ValueError("Invalid reconstructed hidden sequence")
            output = args.fixtures / path.name.replace("_inputs", "_cross")
            np.savez(output, hidden=hidden, key=values["key_1"], value=values["value_1"])
        manifest["cases"][output.name] = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    if not manifest["cases"]:
        raise ValueError("No captured real policy inputs")
    (args.fixtures / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def measure(args):
    os.sched_setaffinity(0, {4, 5, 6, 7})
    library = ctypes.CDLL(str(args.library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    timing = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.cross_boundary_error.restype = ctypes.c_char_p
    library.cross_followup_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.cross_followup_create.restype = ctypes.c_void_p
    library.cross_followup_destroy.argtypes = [ctypes.c_void_p]
    library.cross_followup_prefix.argtypes = [ctypes.c_void_p, array, array]
    library.cross_followup_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_uint, array, timing]
    handle = library.cross_followup_create(str(args.directory / "models").encode(), str(args.base / "models").encode())
    if not handle:
        raise RuntimeError(library.cross_boundary_error().decode())

    def check(code):
        if code < 0:
            raise RuntimeError(library.cross_boundary_error().decode())

    def run(mode, hidden):
        output = np.empty((1, 50, 480), np.float32)
        times = np.zeros(8, np.float64)
        check(library.cross_followup_run(handle, mode, hidden, len(hidden), output, times))
        return output, times

    names = {7: "split_parallel_heads", 17: "compact_parallel_cpu_both"}
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    report = {"scope": manifest["scope"], "fixtures": manifest,
              "limitation": "Independent replay of captured hidden inputs: no C17 feedback into Euler, no end-to-end C17 speedup",
              "compiler_opt3": ["SMOLVLA_CROSS_OPT3" in os.environ, "SMOLVLA_FOLLOWUP_OPT3" in os.environ],
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "model_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((args.directory / "models").glob("*.rknn"))},
              "repeats": args.repeats, "cases": {}}
    try:
        for path in sorted(args.fixtures.glob("*_cross.npz")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["cases"][path.name]["fixture_sha256"]:
                raise ValueError("Recorded fixture changed")
            with np.load(path) as values:
                hidden, key, value = [np.ascontiguousarray(values[name]) for name in ("hidden", "key", "value")]
            check(library.cross_followup_prefix(handle, key, value))
            validation = {name: [] for name in names.values()}
            for step in hidden:
                anchor = run(8, step[None])[0]
                for mode, name in names.items():
                    validation[name].append(compare(run(mode, step[None])[0], anchor))
            if not all(result["bitwise_equal"] for values in validation.values() for result in values):
                raise ValueError("Recorded-input boundary differs from original target bits")
            anchors = {mode: run(mode, hidden)[0] for mode in names}
            for _ in range(4):
                for mode in names:
                    run(mode, hidden)
            samples = {name: [] for name in names.values()}
            modes = list(names)
            print("PROBE_MEASURE_START", flush=True)
            for iteration in range(args.repeats):
                for offset in range(2):
                    mode = modes[(iteration + offset) % 2]
                    print(f"PROBE_PLAN_START {path.stem}_{names[mode]}", flush=True)
                    output, times = run(mode, hidden)
                    print("PROBE_PLAN_END", flush=True)
                    if not compare(output, anchors[mode])["bitwise_equal"]:
                        raise ValueError("Nondeterministic recorded replay")
                    samples[names[mode]].append(times.tolist())
            print("PROBE_MEASURE_END", flush=True)
            report["cases"][path.name] = {"validation": validation, "plans": {
                name: {"samples_ms": values, "median_ms": np.median(values, axis=0).tolist(),
                       "p95_ms": np.percentile(values, 95, axis=0).tolist()} for name, values in samples.items()}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({name: {plan: values["median_ms"][-1] for plan, values in case["plans"].items()}
                          for name, case in report["cases"].items()}, indent=2))
    finally:
        library.cross_followup_destroy(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reconstruct", "measure"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=60)
    args = parser.parse_args()
    if args.repeats < 1 or (args.mode == "reconstruct" and not args.captures) or (
        args.mode == "measure" and not all((args.directory, args.library, args.output))
    ):
        parser.error("Missing mode-specific paths or invalid repeats")
    (reconstruct if args.mode == "reconstruct" else measure)(args)


if __name__ == "__main__":
    main()
