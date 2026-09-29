#!/usr/bin/env python3
"""Compare a full attention probe with exact, independent head partitions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-model", required=True)
    parser.add_argument("--half-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reservation-probe", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    generator = np.random.default_rng(0)
    inputs = [generator.standard_normal((1, 12, 1024, 64), dtype=np.float32)
              for _ in range(3)]
    partitions = [[np.ascontiguousarray(value[:, part * 6:(part + 1) * 6].transpose(0, 2, 3, 1))
                   for value in inputs] for part in range(2)]
    native_inputs = [np.ascontiguousarray(value.transpose(0, 2, 3, 1)) for value in inputs]
    runtimes = []
    try:
        for model, mask in ((args.full_model, RKNNLite.NPU_CORE_0_1_2),
                            (args.half_model, RKNNLite.NPU_CORE_0),
                            (args.half_model, RKNNLite.NPU_CORE_1)):
            runtime = RKNNLite()
            runtimes.append(runtime)
            if runtime.load_rknn(model) != 0 or runtime.init_runtime(core_mask=mask) != 0:
                raise RuntimeError(f"Failed to initialize {model} on mask {mask}")
        if args.reservation_probe:
            runtime = RKNNLite()
            runtimes.append(runtime)
            if runtime.load_rknn(args.full_model) != 0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                raise RuntimeError("Failed to initialize one-core full attention")

        def execute(runtime, values):
            return runtime.inference(inputs=values, data_format=["nhwc"] * 3)[0].copy()

        reference = execute(runtimes[0], native_inputs)
        for runtime, values in zip(runtimes[1:], partitions):
            execute(runtime, values)
        times = {mode: [] for mode in ("full_three_cores", "serial_head_halves", "parallel_head_halves")}
        errors = {mode: 0.0 for mode in times}
        reservation = {mode: [] for mode in ("full_mask7_with_core1_half", "full_mask1_with_core1_half")}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for iteration in range(args.repeats):
                modes = list(times)
                modes = modes[iteration % 3:] + modes[:iteration % 3]
                for mode in modes:
                    start = time.perf_counter()
                    if mode == "full_three_cores":
                        result = execute(runtimes[0], native_inputs)
                    elif mode == "serial_head_halves":
                        result = np.concatenate([execute(runtimes[1], values)
                                                 for values in partitions], axis=1)
                    else:
                        futures = [pool.submit(execute, runtime, values)
                                   for runtime, values in zip(runtimes[1:], partitions)]
                        result = np.concatenate([future.result() for future in futures], axis=1)
                    times[mode].append((time.perf_counter() - start) * 1000)
                    errors[mode] = max(errors[mode], float(np.max(np.abs(result - reference))))
            if args.reservation_probe:
                def timed_execute(runtime, values):
                    start = time.perf_counter()
                    result = execute(runtime, values)
                    return result, (time.perf_counter() - start) * 1000

                for iteration in range(args.repeats):
                    modes = list(reservation)
                    if iteration % 2:
                        modes.reverse()
                    for mode in modes:
                        runtime = runtimes[0 if "mask7" in mode else 3]
                        start = time.perf_counter()
                        full = pool.submit(timed_execute, runtime, native_inputs)
                        half = pool.submit(timed_execute, runtimes[2], partitions[1])
                        full_result, full_ms = full.result()
                        _, half_ms = half.result()
                        reservation[mode].append({
                            "makespan_ms": (time.perf_counter() - start) * 1000,
                            "full_ms": full_ms,
                            "half_ms": half_ms,
                            "full_max_abs_difference": float(np.max(np.abs(full_result - reference))),
                        })
        report = {
            "scope": "Standalone unmasked vision SDPA geometry; random prepartitioned host inputs; not a complete block",
            "shape": list(inputs[0].shape),
            "output_shape": list(reference.shape),
            "plans": {mode: {"median_ms": float(np.median(samples)),
                             "samples_ms": samples,
                             "max_abs_output_difference": errors[mode]}
                      for mode, samples in times.items()},
            "note": "Includes FP32 runtime input/output conversion and output concatenation; host inputs are prepartitioned native NHWC. Excludes initial slicing/layout conversion. All contexts resident in every plan.",
        }
        if args.reservation_probe:
            report["reservation_probe"] = {
                "scope": "Unequal independent work: full 12 heads with a concurrent 6-head graph on NPU core 1. Not a comparison of equal work to the primary plans.",
                "plans": {mode: {
                    "median_makespan_ms": float(np.median([sample["makespan_ms"] for sample in samples])),
                    "median_full_ms": float(np.median([sample["full_ms"] for sample in samples])),
                    "median_half_ms": float(np.median([sample["half_ms"] for sample in samples])),
                    "samples": samples,
                } for mode, samples in reservation.items()},
            }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for runtime in runtimes:
            runtime.release()


if __name__ == "__main__":
    main()
