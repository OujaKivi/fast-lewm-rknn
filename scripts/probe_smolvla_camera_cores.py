#!/usr/bin/env python3
"""Compare identical two-view encoding work under different NPU core schedules."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    source = np.load(args.input)
    images = [np.ascontiguousarray(source), np.ascontiguousarray(source[..., ::-1])]
    runtimes = []
    try:
        for mask in (RKNNLite.NPU_CORE_0_1_2, RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1):
            runtime = RKNNLite()
            runtimes.append(runtime)
            assert runtime.load_rknn(args.model) == 0
            assert runtime.init_runtime(core_mask=mask) == 0
        if args.batch_model:
            runtime = RKNNLite()
            runtimes.append(runtime)
            assert runtime.load_rknn(args.batch_model) == 0
            assert runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0

        def encode(runtime, image):
            return runtime.inference(inputs=[image], data_format=["nchw"])[0].copy()

        references = [encode(runtimes[0], image) for image in images]
        for runtime in runtimes[1:3]:
            encode(runtime, images[0])
        times = {"serial_three_cores": [], "serial_one_core": [], "parallel_two_single_cores": []}
        if args.batch_model:
            times["batch_two_three_cores"] = []
            batched_images = np.concatenate(images, axis=0)
            encode(runtimes[3], batched_images)
        errors = {mode: 0.0 for mode in times}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for iteration in range(args.repeats):
                order = list(times)
                order = order[iteration % len(order):] + order[:iteration % len(order)]
                for mode in order:
                    start = time.perf_counter()
                    if mode == "parallel_two_single_cores":
                        futures = [pool.submit(encode, runtime, image)
                                   for runtime, image in zip(runtimes[1:], images)]
                        outputs = [future.result() for future in futures]
                    elif mode == "batch_two_three_cores":
                        batch_outputs = encode(runtimes[3], batched_images)
                        outputs = [batch_outputs[index] for index in range(2)]
                    else:
                        runtime = runtimes[0 if mode == "serial_three_cores" else 1]
                        outputs = [encode(runtime, image) for image in images]
                    times[mode].append((time.perf_counter() - start) * 1000)
                    for expected, actual in zip(references, outputs):
                        errors[mode] = max(errors[mode], float(np.max(np.abs(expected - actual))))
        result = {
            "scope": "two fixed inputs (export fixture and horizontal flip), same complete vision graph; not a rollout",
            "input_shape": list(source.shape),
            "plans": {mode: {"median_ms": float(np.median(samples)),
                             "p95_ms": float(np.percentile(samples, 95)),
                             "samples_ms": samples,
                             "max_abs_output_difference": errors[mode]}
                      for mode, samples in times.items()},
            "note": "Three contexts resident in every plan; weights not explicitly shared. Plan order rotates each trial.",
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        for runtime in runtimes:
            runtime.release()


if __name__ == "__main__":
    main()
