#!/usr/bin/env python3
"""Compare two complete RKNN vision graphs on identical images and precision."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    image = np.load(args.input).astype(np.float32)
    reference = np.load(args.reference).astype(np.float32)
    runtimes = {}
    try:
        for name, model in (("original", args.original), ("variant", args.variant)):
            runtime = RKNNLite()
            runtimes[name] = runtime
            assert runtime.load_rknn(model) == 0
            assert runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0

        def infer(runtime):
            return runtime.inference(inputs=[image.copy()], data_format=["nchw"])[0].copy()

        times = {name: [] for name in runtimes}
        outputs = {}
        for name, runtime in runtimes.items():
            for _ in range(3):
                infer(runtime)
        for iteration in range(args.repeats):
            order = list(runtimes) if iteration % 2 == 0 else list(reversed(runtimes))
            for name in order:
                start = time.perf_counter()
                outputs[name] = infer(runtimes[name])
                times[name].append((time.perf_counter() - start) * 1000)
        result = {
            "input_shape": list(image.shape),
            "scope": "same image, two FP16 graphs, alternating order; not a closed-loop test",
            "plans": {
                name: {
                    "median_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "samples_ms": samples,
                    "mae_vs_fp32": float(np.abs(outputs[name].astype(np.float32) - reference).mean()),
                    "max_abs_vs_fp32": float(np.abs(outputs[name].astype(np.float32) - reference).max()),
                }
                for name, samples in times.items()
            },
            "variant_vs_original": {
                "mae": float(np.abs(outputs["variant"] - outputs["original"]).mean()),
                "max_abs": float(np.abs(outputs["variant"] - outputs["original"]).max()),
            },
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        for runtime in runtimes.values():
            runtime.release()


if __name__ == "__main__":
    main()
