#!/usr/bin/env python3
"""Measure exact attention primitives on RK3588 CPU big cores."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def measure(operation, repeats):
    operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - start) * 1000)
    if not torch.isfinite(result).all().item():
        raise ValueError("Non-finite attention output")
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--key-tokens", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads not in (1, 2, 4):
        parser.error("positive repeats and 1/2/4 threads required")
    if min(args.heads, args.query_tokens, args.key_tokens, args.head_dim) < 1:
        parser.error("positive attention dimensions required")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    inputs = [torch.randn(1, args.heads, length, args.head_dim)
              for length in (args.query_tokens, args.key_tokens, args.key_tokens)]
    report = {"scope": "Standalone CPU attention geometry; no NPU/native-layout transitions; FP16/FP32 are separate baselines",
              "affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
              "torch_version": torch.__version__, "shape": list(inputs[0].shape),
              "key_shape": list(inputs[1].shape), "input_kind": "synthetic random Q/K/V, not checkpoint activations",
              "dtypes": {}}
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16):
            query, key, value = [tensor.to(dtype) for tensor in inputs]
            stages = {}
            try:
                scale = args.head_dim ** -0.5
                scores = torch.matmul(query * scale, key.transpose(-1, -2))
                probabilities = torch.softmax(scores, dim=-1)
                operations = {
                    "qk": lambda: torch.matmul(query * scale, key.transpose(-1, -2)),
                    "softmax": lambda: torch.softmax(scores, dim=-1),
                    "pv": lambda: torch.matmul(probabilities, value),
                    "sdpa": lambda: F.scaled_dot_product_attention(query, key, value),
                }
                for name, operation in operations.items():
                    stages[name] = measure(operation, args.repeats)
            except (RuntimeError, ValueError) as error:
                stages["error"] = str(error)
            report["dtypes"][str(dtype)] = stages
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
