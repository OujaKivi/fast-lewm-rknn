#!/usr/bin/env python3
"""Compare every exported SmolVLA prefill cache against PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def compare(reference, output):
    left = reference.astype(np.float64).ravel()
    right = output.astype(np.float64).ravel()
    return {
        "cosine": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
        "mae": float(np.mean(np.abs(left - right))),
        "max_abs": float(np.max(np.abs(left - right))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    fixture = Path(args.fixture_dir)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    prefix = np.load(fixture / "prefix.npy")
    outputs = session.run(None, {"prefix": prefix})
    names = [entry.name for entry in session.get_outputs()]
    metrics = {}
    for name, output in zip(names, outputs, strict=True):
        reference = np.load(fixture / f"{name}_reference.npy")
        metrics[name] = compare(reference, output)
    result = {
        "onnx": args.onnx,
        "outputs": len(outputs),
        "minimum_cosine": min(row["cosine"] for row in metrics.values()),
        "maximum_mae": max(row["mae"] for row in metrics.values()),
        "maximum_abs": max(row["max_abs"] for row in metrics.values()),
        "per_output": metrics,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_output"}, indent=2))
    if result["minimum_cosine"] < 0.99999 or result["maximum_mae"] > 0.0001:
        raise RuntimeError("ONNX prefill cache parity failed")


if __name__ == "__main__":
    main()
