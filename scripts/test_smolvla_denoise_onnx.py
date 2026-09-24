#!/usr/bin/env python3
"""Check cached SmolVLA denoise ONNX numerics before RKNN conversion."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    directory = Path(args.data_dir)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inputs = {
        item.name: np.load(directory / f"{item.name}.npy")
        for item in session.get_inputs()
    }
    output = session.run(None, inputs)[0].astype(np.float64).ravel()
    reference = np.load(directory / "reference.npy").astype(np.float64).ravel()
    difference = np.abs(output - reference)
    result = {
        "inputs": len(inputs),
        "cosine": float(np.dot(output, reference) / (np.linalg.norm(output) * np.linalg.norm(reference))),
        "mae": float(difference.mean()),
        "max_abs_error": float(difference.max()),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
