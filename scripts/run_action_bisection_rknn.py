"""Run Action Encoder bisection ONNX graphs in the RKNN simulator."""

import argparse
from pathlib import Path

import numpy as np
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/models")
    args = parser.parse_args()
    root = Path(args.model_dir)

    stages = (
        "frontend", "block1_norm1", "attention_norm", "attention_qkv",
        "attention_q", "attention_scores", "attention_softmax",
        "attention_context", "attention_flattened", "attention_projected",
        "block1_attention", "block1_residual",
        "block1_norm2", "block1_mlp", "block1", "block2", "block3", "final",
    )
    for name in stages:
        reference = np.load(root / f"{name}_ref.npz")
        expected = reference["output"]
        rknn = RKNN(verbose=False)
        rknn.config(
            mean_values=[[0.0] * 5, [0.0]],
            std_values=[[1.0] * 5, [1.0]],
            target_platform="rk3588",
            optimization_level=0,
        )
        if rknn.load_onnx(model=str(root / f"{name}.onnx")) != 0:
            raise RuntimeError(f"Unable to load {name}")
        if rknn.build(do_quantization=False) != 0 or rknn.init_runtime() != 0:
            raise RuntimeError(f"Unable to build {name}")
        actual = rknn.inference(
            inputs=[reference["actions"], reference["latent"]]
        )[0]
        rknn.release()
        expected_flat, actual_flat = expected.reshape(-1), actual.reshape(-1)
        cosine = np.dot(expected_flat, actual_flat) / (
            np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
        )
        print(
            f"{name:8s} shape={actual.shape} "
            f"mae={np.abs(expected - actual).mean():.6f} "
            f"max={np.abs(expected - actual).max():.6f} cos={cosine:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
