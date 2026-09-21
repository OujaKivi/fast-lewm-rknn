#!/usr/bin/env python3
"""Measure terminal Predictor RKNN accuracy and latency by fixed batch size."""

import json
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/root/Fast-LeWorldModel")
from module import ARPredictor, MLP  # noqa: E402
from rknnlite.api import RKNNLite  # noqa: E402


WEIGHTS = "/root/Fast-LeWorldModel/weights/full_model_state.pt"
BATCHES = (64, 150, 300)


def build_cpu():
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    predictor = ARPredictor(
        depth=6, mlp_dim=2048, input_dim=192, hidden_dim=192,
        output_dim=192, value_heads=16, value_dim_head=64,
        action_fusion_hidden_dim=768, action_fusion_zero_init=True,
        token_processing="batch",
    )
    predictor.load_state_dict({
        key.removeprefix("predictor."): value
        for key, value in state.items() if key.startswith("predictor.")
    })
    projection = MLP(
        input_dim=192, hidden_dim=2048, output_dim=192,
        norm_fn=nn.BatchNorm1d,
    )
    projection.load_state_dict({
        key.removeprefix("pred_proj."): value
        for key, value in state.items() if key.startswith("pred_proj.")
    })
    return predictor.eval(), projection.eval()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    predictor, projection = build_cpu()
    results = []

    for batch in BATCHES:
        latent = torch.randn(batch, 1, 192)
        action = torch.randn(batch, 1, 192)
        with torch.no_grad():
            reference = projection(predictor(latent, action)[:, 0]).unsqueeze(1)

        runtime = RKNNLite()
        path = f"/root/Fast-LeWorldModel/predictor_terminal_with_proj_b{batch}_fp16.rknn"
        if runtime.load_rknn(path) != 0:
            raise RuntimeError(f"failed to load {path}")
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError(f"failed to initialize {path}")

        inputs = [latent.numpy(), action.numpy()]
        for _ in range(5):
            runtime.inference(inputs=inputs)
        times = []
        output = None
        for _ in range(30):
            start = time.perf_counter()
            output = runtime.inference(inputs=inputs)[0]
            times.append((time.perf_counter() - start) * 1000)
        runtime.release()

        actual = torch.from_numpy(output)
        cosine = torch.nn.functional.cosine_similarity(
            reference.reshape(1, -1), actual.reshape(1, -1)
        ).item()
        error = (reference - actual).abs()
        results.append({
            "batch": batch,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "std_ms": statistics.pstdev(times),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": min(times),
            "max_ms": max(times),
            "cosine_similarity": cosine,
            "mae": error.mean().item(),
            "max_abs_error": error.max().item(),
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
