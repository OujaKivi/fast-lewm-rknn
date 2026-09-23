#!/usr/bin/env python3
"""Test candidate-level CPU/NPU coexecution of the deployed action encoder."""

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from rknnlite.api import RKNNLite


MODEL_DIR = Path("/root/Fast-LeWorldModel")
sys.path.insert(0, str(MODEL_DIR))
from module import ActionPrefixEmbedder  # noqa: E402


def load_action_encoder():
    state = torch.load(
        MODEL_DIR / "weights/full_model_state.pt",
        map_location="cpu", weights_only=True,
    )
    model = ActionPrefixEmbedder(
        input_dim=10, emb_dim=192, use_latent_condition=True, latent_dim=192,
        transformer_depth=3, transformer_heads=6, transformer_dim_head=32,
        transformer_mlp_dim=768,
    ).eval()
    model.load_state_dict({
        key.removeprefix("action_encoder.impl."): value
        for key, value in state.items() if key.startswith("action_encoder.impl.")
    })
    return model.requires_grad_(False)


def load_rknn(path):
    context = RKNNLite()
    if context.load_rknn(str(path)) != 0:
        raise RuntimeError(f"Unable to load {path}")
    if context.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError(f"Unable to initialize {path}")
    return context


def median_and_p95(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = load_action_encoder()
    actions = torch.randn(300, 5, 10)
    latent = torch.randn(300, 1, 192)
    actions_np = actions.numpy()
    latent_np = latent.numpy()
    configs = [
        ("cpu_300", 300, 0),
        ("cpu_236_npu_64", 236, 64),
        ("cpu_200_npu_100", 200, 100),
        ("cpu_150_npu_150", 150, 150),
        ("npu_300", 0, 300),
    ]
    action_contexts = {}
    predictor = None
    try:
        for _, _, npu_count in configs:
            if npu_count and npu_count not in action_contexts:
                path = MODEL_DIR / f"action_encoder_terminal_b{npu_count}_fp16_conv.rknn"
                action_contexts[npu_count] = load_rknn(path)
        predictor = load_rknn(
            MODEL_DIR / "predictor_terminal_with_proj_b300_fp16.rknn"
        )

        def npu_action(start, count):
            output = action_contexts[count].inference(
                inputs=[actions_np[start:start + count], latent_np[start:start + count]],
                data_format=["nchw", "nchw"],
            )
            if output is None:
                raise RuntimeError(f"NPU action inference failed for batch {count}")
            return output[0]

        def run(cpu_count, npu_count, executor):
            start = time.perf_counter_ns()
            future = (
                executor.submit(npu_action, cpu_count, npu_count)
                if npu_count else None
            )
            if cpu_count:
                with torch.inference_mode():
                    cpu_output = model(
                        actions[:cpu_count], return_last_only=True,
                        latent=latent[:cpu_count],
                    ).numpy()
            else:
                cpu_output = None
            npu_output = future.result() if future else None
            encoded = (
                np.concatenate([cpu_output, npu_output], axis=0)
                if cpu_count and npu_count else
                (cpu_output if cpu_count else npu_output)
            )
            action_ms = (time.perf_counter_ns() - start) / 1e6

            start = time.perf_counter_ns()
            prediction = predictor.inference(
                inputs=[latent_np, encoded.astype(np.float32, copy=False)],
                data_format=["nchw", "nchw"],
            )
            if prediction is None:
                raise RuntimeError("NPU predictor inference failed")
            predictor_ms = (time.perf_counter_ns() - start) / 1e6
            return encoded, prediction[0], action_ms, predictor_ms

        measurements = {name: [] for name, _, _ in configs}
        with ThreadPoolExecutor(max_workers=1) as executor:
            reference, reference_prediction, _, _ = run(300, 0, executor)
            for _ in range(args.warmup):
                for _, cpu_count, npu_count in configs:
                    run(cpu_count, npu_count, executor)

            for round_index in range(args.repeats):
                order = configs.copy()
                random.Random(42 + round_index).shuffle(order)
                for name, cpu_count, npu_count in order:
                    output, prediction, action_ms, predictor_ms = run(
                        cpu_count, npu_count, executor
                    )
                    measurements[name].append({
                        "action_ms": action_ms,
                        "predictor_ms": predictor_ms,
                        "total_ms": action_ms + predictor_ms,
                        "action_mae_vs_cpu": float(np.mean(np.abs(output - reference))),
                        "action_max_abs_vs_cpu": float(np.max(np.abs(output - reference))),
                        "prediction_mae_vs_cpu": float(
                            np.mean(np.abs(prediction - reference_prediction))
                        ),
                    })
    finally:
        if predictor is not None:
            predictor.release()
        for context in action_contexts.values():
            context.release()

    result = {
        "note": "Fixed-input operator benchmark, not complete CEM or task success",
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "configs": [],
    }
    for name, cpu_count, npu_count in configs:
        samples = measurements[name]
        result["configs"].append({
            "name": name,
            "cpu_candidates": cpu_count,
            "npu_candidates": npu_count,
            "action": median_and_p95([s["action_ms"] for s in samples]),
            "predictor": median_and_p95([s["predictor_ms"] for s in samples]),
            "action_plus_predictor": median_and_p95(
                [s["total_ms"] for s in samples]
            ),
            "action_mae_vs_cpu": statistics.mean(
                s["action_mae_vs_cpu"] for s in samples
            ),
            "action_max_abs_vs_cpu": max(
                s["action_max_abs_vs_cpu"] for s in samples
            ),
            "prediction_mae_vs_cpu": statistics.mean(
                s["prediction_mae_vs_cpu"] for s in samples
            ),
        })
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
