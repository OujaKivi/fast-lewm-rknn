#!/usr/bin/env python3
"""Profile the deployed action encoder by operator and candidate population."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = (
    ROOT / "Fast-LeWorldModel"
    if (ROOT / "Fast-LeWorldModel/module.py").exists()
    else Path(__file__).resolve().parent
)
sys.path.insert(0, str(MODEL_DIR))

from module import ActionPrefixEmbedder  # noqa: E402


def load_model(weights_path):
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
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


def profile_batch(model, batch, warmup, repeats):
    generator = torch.Generator().manual_seed(42)
    actions = torch.randn(batch, 5, 10, generator=generator)
    latent = torch.randn(batch, 1, 192, generator=generator)

    with torch.inference_mode():
        for _ in range(warmup):
            model(actions, return_last_only=True, latent=latent)

        wall_ms = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            output = model(actions, return_last_only=True, latent=latent)
            wall_ms.append((time.perf_counter_ns() - start) / 1e6)

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as trace:
            model(actions, return_last_only=True, latent=latent)

    operators = sorted(
        (
            {
                "name": event.key,
                "self_cpu_ms": event.self_cpu_time_total / 1000,
                "cpu_total_ms": event.cpu_time_total / 1000,
                "calls": event.count,
            }
            for event in trace.key_averages()
        ),
        key=lambda row: row["self_cpu_ms"],
        reverse=True,
    )
    return {
        "batch": batch,
        "output_shape": list(output.shape),
        "wall_median_ms": statistics.median(wall_ms),
        "wall_p95_ms": sorted(wall_ms)[int(0.95 * (len(wall_ms) - 1))],
        "wall_min_ms": min(wall_ms),
        "top_operators": operators[:15],
        "all_operators": operators,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path,
        default=MODEL_DIR / "weights/full_model_state.pt",
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 150, 300])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = load_model(args.weights)
    result = {
        "torch_version": torch.__version__,
        "threads": args.threads,
        "weights": str(args.weights),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batches": [
            profile_batch(model, batch, args.warmup, args.repeats)
            for batch in args.batches
        ],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
