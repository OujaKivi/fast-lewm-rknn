#!/usr/bin/env python3
"""Run one complete SmolVLA action-chunk inference and report resource use."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import psutil
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="lerobot/smolvla_base")
    parser.add_argument("--vlm-path", default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be at least 1")
        torch.set_num_threads(args.threads)

    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = args.device
    # The complete SmolVLA checkpoint is loaded below; avoid loading a second
    # copy of the pretrained VLM just to initialize its architecture.
    config.load_vlm_weights = False
    if args.vlm_path:
        config.vlm_model_name = args.vlm_path
    process = psutil.Process(os.getpid())
    before_load = process.memory_info().rss
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval()
    load_seconds = time.perf_counter() - started
    after_load = process.memory_info().rss

    processor = policy.model.vlm_with_expert.processor
    tokens = processor.tokenizer(
        "Pick up the object.", return_tensors="pt", padding=True
    )
    image_key = next(iter(config.image_features))
    side = config.image_features[image_key].shape[-1]
    # A deterministic, non-blank synthetic observation tests the full graph,
    # not task quality or a pretrained action's physical correctness.
    axis = torch.linspace(0, 1, side)
    image = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0)
    image = torch.cat([image, (image[:1] + image[1:]) / 2], dim=0)
    batch = {
        image_key: image.unsqueeze(0).to(args.device),
        OBS_STATE: torch.zeros(
            1, config.input_features[OBS_STATE].shape[0], device=args.device
        ),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(args.device),
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(args.device).bool(),
    }
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_seconds = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()
        inference_seconds.append(round(time.perf_counter() - started, 3))
    result = {
        "model": str(args.model_path),
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "image_key": image_key,
        "image_shape": list(batch[image_key].shape),
        "action_shape": list(actions.shape),
        "action_finite": bool(torch.isfinite(actions).all()),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": inference_seconds,
        "first_inference_seconds": inference_seconds[0],
        "rss_before_load_mb": round(before_load / 1e6, 1),
        "rss_after_load_mb": round(after_load / 1e6, 1),
        "rss_after_inference_mb": round(process.memory_info().rss / 1e6, 1),
        "max_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 if os.uname().sysname == "Linux" else 1e6),
            1,
        ),
    }
    if args.device == "cuda":
        result["cuda_peak_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 1e6, 1
        )
        result["cuda_peak_reserved_mb"] = round(
            torch.cuda.max_memory_reserved() / 1e6, 1
        )
    elif args.device == "mps":
        result["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        result["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["action_finite"]:
        raise SystemExit("SmolVLA returned non-finite actions")


if __name__ == "__main__":
    main()
