#!/usr/bin/env python3
"""Capture actual rotated action queries for the real layer-1 graph audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from probe_smolvla_denoise import CachedDenoiseStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval().float()
    trunk = policy.model.vlm_with_expert
    cache = tuple(torch.from_numpy(np.load(args.data_dir / f"{kind}_{layer}.npy"))
                  for layer in range(trunk.num_vlm_layers) for kind in ("key", "value"))
    suffix = torch.from_numpy(np.load(args.data_dir / "suffix.npy"))
    wrapper = CachedDenoiseStep(policy.model, torch.ones(1, cache[0].shape[1], dtype=torch.bool),
                               trunk.num_vlm_layers).eval()
    original = trunk.eager_attention_forward
    captured = []

    def capture(mask, batch, dim, query, key, value):
        captured.append((mask.clone(), query.clone(), key.clone(), value.clone()))
        return original(mask, batch, dim, query, key, value)

    generator = torch.Generator().manual_seed(812)
    report = {"scope": "Real checkpoint activations; synthetic fixed exported prefix; two different suffixes, not a robot observation or dependent flow trajectory"}
    with torch.inference_mode():
        try:
            trunk.eager_attention_forward = capture
            for index, variant in enumerate((suffix, suffix + 0.1 * torch.randn(suffix.shape, generator=generator))):
                captured.clear()
                wrapper(variant, *cache)
                mask, query, key, value = captured[1]
                if not mask.all() or list(query.shape) != [1, 50, 15, 64]:
                    raise ValueError("Unexpected real cross attention")
                np.save(args.output_dir / f"query_{index}.npy", query.numpy())
                if index == 0:
                    fixed = (key.clone(), value.clone())
                elif not all(torch.equal(a, b) for a, b in zip(fixed, (key, value))):
                    raise ValueError("Fixed conditioning changed with suffix")
        finally:
            trunk.eager_attention_forward = original
    for offset, kind in enumerate(("key", "value")):
        np.save(args.output_dir / f"{kind}_1.npy", cache[2 + offset].numpy())
    report["cross_kv_invariant"] = True
    (args.output_dir / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
