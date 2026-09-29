#!/usr/bin/env python3
"""Audit real cached action attention masks/GQA before a backend experiment."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from probe_smolvla_denoise import CachedDenoiseStep


def metrics(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError("Invalid attention output")
    delta = (actual.float() - expected.float()).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "bitwise_equal": bool(np.array_equal(actual.numpy().view(np.uint8),
                                                  expected.numpy().view(np.uint8)))}


def measure(operation, reference, repeats):
    for _ in range(3):
        result = operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples,
            "vs_eager": metrics(result, reference)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval().float()
    trunk = policy.model.vlm_with_expert
    layers = trunk.num_vlm_layers
    cache = tuple(torch.from_numpy(np.load(args.data_dir / f"{kind}_{layer}.npy"))
                  for layer in range(layers) for kind in ("key", "value"))
    suffix = torch.from_numpy(np.load(args.data_dir / "suffix.npy"))
    prefix_length = cache[0].shape[1]
    wrapper = CachedDenoiseStep(policy.model, torch.ones(1, prefix_length, dtype=torch.bool), layers).eval()
    original = trunk.eager_attention_forward
    groups = trunk.num_attention_heads // trunk.num_key_value_heads

    def expand_kv(tensor):
        batch, length, heads, dim = tensor.shape
        return tensor[:, :, :, None, :].expand(batch, length, heads, groups, dim).reshape(
            batch, length, heads * groups, dim).permute(0, 2, 1, 3).contiguous()

    def fused(mask, batch, dim, query, key, value):
        output = F.scaled_dot_product_attention(query.permute(0, 2, 1, 3),
                                               expand_kv(key), expand_kv(value),
                                               attn_mask=mask[:, None])
        return output.permute(0, 2, 1, 3).reshape(batch, query.shape[1], -1)

    captures = []

    def capture(mask, batch, dim, query, key, value):
        captures.append((mask.clone(), batch, dim, query.clone(), key.clone(), value.clone()))
        return original(mask, batch, dim, query, key, value)

    with torch.inference_mode():
        generator = torch.Generator().manual_seed(812)
        variants = (suffix, suffix + 0.1 * torch.randn(suffix.shape, generator=generator))
        by_suffix = []
        references = []
        try:
            trunk.eager_attention_forward = capture
            for variant in variants:
                captures.clear()
                references.append(wrapper(variant, *cache))
                by_suffix.append(list(captures))
        finally:
            trunk.eager_attention_forward = original
        if any(len(values) != layers for values in by_suffix):
            raise ValueError("Expected exactly one action attention per layer")

        reports = []
        representatives = {}
        for layer, (first, second) in enumerate(zip(*by_suffix, strict=True)):
            mask, batch, dim, query, key, value = first
            geometry = f"Q{query.shape[1]}_K{key.shape[1]}"
            kind = "cross" if key.shape[1] == prefix_length else "self"
            invariant = [torch.equal(a, b) for a, b in zip(first[4:], second[4:])]
            prefix_visibility = bool(mask[:, :, :prefix_length].all())
            suffix_causal = None
            if kind == "self":
                suffix_causal = bool(torch.equal(mask[:, :, prefix_length:],
                                                torch.ones(1, query.shape[1], query.shape[1],
                                                           dtype=torch.bool).tril()))
            reports.append({"layer": layer, "kind": kind, "query": list(query.shape),
                            "key": list(key.shape), "value": list(value.shape),
                            "mask": list(mask.shape), "masked_entries": int((~mask).sum()),
                            "prefix_all_visible": prefix_visibility, "suffix_causal": suffix_causal,
                            "kv_equal_across_two_suffixes": invariant})
            representatives.setdefault(geometry, first)
            if kind == "cross" and not all(invariant):
                raise ValueError("Cross-attention K/V changed with the suffix")

        timing = {}
        for geometry, arguments in representatives.items():
            mask, batch, dim, query, key, value = arguments
            reference = original(*arguments)
            timing[geometry] = {
                "eager_actual_mask_gqa": measure(lambda: original(*arguments), reference, args.repeats),
                "fused_dynamic_gqa_layout": measure(lambda: fused(*arguments), reference, args.repeats),
            }
            if key.shape[1] == prefix_length:
                start = time.perf_counter()
                prepared_key, prepared_value = expand_kv(key), expand_kv(value)
                prepare_ms = (time.perf_counter() - start) * 1000

                def static_kv():
                    output = F.scaled_dot_product_attention(query.permute(0, 2, 1, 3),
                                                           prepared_key, prepared_value,
                                                           attn_mask=mask[:, None])
                    return output.permute(0, 2, 1, 3).reshape(batch, query.shape[1], -1)

                timing[geometry]["fused_static_kv"] = measure(static_kv, reference, args.repeats)
                timing[geometry]["static_kv_prepare_ms_single_sample"] = prepare_ms
                timing[geometry]["static_expanded_fp32_kv_bytes"] = (
                    prepared_key.numel() + prepared_value.numel()) * 4

        try:
            trunk.eager_attention_forward = fused
            substituted = [wrapper(variant, *cache) for variant in variants]
        finally:
            trunk.eager_attention_forward = original

    result = {
        "scope": "Real checkpoint action activations from exported synthetic prefix and two suffixes; CPU FP32 only, not NPU activations or a rollout",
        "boundary_scope": "No NPU producer/consumer, device cache synchronization, or native conversions timed; not an NPU speedup",
        "precision": "Algorithmically equivalent FP32 SDPA with no skipped work; accumulation order differs, so unchanged numerical outputs/closed-loop quality are not established",
        "torch_version": torch.__version__, "affinity": sorted(os.sched_getaffinity(0)),
        "threads": 4, "num_attention_heads": trunk.num_attention_heads,
        "num_key_value_heads": trunk.num_key_value_heads, "groups": groups,
        "layer_audit": reports, "representative_timings": timing,
        "full_32_layer_velocity_vs_cpu_eager": [metrics(a, b) for a, b in zip(substituted, references)],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"timings": {key: {name: data.get("median_ms") for name, data in values.items()
                                       if isinstance(data, dict)} for key, values in timing.items()},
                      "full_velocity": result["full_32_layer_velocity_vs_cpu_eager"]}, indent=2))


if __name__ == "__main__":
    main()
