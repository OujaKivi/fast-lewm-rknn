#!/usr/bin/env python3
"""Check exact cross-attention KV projection hoisting on an exported prefix."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rknn-model", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval().float()
    model = policy.model
    trunk = model.vlm_with_expert
    layers = trunk.num_vlm_layers
    cache = tuple(torch.from_numpy(np.load(args.data_dir / f"{kind}_{layer}.npy"))
                  for layer in range(layers) for kind in ("key", "value"))
    suffix = torch.from_numpy(np.load(args.data_dir / "suffix.npy"))
    mask = torch.ones((1, cache[0].shape[1]), dtype=torch.bool)
    wrapper = CachedDenoiseStep(model, mask, layers).eval()
    expert_layers = trunk.get_model_layers(
        [trunk.get_vlm_model().text_model, trunk.lm_expert])[1]
    projected = []
    for layer, expert in enumerate(expert_layers):
        if expert is None or "cross" not in trunk.attention_mode:
            continue
        if trunk.self_attn_every_n_layers > 0 and layer % trunk.self_attn_every_n_layers == 0:
            continue
        for offset, name in enumerate(("k_proj", "v_proj")):
            module = getattr(expert.self_attn, name)
            value = cache[2 * layer + offset].to(module.weight.dtype)
            value = value.reshape(*value.shape[:2], -1)
            projected.append((module, value))

    with torch.inference_mode():
        projection_times = []
        for _ in range(6):
            start = time.perf_counter()
            prepared = [module(value) for module, value in projected]
            projection_times.append((time.perf_counter() - start) * 1000)
        generator = torch.Generator().manual_seed(812)
        variants = [suffix, suffix + 0.1 * torch.randn(suffix.shape, generator=generator)]
        references = [wrapper(value, *cache) for value in variants]
        originals = [module.forward for module, _ in projected]
        calls = [0]

        def fixed_projection(expected, output):
            def forward(value):
                if not torch.equal(value, expected):
                    raise RuntimeError("Projection input changed with the action suffix")
                calls[0] += 1
                return output
            return forward

        try:
            for (module, expected), output in zip(projected, prepared):
                module.forward = fixed_projection(expected, output)
            actual = [wrapper(value, *cache) for value in variants]
        finally:
            for (module, _), original in zip(projected, originals):
                module.forward = original
    result = {
        "scope": "real checkpoint, fixed prefix, two distinct synthetic action suffixes; FP32 CPU",
        "layers": layers,
        "hoisted_projections": len(projected),
        "checked_calls": calls[0],
        "projection_input_shapes": sorted({str(list(value.shape)) for _, value in projected}),
        "projection_weight_shapes": sorted({str(list(module.weight.shape)) for module, _ in projected}),
        "projected_bytes": sum(value.numel() * value.element_size() for value in prepared),
        "projection_once_median_ms": float(np.median(projection_times[1:])),
        "output_max_abs_errors": [float((a - b).abs().max()) for a, b in zip(references, actual)],
        "bitwise_equal": [torch.equal(a, b) for a, b in zip(references, actual)],
        "note": "CPU projection timing is not NPU savings. No full denoising rollout or recompiled NPU graph tested.",
    }
    if args.rknn_model:
        from rknnlite.api import RKNNLite

        runtime = RKNNLite()
        inputs = [suffix.numpy(), *(value.numpy().transpose(0, 2, 3, 1).copy() for value in cache)]

        def prepare():
            with torch.inference_mode():
                return [module(value) for module, value in projected]

        durations = {"serial": [], "overlap": []}
        overlap_error = 0.0
        try:
            assert runtime.load_rknn(str(args.rknn_model)) == 0
            assert runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
            reference = runtime.inference(inputs=inputs, data_format=None)[0].copy()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(prepare).result()
                for iteration in range(10):
                    order = list(durations) if iteration % 2 == 0 else list(reversed(durations))
                    for mode in order:
                        start = time.perf_counter()
                        if mode == "overlap":
                            future = pool.submit(prepare)
                        else:
                            cpu_result = prepare()
                        output = runtime.inference(inputs=inputs, data_format=None)[0]
                        if mode == "overlap":
                            cpu_result = future.result()
                        durations[mode].append((time.perf_counter() - start) * 1000)
                        assert all(torch.equal(a, b) for a, b in zip(prepared, cpu_result))
                        overlap_error = max(overlap_error, float(np.max(np.abs(output - reference))))
            result["preparation_overlap"] = {
                mode: {"median_ms": float(np.median(samples)),
                       "p95_ms": float(np.percentile(samples, 95))}
                for mode, samples in durations.items()
            }
            result["overlap_npu_output_max_abs_error"] = overlap_error
            result["overlap_scope"] = "One original NPU step plus CPU preparation; warm NPU graph not implemented."
        finally:
            runtime.release()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
