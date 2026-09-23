#!/usr/bin/env python3
"""Profile matched-input SmolVLA stages and save actions for device parity."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def summary(values):
    return {
        "median_ms": round(float(np.median(values)), 3),
        "samples_ms": [round(float(value), 3) for value in values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vision-rknn")
    parser.add_argument("--denoise-rknn")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.vision_rknn or args.denoise_rknn:
        if args.device != "cpu":
            parser.error("RKNN paths require --device cpu")
        from rknnlite.api import RKNNLite

    torch.set_num_threads(args.threads)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = args.device
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval()
    image_key = next(iter(config.image_features))
    side = config.image_features[image_key].shape[-1]
    axis = torch.linspace(0, 1, side)
    image = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0)
    image = torch.cat([image, (image[:1] + image[1:]) / 2], dim=0)
    tokens = policy.model.vlm_with_expert.processor.tokenizer(
        "Pick up the object.", return_tensors="pt", padding=True
    )
    batch = {
        image_key: image.unsqueeze(0).to(args.device),
        OBS_STATE: torch.zeros(
            1, config.input_features[OBS_STATE].shape[0], device=args.device
        ),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(args.device),
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(args.device).bool(),
    }
    torch.manual_seed(42)
    noise = torch.randn(1, config.chunk_size, config.max_action_dim).to(args.device)

    runtimes = []

    def load_rknn(path):
        runtime = RKNNLite()
        if runtime.load_rknn(path) != 0:
            raise RuntimeError(f"RKNN load failed: {path}")
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError(f"RKNN initialization failed: {path}")
        runtimes.append(runtime)
        return runtime

    vision_runtime = load_rknn(args.vision_rknn) if args.vision_rknn else None
    denoise_runtime = load_rknn(args.denoise_rknn) if args.denoise_rknn else None
    original_embed = policy.model.vlm_with_expert.embed_image
    original_forward = policy.model.vlm_with_expert.forward
    original_denoise = policy.model.denoise_step
    current = None

    def timed_embed(image_tensor):
        synchronize(args.device)
        started = time.perf_counter()
        if vision_runtime is None:
            output = original_embed(image_tensor)
        else:
            value = vision_runtime.inference(
                inputs=[image_tensor.detach().numpy()], data_format=["nchw"]
            )[0]
            output = torch.from_numpy(value).to(dtype=image_tensor.dtype)
        synchronize(args.device)
        current["vision_ms"] += (time.perf_counter() - started) * 1e3
        return output

    def timed_forward(*forward_args, **forward_kwargs):
        is_prefix = forward_kwargs.get("fill_kv_cache", False)
        if is_prefix:
            synchronize(args.device)
            started = time.perf_counter()
        output = original_forward(*forward_args, **forward_kwargs)
        if is_prefix:
            synchronize(args.device)
            current["prefix_ms"] += (time.perf_counter() - started) * 1e3
        return output

    def timed_denoise(prefix_pad_masks, past_key_values, x_t, timestep):
        synchronize(args.device)
        started = time.perf_counter()
        if denoise_runtime is None:
            output = original_denoise(prefix_pad_masks, past_key_values, x_t, timestep)
        else:
            if prefix_pad_masks.shape != (1, 70) or not bool(prefix_pad_masks.all()):
                raise ValueError("Denoise RKNN requires 70 valid prefix tokens")
            if len(past_key_values) != 16:
                raise ValueError("Denoise RKNN requires 16 cache layers")
            suffix, _, _ = policy.model.embed_suffix(x_t, timestep)
            inputs = [suffix.detach().float().numpy()]
            for index in range(16):
                for name in ("key_states", "value_states"):
                    value = past_key_values[index][name].detach().float().numpy()
                    inputs.append(value.transpose(0, 2, 3, 1).copy())
            output = torch.from_numpy(
                denoise_runtime.inference(inputs=inputs, data_format=None)[0].copy()
            )
        synchronize(args.device)
        current["denoise_ms"] += (time.perf_counter() - started) * 1e3
        current["denoise_steps"] += 1
        return output

    policy.model.vlm_with_expert.embed_image = timed_embed
    policy.model.vlm_with_expert.forward = timed_forward
    policy.model.denoise_step = timed_denoise
    records = []
    try:
        with torch.inference_mode():
            for run in range(args.warmups + args.repeats):
                current = {"vision_ms": 0.0, "prefix_ms": 0.0, "denoise_ms": 0.0, "denoise_steps": 0}
                synchronize(args.device)
                started = time.perf_counter()
                actions = policy.predict_action_chunk(batch.copy(), noise=noise)
                synchronize(args.device)
                current["total_ms"] = (time.perf_counter() - started) * 1e3
                current["other_ms"] = current["total_ms"] - sum(
                    current[name] for name in ("vision_ms", "prefix_ms", "denoise_ms")
                )
                if run >= args.warmups:
                    records.append(current)
        actions_np = actions.detach().float().cpu().numpy()
    finally:
        for runtime in runtimes:
            runtime.release()

    result = {
        "device": args.device,
        "vision_rknn": bool(args.vision_rknn),
        "denoise_rknn": bool(args.denoise_rknn),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "threads": args.threads,
        "image_shape": list(batch[image_key].shape),
        "action_shape": list(actions_np.shape),
        "action_finite": bool(np.isfinite(actions_np).all()),
        "denoise_steps": [record["denoise_steps"] for record in records],
        "timing": {
            name: summary([record[name] for record in records])
            for name in ("vision_ms", "prefix_ms", "denoise_ms", "other_ms", "total_ms")
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    np.save(path.with_suffix(".npy"), actions_np)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
