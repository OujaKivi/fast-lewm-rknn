#!/usr/bin/env python3
"""Measure RK3588 SmolVLA CPU and NPU-vision hybrid on identical inputs."""

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
from rknnlite.api import RKNNLite


def similarity(reference, actual):
    reference = reference.astype(np.float64).ravel()
    actual = actual.astype(np.float64).ravel()
    difference = np.abs(reference - actual)
    return {
        "cosine": float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual))),
        "mae": float(difference.mean()),
        "p99_abs_error": float(np.percentile(difference, 99)),
        "max_abs_error": float(difference.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--rknn-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
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
        image_key: image.unsqueeze(0),
        OBS_STATE: torch.zeros(1, config.input_features[OBS_STATE].shape[0]),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"],
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].bool(),
    }
    prepared_image = policy.prepare_images(batch)[0][0]
    original_embed = policy.model.vlm_with_expert.embed_image
    original_denoise = policy.model.denoise_step
    denoise_times = []

    def timed_denoise(*args, **kwargs):
        started = time.perf_counter()
        output = original_denoise(*args, **kwargs)
        denoise_times.append(time.perf_counter() - started)
        return output

    policy.model.denoise_step = timed_denoise

    runtime = RKNNLite()
    if runtime.load_rknn(args.rknn_path) != 0:
        raise RuntimeError("RKNN load failed")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("RKNN runtime initialization failed")
    try:
        def npu_embed(image_tensor):
            output = runtime.inference(
                inputs=[image_tensor.detach().numpy()], data_format=["nchw"]
            )[0]
            return torch.from_numpy(output).to(dtype=image_tensor.dtype)

        with torch.inference_mode():
            started = time.perf_counter()
            cpu_embedding = original_embed(prepared_image)
            cpu_vision_seconds = time.perf_counter() - started
            started = time.perf_counter()
            npu_embedding = npu_embed(prepared_image)
            npu_vision_seconds = time.perf_counter() - started

            torch.manual_seed(42)
            noise = torch.randn(1, config.chunk_size, config.max_action_dim)
            started = time.perf_counter()
            cpu_actions = policy.predict_action_chunk(batch.copy(), noise=noise)
            cpu_full_seconds = time.perf_counter() - started
            cpu_denoise_times = denoise_times.copy()
            denoise_times.clear()

            policy.model.vlm_with_expert.embed_image = npu_embed
            started = time.perf_counter()
            hybrid_actions = policy.predict_action_chunk(batch.copy(), noise=noise)
            hybrid_full_seconds = time.perf_counter() - started
            hybrid_denoise_times = denoise_times.copy()

        result = {
            "image_shape": list(prepared_image.shape),
            "embedding_shape": list(cpu_embedding.shape),
            "actions_shape": list(cpu_actions.shape),
            "vision_cpu_seconds": round(cpu_vision_seconds, 3),
            "vision_npu_seconds": round(npu_vision_seconds, 3),
            "full_cpu_seconds": round(cpu_full_seconds, 3),
            "full_hybrid_seconds": round(hybrid_full_seconds, 3),
            "cpu_denoise_seconds": round(sum(cpu_denoise_times), 3),
            "hybrid_denoise_seconds": round(sum(hybrid_denoise_times), 3),
            "denoise_steps": len(cpu_denoise_times),
            "embedding_agreement": similarity(cpu_embedding.numpy(), npu_embedding.numpy()),
            "action_agreement": similarity(cpu_actions.numpy(), hybrid_actions.numpy()),
            "cpu_actions_finite": bool(torch.isfinite(cpu_actions).all()),
            "hybrid_actions_finite": bool(torch.isfinite(hybrid_actions).all()),
        }
    finally:
        runtime.release()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
