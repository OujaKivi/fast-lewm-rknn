#!/usr/bin/env python3
"""Write exact SmolVLA profile inputs for a vla.cpp parity experiment."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    side = 256
    axis = torch.linspace(0, 1, side)
    image = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0)
    image = torch.cat([image, (image[:1] + image[1:]) / 2], dim=0)
    image = F.interpolate(image[None], size=(512, 512), mode="bilinear", align_corners=False)[0]
    image.permute(1, 2, 0).contiguous().numpy().astype(np.float32).tofile(
        directory / "image_f32_rgb_01.bin"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokens = tokenizer("Pick up the object.", return_tensors="pt")["input_ids"][0]
    tokens.numpy().astype(np.int32).tofile(directory / "tokens_i32.bin")

    torch.manual_seed(42)
    noise = torch.randn(1, 50, 32)
    noise.numpy().astype(np.float32).tofile(directory / "noise_f32.bin")
    np.zeros(32, dtype=np.float32).tofile(directory / "state_f32.bin")

    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.tokenizer_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval()
    with torch.inference_mode():
        image_embedding = policy.model.vlm_with_expert.embed_image(image[None] * 2 - 1)
    image_embedding.float().numpy().astype(np.float32).tofile(
        directory / "image_emb_f32.bin"
    )
    print(f"wrote {directory}, {len(tokens)} language tokens")


if __name__ == "__main__":
    main()
