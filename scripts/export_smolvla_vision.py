#!/usr/bin/env python3
"""Export the pretrained SmolVLA image encoder and connector for RKNN probing."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class VisionAndConnector(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.vlm_with_expert = policy.model.vlm_with_expert

    def forward(self, image):
        return self.vlm_with_expert.embed_image(image)


class FixedResolutionEmbeddings(torch.nn.Module):
    """Equivalent SmolVLM patch embedding for fully valid 512x512 images."""

    def __init__(self, original):
        super().__init__()
        self.patch_embedding = original.patch_embedding
        self.position_embedding = original.position_embedding
        side = original.num_patches_per_side
        boundaries = torch.arange(1 / side, 1.0, 1 / side)
        fractional = torch.arange(side, dtype=torch.float32) / side * (1 - 1e-6)
        buckets = torch.bucketize(fractional, boundaries, right=True)
        position_ids = (buckets[:, None] * side + buckets[None, :]).flatten()
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, pixel_values, patch_attention_mask):
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.position_ids).unsqueeze(0)


class ExactPatchEmbedding(torch.nn.Module):
    """Express nonoverlapping patch convolution as a 1x1 matrix product."""

    def __init__(self, original):
        super().__init__()
        if original.kernel_size != original.stride or original.padding not in (0, (0, 0), "valid"):
            raise ValueError(
                f"patch convolution must have nonoverlapping patches: "
                f"kernel={original.kernel_size}, stride={original.stride}, padding={original.padding}"
            )
        self.original = original
        self.patch_size = original.kernel_size[0]

    def forward(self, image):
        patches = F.pixel_unshuffle(image, self.patch_size)
        weight = self.original.weight.reshape(self.original.out_channels, -1, 1, 1)
        return F.conv2d(patches, weight, self.original.bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--patch-space-to-depth", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval()
    module = VisionAndConnector(policy).eval()

    height, width = config.resize_imgs_with_padding
    axis_y = torch.linspace(-1, 1, height)
    axis_x = torch.linspace(-1, 1, width)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    image = torch.stack([xx, yy, (xx + yy) / 2], dim=0).unsqueeze(0)
    if args.batch_size == 2:
        image = torch.cat([image, image.flip(-1)], dim=0)
    with torch.inference_mode():
        original_reference = module(image)
        vision_model = module.vlm_with_expert.get_vlm_model().vision_model
        vision_model.embeddings = FixedResolutionEmbeddings(vision_model.embeddings)
        if args.patch_space_to_depth:
            embeddings = vision_model.embeddings
            embeddings.patch_embedding = ExactPatchEmbedding(embeddings.patch_embedding)
        reference = module(image)
    max_error = (original_reference - reference).abs().max().item()
    if max_error > 1e-5:
        raise RuntimeError(f"Fixed-resolution embedding changed output: {max_error}")
    np.save(output_dir / "input.npy", image.numpy())
    np.save(output_dir / "reference.npy", reference.numpy())
    torch.onnx.export(
        module,
        image,
        output_dir / "vision_connector.onnx",
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(
        f"input={list(image.shape)} output={list(reference.shape)} "
        f"embedding_rewrite_max_error={max_error:.2g} "
        f"patch_space_to_depth={args.patch_space_to_depth}"
    )


if __name__ == "__main__":
    main()
