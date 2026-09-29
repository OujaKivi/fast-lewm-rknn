#!/usr/bin/env python3
"""Export fixed-shape SDPA probes for the SmolVLA vision attention geometry."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


class Attention(torch.nn.Module):
    def __init__(self, batch_size, zero_mask, derived_mask):
        super().__init__()
        mask = torch.zeros(batch_size, 1, 1024, 1024) if zero_mask else None
        self.register_buffer("mask", mask)
        self.derived_mask = derived_mask

    def forward(self, query, key, value):
        mask = self.mask
        if self.derived_mask:
            valid = torch.ones(query.shape[0], query.shape[2], device=query.device)
            mask = _prepare_4d_attention_mask(valid, query.dtype)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heads", type=int, choices=[3, 6, 12], required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=1)
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--zero-mask", action="store_true")
    masks.add_argument("--derived-mask", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    inputs = tuple(torch.randn(args.batch_size, args.heads, 1024, 64) for _ in range(3))
    torch.onnx.export(
        Attention(args.batch_size, args.zero_mask, args.derived_mask).eval(),
        inputs,
        output_dir / "attention.onnx",
        input_names=["query", "key", "value"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print(f"heads={args.heads} shape={list(inputs[0].shape)} zero_mask={args.zero_mask} derived_mask={args.derived_mask}")


if __name__ == "__main__":
    main()
