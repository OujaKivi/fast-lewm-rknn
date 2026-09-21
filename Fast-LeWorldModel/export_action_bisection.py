"""Export cumulative Action Encoder stages for RKNN accuracy bisection."""

import argparse
import os

import numpy as np
import torch
from torch import nn

from export_action_encoder import build_model


class ActionStage(nn.Module):
    def __init__(self, model, stage):
        super().__init__()
        self.encoder = model.encoder
        self.stage = stage
        self.register_buffer("position", model.position.detach().clone())

    def forward(self, actions, latent):
        actions = self.encoder.embed(self.encoder.token_proj(actions.float()))
        latent_token = self.encoder.latent_proj(latent[:, 0:1, :].float())
        tokens = torch.cat([latent_token, actions], dim=1) + self.position
        if self.stage == "frontend":
            return tokens

        block = self.encoder.temporal_mixer.layers[0]
        norm1 = block.norm1(tokens)
        if self.stage == "block1_norm1":
            return norm1
        attention_module = block.attn
        attention_norm = attention_module.norm(norm1)
        if self.stage == "attention_norm":
            return attention_norm
        qkv = attention_module.to_qkv(attention_norm)
        if self.stage == "attention_qkv":
            return qkv
        batch, token_count, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, token_count, 6, 32).permute(0, 2, 1, 3)
        k = k.reshape(batch, token_count, 6, 32).permute(0, 2, 1, 3)
        v = v.reshape(batch, token_count, 6, 32).permute(0, 2, 1, 3)
        if self.stage == "attention_q":
            return q
        scores = torch.matmul(q, k.transpose(-2, -1)) * attention_module.scale
        if self.stage == "attention_scores":
            return scores
        weights = torch.softmax(scores + attention_module.causal_bias, dim=-1)
        if self.stage == "attention_softmax":
            return weights
        context = torch.matmul(weights, v)
        if self.stage == "attention_context":
            return context
        flattened = context.permute(0, 2, 1, 3).reshape(batch, token_count, -1)
        if self.stage == "attention_flattened":
            return flattened
        projected = attention_module.out_conv(
            flattened.transpose(1, 2)
        ).transpose(1, 2)
        if self.stage == "attention_projected":
            return projected
        attention = block.attn(norm1)
        if self.stage == "block1_attention":
            return attention
        tokens = tokens + attention
        if self.stage == "block1_residual":
            return tokens
        norm2 = block.norm2(tokens)
        if self.stage == "block1_norm2":
            return norm2
        mlp = block.mlp(norm2)
        if self.stage == "block1_mlp":
            return mlp
        tokens = tokens + mlp
        if self.stage == "block1":
            return tokens

        block_count = {"block2": 2, "block3": 3, "final": 3}[self.stage]
        for index in range(1, block_count):
            tokens = self.encoder.temporal_mixer.layers[index](tokens)
        if self.stage == "final":
            tokens = self.encoder.temporal_mixer.norm(tokens)
            tokens = self.encoder.out_norm(tokens[:, 1:6, :])
        return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(7)
    base = build_model(args.checkpoint)
    actions = torch.randn(args.batch, 5, 10)
    latent = torch.randn(args.batch, 1, 192)
    os.makedirs(args.outdir, exist_ok=True)

    stages = (
        "frontend", "block1_norm1", "attention_norm", "attention_qkv",
        "attention_q", "attention_scores", "attention_softmax",
        "attention_context", "attention_flattened", "attention_projected",
        "block1_attention", "block1_residual",
        "block1_norm2", "block1_mlp", "block1", "block2", "block3", "final",
    )
    for name in stages:
        model = ActionStage(base, name).eval()
        with torch.no_grad():
            reference = model(actions, latent).numpy()
        np.savez(
            os.path.join(args.outdir, f"{name}_ref.npz"),
            actions=actions.numpy(), latent=latent.numpy(), output=reference,
        )
        torch.onnx.export(
            model, (actions, latent), os.path.join(args.outdir, f"{name}.onnx"),
            input_names=["actions", "latent"], output_names=["output"],
            opset_version=14, do_constant_folding=True, dynamic_axes=None,
        )
        print(name, reference.shape)


if __name__ == "__main__":
    main()
