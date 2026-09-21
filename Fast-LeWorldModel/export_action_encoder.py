"""Export the paper-aligned terminal action-prefix encoder for RKNN."""

import argparse
import os

import torch
from torch import nn

from module import ActionPrefixEmbedder


class TerminalActionEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for block in self.encoder.temporal_mixer.layers:
            block.attn = RKNNFriendlyAttention(block.attn, sequence_length=6)
        position = encoder._sinusoidal_pos_encoding(
            length=6, dim=192, device=torch.device("cpu"), dtype=torch.float32
        )
        self.register_buffer("position", position)

    def forward(self, actions, latent):
        # Keep the six-token graph static and avoid RKNN's SDPA fusion. The
        # attention output projection below also uses Conv1d to avoid a
        # Toolkit2 2.3.2 Transpose/Reshape + Linear miscompile.
        actions = self.encoder.embed(self.encoder.token_proj(actions.float()))
        latent_token = self.encoder.latent_proj(latent[:, 0:1, :].float())
        tokens = torch.cat([latent_token, actions], dim=1) + self.position
        tokens = self.encoder.temporal_mixer(tokens)
        prefixes = self.encoder.out_norm(tokens[:, 1:6, :])
        return prefixes[:, 4:5, :]


class RKNNFriendlyAttention(nn.Module):
    """Equivalent fixed-length causal attention without Trilu/Where/-inf."""

    def __init__(self, attention, sequence_length):
        super().__init__()
        self.heads = attention.heads
        self.scale = attention.scale
        self.norm = attention.norm
        self.to_qkv = attention.to_qkv
        linear = attention.to_out[0]
        self.out_conv = nn.Conv1d(
            linear.in_features, linear.out_features, kernel_size=1, bias=linear.bias is not None
        )
        with torch.no_grad():
            self.out_conv.weight.copy_(linear.weight.unsqueeze(-1))
            if linear.bias is not None:
                self.out_conv.bias.copy_(linear.bias)
        bias = torch.triu(
            torch.full((sequence_length, sequence_length), -10000.0), diagonal=1
        )
        self.register_buffer("causal_bias", bias.reshape(1, 1, sequence_length, sequence_length))

    def forward(self, x, causal=True):
        batch, tokens, _ = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, tokens, self.heads, 32).permute(0, 2, 1, 3)
        k = k.reshape(batch, tokens, self.heads, 32).permute(0, 2, 1, 3)
        v = v.reshape(batch, tokens, self.heads, 32).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores + self.causal_bias, dim=-1)
        output = torch.matmul(weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, -1)
        return self.out_conv(output.transpose(1, 2)).transpose(1, 2)


def _load_state(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
        return state.state_dict() if isinstance(state, nn.Module) else state
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _action_encoder_state(state):
    prefixes = (
        "action_encoder.impl.",
        "model.action_encoder.impl.",
        "module.action_encoder.impl.",
    )
    for prefix in prefixes:
        values = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if values:
            return values
    raise KeyError("Checkpoint does not contain action_encoder.impl weights")


def build_model(checkpoint_path):
    encoder = ActionPrefixEmbedder(
        input_dim=10,
        emb_dim=192,
        use_latent_condition=True,
        latent_dim=192,
        transformer_depth=3,
        transformer_heads=6,
        transformer_dim_head=32,
        transformer_mlp_dim=768,
    )
    encoder.load_state_dict(_action_encoder_state(_load_state(checkpoint_path)), strict=True)
    return TerminalActionEncoder(encoder).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="./onnx_out")
    parser.add_argument("--batch", type=int, default=300)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    model = build_model(args.checkpoint)
    actions = torch.randn(args.batch, 5, 10)
    latent = torch.randn(args.batch, 1, 192)
    with torch.no_grad():
        output = model(actions, latent)
    assert output.shape == (args.batch, 1, 192)

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, f"action_encoder_terminal_b{args.batch}.onnx")
    torch.onnx.export(
        model,
        (actions, latent),
        output_path,
        input_names=["actions", "latent"],
        output_names=["terminal_act_emb"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Exported {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
