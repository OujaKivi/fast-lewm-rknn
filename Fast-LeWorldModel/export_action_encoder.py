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

    def forward(self, actions, latent):
        # Use a positive, fixed slice instead of x[:, -1:, :]. RKNN Toolkit2
        # accepts the latter during conversion but miscompiles the negative Slice.
        all_prefixes = self.encoder(actions, return_last_only=False, latent=latent)
        return all_prefixes[:, 4:5, :]


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
