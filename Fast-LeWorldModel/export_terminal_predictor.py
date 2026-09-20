"""Export the paper-aligned terminal predictor used by CEM planning.

The action-prefix encoder still consumes all five action blocks. Planning only
uses its final prefix token, so this graph accepts [B, 1, 192] inputs and fuses
predictor + pred_proj into a single [B, 1, 192] output.
"""

import argparse
import os

import torch
from torch import nn

from module import ARPredictor, MLP


EMBED_DIM = 192


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
        if isinstance(state, nn.Module):
            state = state.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    # Lightning/stable-pretraining checkpoints commonly wrap the JEPA module.
    prefixes = ("model.", "module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if state and all(key.startswith(prefix) for key in state):
                state = {key[len(prefix):]: value for key, value in state.items()}
                changed = True
    return state


def _submodule_state(state, name):
    prefixes = (f"{name}.", f"model.{name}.", f"module.{name}.")
    for prefix in prefixes:
        values = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if values:
            return values
    raise KeyError(f"Checkpoint does not contain {name!r} weights")


class TerminalPredictorWithProjection(nn.Module):
    def __init__(self, predictor, pred_proj):
        super().__init__()
        self.predictor = predictor
        self.pred_proj = pred_proj

    def forward(self, latent, terminal_act_emb):
        pred = self.predictor(latent, terminal_act_emb)
        # The graph is intentionally terminal-only; reject accidental T=5 use.
        if pred.shape[1] != 1:
            raise RuntimeError("terminal predictor expects exactly one prefix token")
        return self.pred_proj(pred[:, 0]).unsqueeze(1)


def build_model(checkpoint_path):
    predictor = ARPredictor(
        depth=6,
        mlp_dim=2048,
        input_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        value_heads=16,
        value_dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
        action_fusion_hidden_dim=768,
        action_fusion_zero_init=True,
        token_processing="batch",
    )
    pred_proj = MLP(
        input_dim=EMBED_DIM,
        hidden_dim=2048,
        output_dim=EMBED_DIM,
        norm_fn=nn.BatchNorm1d,
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        # Official releases may pickle the JEPA module instead of a plain state dict.
        # Only use this fallback for a checkpoint supplied and trusted by the caller.
        print(f"weights_only load unavailable ({exc}); loading trusted checkpoint module")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _unwrap_state_dict(checkpoint)
    predictor.load_state_dict(_submodule_state(state, "predictor"), strict=True)
    pred_proj.load_state_dict(_submodule_state(state, "pred_proj"), strict=True)
    return TerminalPredictorWithProjection(predictor, pred_proj).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="./onnx_out")
    parser.add_argument("--batch", type=int, default=300)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    model = build_model(args.checkpoint)
    latent = torch.randn(args.batch, 1, EMBED_DIM)
    terminal_act_emb = torch.randn(args.batch, 1, EMBED_DIM)
    with torch.no_grad():
        output = model(latent, terminal_act_emb)
    assert output.shape == (args.batch, 1, EMBED_DIM)

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(
        args.outdir, f"predictor_terminal_with_proj_b{args.batch}.onnx"
    )
    torch.onnx.export(
        model,
        (latent, terminal_act_emb),
        output_path,
        input_names=["latent", "terminal_act_emb"],
        output_names=["terminal_pred"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Exported {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
