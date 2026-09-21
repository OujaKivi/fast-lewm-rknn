"""Export the paper-aligned ViT image encoder with its projector."""

import argparse
import os

import torch
from torch import nn
from transformers import ViTConfig, ViTModel

from module import MLP


class ProjectedViT(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, image):
        cls_token = self.encoder(image).last_hidden_state[:, 0]
        return self.projector(cls_token)


def load_state(checkpoint_path):
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


def build_model(checkpoint_path):
    state = load_state(checkpoint_path)
    config = ViTConfig(
        hidden_size=192, num_attention_heads=3, num_hidden_layers=12,
        intermediate_size=768, patch_size=14, image_size=224,
        num_channels=3, add_pooling_layer=False,
    )
    encoder = ViTModel(config)
    encoder.load_state_dict({
        key.removeprefix("encoder."): value
        for key, value in state.items() if key.startswith("encoder.")
    }, strict=False)
    projector = MLP(
        input_dim=192, hidden_dim=2048, output_dim=192,
        norm_fn=nn.BatchNorm1d,
    )
    projector.load_state_dict({
        key.removeprefix("projector."): value
        for key, value in state.items() if key.startswith("projector.")
    })
    return ProjectedViT(encoder, projector).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="./onnx_out")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    model = build_model(args.checkpoint)
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert model(image).shape == (1, 192)
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, "vit_encoder_projected.onnx")
    torch.onnx.export(
        model, image, output_path,
        input_names=["image"], output_names=["image_embedding"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Exported {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
