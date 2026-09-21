"""Validate the fused ViT plus projector RKNN graph against PyTorch on-board."""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

sys.path.insert(0, "/root/Fast-LeWorldModel")

from module import MLP
from rknnlite.api import RKNNLite


WEIGHTS_PATH = "/root/Fast-LeWorldModel/weights/full_model_state.pt"
RKNN_PATH = "/root/Fast-LeWorldModel/vit_encoder_projected_fp16.rknn"


def main():
    torch.manual_seed(7)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    config = ViTConfig(
        hidden_size=192, num_attention_heads=3, num_hidden_layers=12,
        intermediate_size=768, patch_size=14, image_size=224,
        num_channels=3, add_pooling_layer=False,
    )
    encoder = ViTModel(config).eval()
    encoder.load_state_dict({
        key.removeprefix("encoder."): value
        for key, value in state.items() if key.startswith("encoder.")
    }, strict=False)
    projector = MLP(
        input_dim=192, hidden_dim=2048, output_dim=192,
        norm_fn=nn.BatchNorm1d,
    ).eval()
    projector.load_state_dict({
        key.removeprefix("projector."): value
        for key, value in state.items() if key.startswith("projector.")
    })

    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        cpu = projector(encoder(image).last_hidden_state[:, 0]).numpy()

    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_PATH) != 0:
        raise RuntimeError("Unable to load RKNN ViT")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("Unable to initialize RKNN runtime")
    try:
        start = time.perf_counter()
        npu = rknn.inference(inputs=[image.numpy()], data_format=["nchw"])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        rknn.release()

    cpu_flat, npu_flat = cpu.reshape(-1), npu.reshape(-1)
    cosine = np.dot(cpu_flat, npu_flat) / (
        np.linalg.norm(cpu_flat) * np.linalg.norm(npu_flat)
    )
    print(f"shape={npu.shape}")
    print(f"npu_ms={elapsed_ms:.2f}")
    print(f"mae={np.abs(cpu - npu).mean():.6f}")
    print(f"max_abs_error={np.abs(cpu - npu).max():.6f}")
    print(f"cosine_similarity={cosine:.6f}")


if __name__ == "__main__":
    main()
