"""Validate the official terminal action encoder against its RKNN graph on-board."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/Fast-LeWorldModel")

from module import ActionPrefixEmbedder
from rknnlite.api import RKNNLite


WEIGHTS_PATH = "/root/Fast-LeWorldModel/weights/full_model_state.pt"
RKNN_PATH = "/root/Fast-LeWorldModel/action_encoder_terminal_b300_fp16_conv.rknn"


def main():
    torch.manual_seed(7)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model = ActionPrefixEmbedder(
        input_dim=10,
        emb_dim=192,
        use_latent_condition=True,
        latent_dim=192,
        transformer_depth=3,
        transformer_heads=6,
        transformer_dim_head=32,
        transformer_mlp_dim=768,
    ).eval()
    model.load_state_dict({
        key.removeprefix("action_encoder.impl."): value
        for key, value in state.items()
        if key.startswith("action_encoder.impl.")
    })

    actions = torch.randn(300, 5, 10)
    latent = torch.randn(300, 1, 192)
    with torch.no_grad():
        cpu = model(actions, return_last_only=True, latent=latent).numpy()

    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_PATH) != 0:
        raise RuntimeError("Unable to load RKNN action encoder")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("Unable to initialize RKNN runtime")
    try:
        start = time.perf_counter()
        npu = rknn.inference(
            inputs=[actions.numpy(), latent.numpy()],
            data_format=["nchw", "nchw"],
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        rknn.release()

    cpu_flat = cpu.reshape(-1)
    npu_flat = npu.reshape(-1)
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
