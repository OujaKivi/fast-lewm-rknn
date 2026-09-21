"""Measure accurate RKNN Action Encoder micro-batches over 300 candidates."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/Fast-LeWorldModel")

from module import ActionPrefixEmbedder
from rknnlite.api import RKNNLite


WEIGHTS_PATH = "/root/Fast-LeWorldModel/weights/full_model_state.pt"
BATCHES = (1, 16, 32, 64, 100, 150, 300)


def build_cpu():
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model = ActionPrefixEmbedder(
        input_dim=10, emb_dim=192, use_latent_condition=True, latent_dim=192,
        transformer_depth=3, transformer_heads=6, transformer_dim_head=32,
        transformer_mlp_dim=768,
    ).eval()
    model.load_state_dict({
        key.removeprefix("action_encoder.impl."): value
        for key, value in state.items() if key.startswith("action_encoder.impl.")
    })
    return model


def infer_300(rknn, batch, actions, latent):
    outputs = []
    for start in range(0, 300, batch):
        count = min(batch, 300 - start)
        action_chunk = np.zeros((batch, 5, 10), dtype=np.float32)
        latent_chunk = np.zeros((batch, 1, 192), dtype=np.float32)
        action_chunk[:count] = actions[start:start + count]
        latent_chunk[:count] = latent[start:start + count]
        outputs.append(rknn.inference(
            inputs=[action_chunk, latent_chunk], data_format=["nchw", "nchw"]
        )[0][:count])
    return np.concatenate(outputs, axis=0)


def main():
    torch.manual_seed(17)
    torch.set_num_threads(4)
    actions_t = torch.randn(300, 5, 10)
    latent_t = torch.randn(300, 1, 192)
    with torch.no_grad():
        reference = build_cpu()(actions_t, return_last_only=True, latent=latent_t).numpy()
    actions, latent = actions_t.numpy(), latent_t.numpy()

    for batch in BATCHES:
        path = f"/root/Fast-LeWorldModel/action_encoder_terminal_b{batch}_fp16_conv.rknn"
        rknn = RKNNLite()
        if rknn.load_rknn(path) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError(f"Unable to initialize {path}")
        try:
            infer_300(rknn, batch, actions, latent)
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                actual = infer_300(rknn, batch, actions, latent)
                samples.append((time.perf_counter() - start) * 1000)
        finally:
            rknn.release()
        expected_flat, actual_flat = reference.reshape(-1), actual.reshape(-1)
        cosine = np.dot(expected_flat, actual_flat) / (
            np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
        )
        print(
            f"batch={batch:3d} calls={(300 + batch - 1) // batch:3d} "
            f"total_ms={np.mean(samples):8.2f} std_ms={np.std(samples):5.2f} "
            f"mae={np.abs(reference - actual).mean():.6f} cos={cosine:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
