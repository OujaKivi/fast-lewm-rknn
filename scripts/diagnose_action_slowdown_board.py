"""Isolate causes of CPU action-encoder slowdown in the heterogeneous loop."""

import json
import random
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/Fast-LeWorldModel")

from module import ActionPrefixEmbedder
from rknnlite.api import RKNNLite


WEIGHTS_PATH = "/root/Fast-LeWorldModel/weights/full_model_state.pt"
PREDICTOR_PATH = "/root/Fast-LeWorldModel/predictor_terminal_with_proj_b300_fp16.rknn"
THERMAL_ROOT = Path("/sys/class/thermal")


def temperature_c():
    values = []
    for zone in THERMAL_ROOT.glob("thermal_zone*"):
        try:
            if (zone / "type").read_text().strip() in {"soc-thermal", "bigcore0-thermal", "bigcore1-thermal", "npu-thermal"}:
                values.append(int((zone / "temp").read_text()) / 1000)
        except OSError:
            pass
    return max(values) if values else None


def build_action_encoder():
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


def run_condition(model, rknn, actions, latent, condition, iterations=30):
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter_ns()
            act_emb = model(actions, return_last_only=True, latent=latent)
            times.append((time.perf_counter_ns() - start) / 1e6)

            if condition == "numpy_copy":
                latent.numpy().astype(np.float32)
                act_emb.numpy().astype(np.float32)
            elif condition == "sleep_18ms":
                time.sleep(0.018)
            elif condition == "npu_predictor":
                rknn.inference(inputs=[
                    latent.numpy().astype(np.float32),
                    act_emb.numpy().astype(np.float32),
                ])
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.set_num_threads(args.threads)
    model = build_action_encoder()
    actions = torch.randn(300, 5, 10)
    latent = torch.randn(300, 1, 192)

    rknn = RKNNLite()
    if rknn.load_rknn(PREDICTOR_PATH) != 0:
        raise RuntimeError("Unable to load predictor RKNN")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("Unable to initialize RKNN runtime")

    conditions = ["action_only", "numpy_copy", "sleep_18ms", "npu_predictor"]
    records = {name: [] for name in conditions}
    try:
        run_condition(model, rknn, actions, latent, "action_only", iterations=5)
        for round_index in range(5):
            order = conditions.copy()
            random.Random(100 + round_index).shuffle(order)
            for condition in order:
                before = temperature_c()
                times = run_condition(model, rknn, actions, latent, condition)
                records[condition].append({
                    "round": round_index,
                    "mean_ms": float(np.mean(times)),
                    "std_ms": float(np.std(times)),
                    "temperature_before_c": before,
                    "temperature_after_c": temperature_c(),
                })
    finally:
        rknn.release()

    summary = {}
    for condition, rounds in records.items():
        means = [item["mean_ms"] for item in rounds]
        summary[condition] = {
            "mean_ms": float(np.mean(means)),
            "round_std_ms": float(np.std(means)),
            "rounds": rounds,
        }
    baseline = summary["action_only"]["mean_ms"]
    for values in summary.values():
        values["delta_vs_action_only_pct"] = (values["mean_ms"] / baseline - 1) * 100
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
