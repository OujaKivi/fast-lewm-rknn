#!/usr/bin/env python3
"""Plot the fixed-observation hardware-aware CEM pilot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/hardware_icem_fixed_observation.json"
OUTPUT = ROOT / "hardware_icem_pilot.png"


def main():
    data = json.loads(DATA.read_text())["results"]
    selected = data[:2]
    labels = ["Fixed 300", "300 -> 150 -> 64"]
    action_ms = [item["mean_action_encoder_ms"] for item in selected]
    predictor_ms = [item["mean_predictor_ms"] for item in selected]
    other_ms = [
        item["mean_total_ms"] - action - predictor
        for item, action, predictor in zip(selected, action_ms, predictor_ms)
    ]

    fig = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.4])
    ax = fig.add_subplot(grid[0])
    x = np.arange(2)
    ax.bar(x, action_ms, color="#4C8BCB", label="CPU Action Encoder")
    ax.bar(x, predictor_ms, bottom=action_ms, color="#E66A65", label="NPU Predictor")
    bottoms = np.asarray(action_ms) + np.asarray(predictor_ms)
    ax.bar(x, other_ms, bottom=bottoms, color="#9AA3A8", label="Other")
    for index, item in enumerate(selected):
        ax.text(index, item["mean_total_ms"] + 35,
                f'{item["mean_total_ms"]:.0f} ms', ha="center", fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Latency per complete replan (ms)")
    ax.set_ylim(0, max(item["mean_total_ms"] for item in selected) * 1.18)
    ax.set_title("Fixed-observation latency (3-run mean)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[1])
    fixed = selected[0]["cem"]["candidate_schedule"]
    tiered = selected[1]["cem"]["candidate_schedule"]
    steps = np.arange(1, len(fixed) + 1)
    ax.step(steps, fixed, where="mid", linewidth=2.5, color="#777777",
            label="Fixed population")
    ax.step(steps, tiered, where="mid", linewidth=2.5, color="#198754",
            label="NPU-aligned tiers")
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("Candidates")
    ax.set_xticks([1, 10, 20, 30])
    ax.set_ylim(0, 330)
    ax.set_title("Schedule maps exactly to fixed RKNN batches")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Hardware-aware CEM pilot on RK3588", fontsize=16, weight="bold")
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
