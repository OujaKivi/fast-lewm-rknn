"""Render the latest aligned RK3588 CEM latency breakdown."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "latest_benchmark.json"
OUTPUT = ROOT / "breakdown.png"


def main():
    data = json.loads(RESULTS.read_text())
    modes = ["CPU", "CPU + NPU FP16"]
    records = [data["cpu"], data["heterogeneous_fp16"]]
    stages = [
        ("Image encoder", "image_encoder_ms", "#3B82C4"),
        ("Action encoder", "action_encoder_ms", "#45A96B"),
        ("Terminal predictor + projection", "predictor_with_projection_ms", "#E45D5D"),
        ("Cost + CEM update", None, "#E5A94D"),
        ("Other", "other_ms", "#8B9098"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bottoms = [0.0, 0.0]
    for label, key, color in stages:
        if key is None:
            values = [r["cost_ms"] + r["cem_update_ms"] for r in records]
        else:
            values = [r[key] for r in records]
        bars = ax.bar(modes, values, bottom=bottoms, width=0.56, label=label, color=color)
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 80:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.0f} ms",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for index, record in enumerate(records):
        ax.text(index, record["total_ms"] + 110, f"{record['total_ms'] / 1000:.2f} s", ha="center", fontsize=13)

    speedup = data["heterogeneous_fp16"]["speedup_vs_cpu"]
    ax.text(1, records[1]["total_ms"] + 520, f"{speedup:.2f}x vs CPU", ha="center", fontsize=11)
    ax.set_title("Aligned Fast-LeWM CEM latency on RK3588", fontsize=16, pad=18)
    ax.set_ylabel("One complete CEM solve (ms)")
    ax.set_ylim(0, max(r["total_ms"] for r in records) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        -0.12,
        "S=300, 30 iterations, top-k=30; terminal-only predictor [300,1,192]",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
