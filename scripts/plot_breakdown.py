"""Render execution sequences and the latest aligned RK3588 CEM breakdown."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "latest_benchmark.json"
OUTPUT = ROOT / "breakdown.png"
OUTPUT_PDF = ROOT / "breakdown.pdf"


def draw_box(ax, x, y, width, text, color, edge="none"):
    box = FancyBboxPatch(
        (x, y - 0.22), width, 0.44,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color, edgecolor=edge, linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=9)
    return x + width


def arrow(ax, start, end, y):
    ax.add_patch(FancyArrowPatch(
        (start + 0.04, y), (end - 0.04, y), arrowstyle="-|>",
        mutation_scale=10, linewidth=1, color="#69707A",
    ))


def draw_sequence(ax):
    cpu, npu, shared, blocked = "#DCEAFE", "#DDF5E5", "#F7E7C6", "#F6DDDD"
    lanes = [
        ("CPU", 2.15, [(0.9, 1.15, "Images\nCPU", cpu), (2.35, 1.35, "Action encoder\nCPU", cpu), (4.00, 1.50, "Predictor + proj\nCPU", cpu), (5.80, 1.20, "Cost + top-k\nCPU", shared), (7.40, 1.45, "Execute 25 actions\nthen replan", "#E8E9EC")]),
        ("CPU + NPU", 1.35, [(0.9, 1.15, "Images\nNPU", npu), (2.35, 1.35, "Action encoder\nCPU", cpu), (4.00, 1.50, "Predictor + proj\nNPU", npu), (5.80, 1.20, "Cost + top-k\nCPU", shared), (7.40, 1.45, "Execute 25 actions\nthen replan", "#E8E9EC")]),
    ]
    for name, y, boxes in lanes:
        ax.text(0.02, y, name, ha="left", va="center", fontsize=10, fontweight="bold")
        ends = [draw_box(ax, x, y, width, label, color) for x, width, label, color in boxes]
        for index in range(len(boxes) - 1):
            arrow(ax, ends[index], boxes[index + 1][0], y)
        ax.text(4.65, y + 0.32, "CEM body repeats 30x", ha="center", fontsize=8, color="#60666F")

    y = 0.52
    ax.text(0.02, y, "All NPU", ha="left", va="center", fontsize=10, fontweight="bold")
    draw_box(ax, 0.9, y, 7.95, "All-NPU not selected: accurate Action Encoder NPU is slower (136 ms vs 67 ms on CPU)", blocked, edge="#C75D5D")
    ax.set_xlim(0, 9.1)
    ax.set_ylim(0.12, 2.62)
    ax.axis("off")
    ax.set_title("One replan: S=300, 30 CEM iterations, then 25 open-loop actions", loc="left", fontsize=14, pad=8)


def draw_breakdown(ax, data):
    modes = ["CPU", "CPU + NPU FP16"]
    records = [data["cpu"], data["heterogeneous_fp16"]]
    stages = [("Image encoder", "image_encoder_ms", "#3B82C4"), ("Action encoder", "action_encoder_ms", "#45A96B"), ("Terminal predictor + projection", "predictor_with_projection_ms", "#E45D5D"), ("Cost + CEM update", None, "#E5A94D"), ("Other", "other_ms", "#8B9098")]
    bottoms = [0.0, 0.0]
    for label, key, color in stages:
        values = [r["cost_ms"] + r["cem_update_ms"] if key is None else r[key] for r in records]
        bars = ax.bar(modes, values, bottom=bottoms, width=0.52, label=label, color=color)
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 80:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.0f} ms", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            elif key == "image_encoder_ms":
                ax.text(bar.get_x() + bar.get_width() / 2, bottom + value + 45, f"{value:.0f} ms", ha="center", va="center", color="#173F63", fontsize=9, fontweight="bold")
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for index, record in enumerate(records):
        amortized = record["total_ms"] / data["workload"]["primitive_actions_per_plan"]
        ax.text(index, record["total_ms"] + 100,
                f"{record['total_ms'] / 1000:.2f} s / replan\n{amortized:.0f} ms / action (amortized)",
                ha="center", fontsize=11, linespacing=1.35)
    ax.text(0.5, max(r["total_ms"] for r in records) * 1.135,
            f"CPU + NPU is {data['heterogeneous_fp16']['speedup_vs_cpu']:.2f}x faster per replan",
            ha="center", fontsize=11, fontweight="bold", color="#263238")
    ax.set_title("Measured latency per replan (one 25-action plan)", loc="left", fontsize=14, pad=10)
    ax.set_ylabel("Latency per replan (ms), mean of 5 requests")
    ax.set_ylim(0, max(r["total_ms"] for r in records) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    data = json.loads(RESULTS.read_text())
    fig = plt.figure(figsize=(11.2, 9.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.75], hspace=0.23)
    draw_sequence(fig.add_subplot(grid[0]))
    draw_breakdown(fig.add_subplot(grid[1]), data)
    fig.suptitle("Aligned Fast-LeWM CEM on RK3588", fontsize=17, y=0.985)
    fig.text(0.5, 0.015, "Amortized ms/action = replan latency / 25; it is not per-action feedback latency. Both paths use the official 6x32 action encoder and 16x64 predictor.", ha="center", fontsize=8.5, color="#555555")
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    print(OUTPUT)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
