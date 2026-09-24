#!/usr/bin/env python3
"""Plot matched SmolVLA stage latency and within-device stage shares."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

STAGES = (
    ("Vision", "vision_ms", "#0072B2"),
    ("Prefill", "prefill_ms", "#009E73"),
    ("Denoising", "denoise_ms", "#D55E00"),
    ("Other", "other_ms", "#747474"),
)
CONFIGS = (
    ("rk3588_cpu", "RK3588\nCPU", "RK3588 CPU"),
    ("rk3588_npu_vision_denoise", "NPU vision\nCPU prefill\nNPU denoising", "RK3588 NPU vision + CPU prefill + NPU denoising"),
    ("rk3588_npu_vision_prefill_denoise", "Vision + prefill\n+ denoising\non NPU", "RK3588 NPU vision + prefill\n+ NPU denoising"),
    ("i5_cpu", "i5-13490F\nCPU", "i5-13490F CPU"),
    ("mac_mps", "Apple M5 Pro\nMPS", "MacBook Pro (Apple M5 Pro, 16-core GPU)"),
    ("rtx5060_cuda", "RTX 5060\nCUDA", "RTX 5060 CUDA"),
)


def fmt_time(ms, decimals=2):
    if ms >= 1000:
        return f"{ms / 1000:.{decimals}f} s"
    if ms >= 100:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def fmt_share(value):
    return f"{value:.2f}%" if value < 1 else f"{value:.1f}%"


def load_profiles():
    profiles = []
    for stem, short_name, full_name in CONFIGS:
        record = json.loads((ROOT / "results" / f"smolvla_profile_{stem}.json").read_text())
        timing = record["timing"]
        total = float(timing["total_ms"]["median_ms"])
        values = [float(timing[key]["median_ms"]) for _, key, _ in STAGES[:3]]
        values.append(max(0.0, total - sum(values)))
        profiles.append({
            "short_name": short_name,
            "full_name": full_name,
            "total": total,
            "values": values,
        })
    return profiles


def save_fig(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    svg_path = OUT / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def stacked_bars(profiles):
    fig = plt.figure(figsize=(17.2, 7.6), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.7, 0.62, 1.05), wspace=0.27)
    groups = (
        (profiles[:3], 44000, "RK3588", True),
        (profiles[3:4], 2600, "Host CPU", True),
        (profiles[4:], 270, "Accelerators", False),
    )
    for panel, (items, y_limit, title, seconds_axis) in enumerate(groups):
        ax = fig.add_subplot(grid[panel])
        x = np.arange(len(items), dtype=float)
        width = 0.68 if len(items) > 1 else 0.55
        for index, item in enumerate(items):
            bottom = 0.0
            for (stage, _, color), value in zip(STAGES, item["values"]):
                ax.bar(index, value, width=width, bottom=bottom,
                       color=color, edgecolor="white", linewidth=0.8)
                if value >= y_limit * 0.065:
                    ax.text(index, bottom + value / 2, fmt_time(value),
                            ha="center", va="center", color="white",
                            fontsize=9.5, weight="bold")
                elif panel == 2 and stage == "Prefill":
                    on_right = index == 0
                    edge = index + width / 2 if on_right else index - width / 2
                    ax.annotate(fmt_time(value), xy=(edge, bottom + value / 2),
                                xytext=(8 if on_right else -8, 0),
                                textcoords="offset points",
                                ha="left" if on_right else "right", va="center",
                                fontsize=9, weight="bold", color="#007f60")
                bottom += value
            ax.text(index, bottom + y_limit * 0.018, fmt_time(item["total"]),
                    ha="center", va="bottom", fontsize=10.5, weight="bold",
                    color="#252525")
        ax.set_xticks(x, [item["short_name"] for item in items], fontsize=10)
        ax.set_xlim(-0.55, len(items) - 0.45)
        ax.set_ylim(0, y_limit)
        ax.set_title(title, fontsize=12, weight="bold", pad=14)
        ax.set_ylabel("Latency per action chunk (s)" if seconds_axis else
                      "Latency per action chunk (ms)", fontsize=10)
        if seconds_axis:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("SmolVLA inference latency by stage", fontsize=18, weight="bold", y=0.985)
    fig.text(0.5, 0.925,
             "RK3588 NPU prefill removes the remaining transformer bottleneck; vision now dominates.",
             ha="center", fontsize=11, color="#333333")
    fig.legend(handles=[Patch(facecolor=color, label=f"{index + 1}  {name}")
                        for index, (name, _, color) in enumerate(STAGES)],
               loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=4,
               frameon=False, fontsize=11)
    fig.text(0.5, 0.052,
             "Stages follow execution order. Panels use different labeled linear scales; values are warmed medians. "
             "Other is the residual needed to match the end-to-end median.",
             ha="center", fontsize=9.5, color="#555555")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.25)
    save_fig(fig, "smolvla_stage_latency_stacked")


def all_devices_stacked(profiles):
    fig, ax = plt.subplots(figsize=(16.5, 7.1), facecolor="white")
    x = np.arange(len(profiles), dtype=float)
    bottoms = np.zeros(len(profiles))
    for stage_index, (stage, _, color) in enumerate(STAGES):
        values = np.array([item["values"][stage_index] for item in profiles])
        ax.bar(x, values, width=0.66, bottom=bottoms, color=color,
               edgecolor="white", linewidth=0.8, label=f"{stage_index + 1}  {stage}")
        bottoms += values
    for index, item in enumerate(profiles):
        ax.text(index, max(item["total"] + 480, 1050), fmt_time(item["total"]),
                ha="center", va="bottom", fontsize=10.5, weight="bold")
    ax.set_xticks(x, [item["short_name"] for item in profiles], fontsize=10)
    ax.set_xlim(-0.62, len(profiles) - 0.38)
    ax.set_ylim(0, 44000)
    ax.set_ylabel("Latency per action chunk (s)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    ax.set_title("SmolVLA inference latency: all devices on one scale",
                 fontsize=17, weight="bold", pad=18)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=10)
    fig.text(0.5, 0.035,
             "Shared linear scale; subsecond paths are visually small. "
             "Colors follow execution order, and labels show end-to-end medians only.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.19)
    save_fig(fig, "smolvla_stage_latency_all_devices")


def donut_shares(profiles):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.4), facecolor="white")
    for ax, item in zip(axes.flat, profiles):
        values = item["values"]
        shares = np.array(values) / sum(values) * 100
        ax.pie(values, colors=[color for _, _, color in STAGES],
               startangle=90, counterclock=False, center=(-0.85, 0), radius=0.98,
               wedgeprops={"width": 0.33, "edgecolor": "white", "linewidth": 1.1})
        ax.text(-0.85, 0.08, fmt_time(item["total"]),
                ha="center", va="center", fontsize=14, weight="bold", color="#222222")
        ax.text(-0.85, -0.18, "total", ha="center", va="center",
                fontsize=9, color="#666666")
        for row, ((name, _, color), value, share) in enumerate(zip(STAGES, values, shares)):
            y = 0.75 - row * 0.50
            ax.scatter(0.40, y, color=color, marker="s", s=64, clip_on=False)
            ax.text(0.58, y, f"{name}: {fmt_time(value)}  ({fmt_share(share)})",
                    ha="left", va="center", fontsize=10, color="#252525")
        ax.set_title(item["full_name"], fontsize=11.5, weight="bold", pad=13)
        ax.set_xlim(-1.95, 2.70)
        ax.set_ylim(-1.20, 1.20)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("SmolVLA: share of inference time within each device", fontsize=18,
                 weight="bold", y=0.985)
    fig.text(0.5, 0.935,
             "Slices follow execution order clockwise: vision, prefill, denoising, other.",
             ha="center", fontsize=11, color="#333333")
    fig.text(0.5, 0.035,
             "Warmed stage medians; percentages use end-to-end median as denominator. "
             "With RK3588 vision, prefill and denoising on NPU, vision is now the largest stage.",
             ha="center", fontsize=9.5, color="#555555")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.095,
                        wspace=0.13, hspace=0.30)
    save_fig(fig, "smolvla_stage_share_donuts")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    profiles = load_profiles()
    stacked_bars(profiles)
    all_devices_stacked(profiles)
    donut_shares(profiles)
    print(OUT / "smolvla_stage_latency_stacked.png")
    print(OUT / "smolvla_stage_latency_all_devices.png")
    print(OUT / "smolvla_stage_share_donuts.png")


if __name__ == "__main__":
    main()
