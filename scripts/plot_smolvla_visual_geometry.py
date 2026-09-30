#!/usr/bin/env python3
"""Show the bounded geometry experiment, including rejected numerical protocols."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.session.read_text())
    if report["status"] != "complete":
        raise ValueError("Complete session required")
    records = [camera["plans"] for case in report["cases"].values() for camera in case["cameras"]]
    groups = [
        ("Whole consumer", ["spatial_1x1024", "spatial_16x64", "spatial_32x32", "spatial_64x16"], "All tested outputs match original bits"),
        ("Cut after normalization", ["split_mlp_original", "split_mlp_spatial_16x64", "split_mlp_spatial_32x32", "split_mlp_spatial_64x16"], "Split anchor matches; original bits FAIL"),
        ("Reshape inside one graph", ["stage_spatial_1x1024", "stage_spatial_16x64", "stage_spatial_32x32", "stage_spatial_64x16"], "Only 1 x 1024 matches original bits"),
    ]
    original = statistics.median(record["original_native"]["median_ms"] for record in records)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharey=True)
    data = {"scope": "One independent session; six actual images, 40 rotating trials per plan/image; bars summarize six per-image medians; whiskers are their min/max, not confidence intervals; last-layer consumer only",
            "original_native_summary_ms": original, "plans": {}}
    for ax, (title, plans, note) in zip(axes, groups):
        for index, name in enumerate(plans):
            values = [record[name]["median_ms"] for record in records]
            value = statistics.median(values)
            matches = all(record[name]["errors_vs_original_native"]["bitwise_equal"] for record in records)
            color = "#277DA1" if matches else "#D17A22"
            ax.bar(index, value, width=0.62, color=color, edgecolor="#333333", linewidth=0.5,
                   hatch="" if matches else "//", yerr=[[value - min(values)], [max(values) - value]], capsize=3)
            ax.text(index, value + 0.8, f"{value:.2f}", ha="center", fontsize=10)
            data["plans"][name] = {"per_image_medians_ms": values, "median_of_medians_ms": value,
                                   "range_ms": [min(values), max(values)], "all_tested_original_bits_equal": matches}
        ax.axhline(original, linestyle="--", color="#777777", linewidth=1)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_xticks(range(4), ["1 x 1024", "16 x 64", "32 x 32", "64 x 16"], fontsize=9)
        ax.set_ylim(0, 47)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, -0.19, note, transform=ax.transAxes, ha="center", fontsize=9)
    axes[0].set_ylabel("Complete consumer latency (ms)", fontsize=11)
    fig.suptitle("Same 1024 positions. Different compiler boundaries.", fontsize=16, y=0.98)
    fig.text(0.5, 0.035, "Dashed line: original consumer. Cut timing includes prefix + MLP calls. Hatched bars fail original-bit parity.\n"
             "Six per-image medians and min/max; not whole vision/policy acceleration or task quality.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.81, bottom=0.27, wspace=0.23)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output_dir / f"visual_geometry.{extension}", dpi=180)
    (args.output_dir / "chart_data.json").write_text(json.dumps(data, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
