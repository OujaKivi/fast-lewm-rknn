#!/usr/bin/env python3
"""Plot the matched-noise SmolVLA LIBERO episode measurements."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "smolvla_libero"
OUTPUT = ROOT / "figures" / "smolvla_libero_matched_breakdown"

RUNS = (
    ("RTX 5060 CUDA", "matched_cuda_full.json"),
    ("Mac M5 Pro MPS", "matched_mac_mps_full.json"),
    ("RK3588 NPU + CPU", "matched_rk_npu_full.json"),
    ("RTX 主机 CPU", "matched_rtx_cpu_full.json"),
)
STAGES = (
    ("视觉编码", "vision_s_total", "#0072B2"),
    ("Prefill", "prefill_s_total", "#009E73"),
    ("去噪", "denoise_s_total", "#D55E00"),
)


def main():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    records = [(label, json.loads((RESULTS / filename).read_text())) for label, filename in RUNS]
    rk = records[2][1]
    assert all(record["success"] and record["noise_mode"] == "matched_cpu_stream"
               for _, record in records)
    assert len({record["first_noise_sha256"] for _, record in records}) == 1

    fig = plt.figure(figsize=(14.2, 5.7), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.26, 1), wspace=0.27,
                           left=0.12, right=0.97, top=0.79, bottom=0.22)
    ax = fig.add_subplot(grid[0])
    values = [record["inference_s_total"] / record["steps"] for _, record in records]
    y = np.arange(len(records))
    bars = ax.barh(y, values, height=0.56,
                   color=("#28688F", "#2B8C75", "#CE7444", "#747474"))
    ax.set_yticks(y, [label for label, _ in records], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 6.1)
    ax.set_xlabel("每次动作推理耗时（秒）", fontsize=11)
    ax.set_title("同一任务、同一噪声序列", loc="left", fontsize=13, weight="bold", pad=13)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, (_, record), value in zip(bars, records, values, strict=True):
        ax.text(value + 0.10, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f} s  ·  {record['steps']} 步成功",
                va="center", ha="left", fontsize=10.5, color="#20282c")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax2 = fig.add_subplot(grid[1])
    stage_values = [(name, rk[key] / rk["steps"], color) for name, key, color in STAGES]
    other = (rk["inference_s_total"] - sum(rk[key] for _, key, _ in STAGES)) / rk["steps"]
    stage_values.append(("其他/传输", other, "#747474"))
    left = 0.0
    for name, value, color in stage_values:
        ax2.barh([0], [value], left=left, height=0.46, color=color,
                 edgecolor="white", linewidth=1.2)
        if value > 0.45:
            ax2.text(left + value / 2, 0, f"{value / sum(v for _, v, _ in stage_values):.0%}",
                     color="white", weight="bold", ha="center", va="center", fontsize=12)
        left += value
    ax2.set_xlim(0, left)
    ax2.set_ylim(-1.19, 0.58)
    ax2.set_yticks([])
    ax2.set_xlabel("每次动作推理耗时（秒）", fontsize=11)
    ax2.set_title(f"RK3588 NPU 主计算分项：{left:.3f} s/步",
                  loc="left", fontsize=13, weight="bold", pad=13)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax2.set_axisbelow(True)
    for index, (name, value, color) in enumerate(stage_values):
        row_y = -0.43 - index * 0.22
        ax2.scatter([0.12], [row_y], s=65, color=color)
        ax2.text(0.28, row_y, f"{name}  {value:.3f} s/步（{value / left:.0%}）",
                 va="center", fontsize=10.5, color="#20282c")

    fig.suptitle("SmolVLA · LIBERO 闭环推理", x=0.12, ha="left",
                 y=0.96, fontsize=20, weight="bold", color="#17232b")
    fig.text(0.12, 0.87, "4 种部署均完成同一任务；RK3588 的双相机视觉编码占主要时间。",
             fontsize=11.5, color="#41515b")
    fig.text(0.12, 0.075,
             "单任务单次结果。推理耗时不含模拟器步进与预处理；远端部署含通信。右侧“其他”包含 CPU 辅助、数据搬运与调度。",
             fontsize=9.5, color="#55636b")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
