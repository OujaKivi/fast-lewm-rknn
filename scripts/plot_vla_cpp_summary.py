#!/usr/bin/env python3
"""Visualize the recorded vla.cpp pilot, including its numerical limitation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next(name for name in ("PingFang SC", "Heiti SC", "STHeiti") if name in fonts),
                         "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    results = ROOT / "results"
    pilot = json.loads((results / "smolvla_vla_cpp_rk3588_pilot.json").read_text())
    cpu = json.loads((results / "smolvla_profile_rk3588_cpu.json").read_text())["timing"]
    hybrid = json.loads((results / "smolvla_profile_rk3588_npu_vision_denoise.json").read_text())["timing"]
    keys = ["vision_ms", "prefill_ms", "denoise_ms", "total_ms"]
    matched = pilot["matched_fixture_full_path"]
    values = np.asarray([[cpu[k]["median_ms"] for k in keys],
                         [hybrid[k]["median_ms"] for k in keys],
                         [matched["vision_ms"], matched["prefill_ms"], matched["denoising_ms"], matched["total_ms"]]]) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=.07, right=.97, top=.81, bottom=.25, wspace=.35)
    colors = ["#0072B2", "#009E73", "#D55E00"]
    for i, (label, color) in enumerate(zip(("PyTorch CPU", "RKNN 视觉/去噪 + CPU Prefill", "vla.cpp BF16 CPU"), colors)):
        x = np.arange(4) + (i-1)*.24
        axes[0].bar(x, values[i], width=.21, color=color, label=label)
        for j, value in enumerate(values[i]):
            axes[0].text(x[j], value+1.1, f"{value:.2f}", ha="center", fontsize=10)
    axes[0].set_xticks(np.arange(4), ["视觉", "Prefill", "十步去噪", "完整推理"])
    axes[0].set_ylabel("耗时 / s（越低越好）")
    axes[0].set_ylim(0, 85)
    axes[0].set_title("分项优势不等于完整推理优势", loc="left", fontsize=14, pad=12)
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    precomputed = pilot["matched_fixture_with_pytorch_vision_embedding"]
    errors = [matched["action_mae_vs_i5_cpu"], precomputed["action_mae_vs_i5_cpu"]]
    axes[1].bar(np.arange(2), errors, color=[colors[2], colors[1]], width=.58)
    for i, value in enumerate(errors):
        axes[1].text(i, value+.006, f"{value:.5f}", ha="center")
    axes[1].set_xticks(np.arange(2), ["vla.cpp 完整路径", "使用 PyTorch\n视觉特征的诊断"])
    axes[1].set_ylabel("动作 MAE（相对 i5 PyTorch CPU）")
    axes[1].set_ylim(0, .25)
    axes[1].set_title("视觉替换显著缩小输出差异", loc="left", fontsize=14, pad=12)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E4E8EB", linewidth=.7)
    fig.suptitle("vla.cpp 部署对照：该版本不是 RK3588 加速替代方案", fontsize=18, x=.05, ha="left", y=.97)
    fig.text(.05, .12, "同一 smolvla_base 合成 fixture、单相机、十步去噪；PyTorch/RKNN 为 2/5 次暖机后中位数，vla.cpp 为一次匹配输入调用。", fontsize=10)
    fig.text(.05, .075, "右图是视觉路径差异诊断，不是任务成功率；特征旁路的计时不包含特征生成，不能当作完整推理加速。", fontsize=10)
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / f"vla_cpp_pilot_summary.{extension}", dpi=170, facecolor="white")
    plt.close(fig)
    print(out / "vla_cpp_pilot_summary.png")


if __name__ == "__main__":
    main()
