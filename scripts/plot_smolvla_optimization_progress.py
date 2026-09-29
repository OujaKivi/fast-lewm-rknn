#!/usr/bin/env python3
"""Plot measured optimization steps without combining incompatible scopes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/smolvla_libero"
OUT = ROOT / "docs/figures/smolvla_optimization_progress"
BLUE, GREEN, ORANGE, GREY = "#0072B2", "#009E73", "#D55E00", "#88939C"
PURPLE, YELLOW = "#CC79A7", "#E0B441"


def read(relative):
    return json.loads((RESULTS / relative).read_text())


def style():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("PingFang SC", "Heiti SC", "STHeiti") if name in fonts), None)
    if chinese is None:
        raise RuntimeError("A Chinese font is required for the chart labels")
    plt.rcParams.update({"font.family": chinese, "font.size": 12,
                         "axes.titlesize": 14, "axes.labelsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "figure.facecolor": "white"})


def clean(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color="#E4E8EB", linewidth=0.7)
    ax.spines["left"].set_color("#BCC4CB")
    ax.spines["bottom"].set_color("#BCC4CB")


def bars(ax, labels, values, title, colors=None, notes=None, hatches=None):
    values = np.asarray(values)
    rects = ax.barh(np.arange(len(labels)), values, color=colors or BLUE,
                    height=0.58, edgecolor="white")
    for i, (rect, value) in enumerate(zip(rects, values)):
        if hatches and hatches[i]:
            rect.set_hatch(hatches[i])
            rect.set_edgecolor("#475159")
        text = f"{value:.2f}"
        if notes and notes[i]:
            text += f"  {notes[i]}"
        ax.text(value + max(values) * 0.018, i, text, va="center", fontsize=11)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * (1.58 if notes else 1.18))
    ax.set_xlabel("耗时中位数 / ms（越短越好）")
    ax.set_title(title, loc="left", pad=13)
    clean(ax, True)


def save(fig, name, subtitle, footer):
    fig.suptitle(subtitle, fontsize=18, fontweight="bold", x=0.04, ha="left", y=0.98)
    fig.text(0.04, 0.015, footer, fontsize=11, color="#475159", va="bottom")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)


def policy_plot(data):
    audit = read("policy_replay_probes/paired_audit.json")
    case = audit["sessions"]["strong2.json"]["cases"]["frame_0.npz"]
    plans = ["deployed_lite", "original_resident", "hoisted_consumer_ready"]
    names = ["原 Lite 接口\n视觉已并发", "原图常驻绑定\n减少接口开销", "C06 条件准备外提\n当前强策略基线"]
    stage_keys = ["vision_ms", "prefill_ms", "denoise_ms", "other_ms"]
    stage_names = ["双相机视觉", "Prefill", "十步 Action（含准备/Euler）", "其他"]
    means = np.asarray([[case["plans"][plan]["mean_stage_ms"][key] for key in stage_keys] for plan in plans])
    totals = np.asarray([case["plans"][plan]["mean_stage_ms"]["total_ms"] for plan in plans])
    np.testing.assert_allclose(means.sum(axis=1), totals, atol=1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1.08, 1]})
    fig.subplots_adjust(left=.07, right=.97, bottom=.25, top=.83, wspace=.36)
    bottom = np.zeros(3)
    for j, (name, color) in enumerate(zip(stage_names, (BLUE, PURPLE, GREEN, GREY))):
        axes[0].bar(np.arange(3), means[:, j], bottom=bottom, width=.63, label=name, color=color)
        if j != 3:
            for i in range(3):
                axes[0].text(i, bottom[i] + means[i, j] / 2, f"{means[i,j]:.1f}", ha="center", va="center", color="white", fontsize=12)
        bottom += means[:, j]
    for i, plan in enumerate(plans):
        median = case["plans"][plan]["median_total_ms"]
        axes[0].text(i, totals[i] + 45, f"均值 {totals[i]:.1f}\n中位 {median:.1f}", ha="center", fontsize=11)
    axes[0].set_xticks(np.arange(3), names)
    axes[0].set_ylim(0, 3050)
    axes[0].set_ylabel("完整策略耗时 / ms")
    axes[0].set_title("A  完整策略：堆叠分项均值", loc="left")
    axes[0].legend(loc="lower left", bbox_to_anchor=(.07, .067),
                   bbox_transform=fig.transFigure, frameon=False, fontsize=10, ncol=2)
    clean(axes[0])
    delta = means[:-1] - means[1:]
    y = np.arange(4)
    for i, (name, color) in enumerate(zip(("常驻绑定相对 Lite", "C06 相对常驻绑定"), (BLUE, GREEN))):
        position = y + (i - .5) * .32
        axes[1].barh(position, delta[i], height=.28, label=name, color=color)
        for k, value in enumerate(delta[i]):
            axes[1].text(max(value, 0) + 5, position[k], f"{value:+.2f}", va="center", fontsize=11)
    axes[1].set_yticks(y, ["视觉", "Prefill", "十步 Action", "其他"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-12, 505)
    axes[1].set_xlabel("比前一步省下的分项均值 / ms（正值为省时）")
    axes[1].set_title("B  每一步到底省在哪里？", loc="left")
    axes[1].legend(loc="lower left", bbox_to_anchor=(.60, .067),
                   bbox_transform=fig.transFigure, frameon=False, fontsize=11, ncol=2)
    clean(axes[1], True)
    save(fig, "01_policy_progress", "完整策略：收益主要来自 Action 路径，视觉仍约占 53%",
         "RK3588 · strong2 / episode 11 / frame 0 · 每方案 8 次；三帧与两会话全表见说明文档\n"
         "同一组调用的均值可相加；微小非目标阶段变化不算优化收益。C17 未集成，离线重放不是闭环。")
    data["policy"] = {"source": "policy_replay_probes/paired_audit.json", "chart_session": "strong2.json", "chart_frame": "frame_0.npz",
                      "plans": case["plans"], "step_stage_savings_mean_ms": delta.tolist(), "all_sessions": audit["sessions"]}


def vision_plot(data):
    one = read("vision_probes/vision12_paired.json")["plans"]
    dual = read("vision_probes/dual_view_paired.json")["plans"]
    patch = read("smolvla_vision_patch_pair.json")["plans"]
    batch = read("vision_probes/b2_full_unmasked_pair.json")["plans"]
    mixed = read("vision_probes/stage_pairs_mixed.json")["plans"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(left=.105, right=.96, top=.86, bottom=.14, wspace=.70, hspace=.65)
    patch_values = [patch[k]["median_ms"] for k in ("original", "variant")]
    bars(axes[0,0], ["原视觉图", "等价 Patch 改写"], patch_values, "A  早期 Patch 专项（单视角）", [GREY, BLUE],
         ["", f"省 {patch_values[0]-patch_values[1]:.1f} ms"], ["", "//"])
    keys = ["patched_fused_original", "patched_fused_unmasked", "native_pipeline_full_attention", "native_pipeline_parallel_h4"]
    values = [one[k]["median_ms"] for k in keys]
    bars(axes[0,1], ["Patch 已修复", "+ 移除恒零 Mask", "仅拆图（全头）", "+ 4+4+4 Head 并行"], values,
         "B  完整 12 层 + Connector（单视角）", [GREY, BLUE, ORANGE, GREEN],
         ["", "省 13.2", "比融合慢 34.8", "比融合省 243.9"], ["", "", "//", "//"])
    keys2 = ["fused_serial_mask7", "fused_parallel_mask1_mask2", "packed_h4_serial_views", "packed_h4_parallel_views"]
    values2 = [dual[k]["timings"]["resident_compute"]["median_ms"] for k in keys2]
    bars(axes[0,2], ["两相机串行融合", "整相机并发（强对照）", "Head 并行 / 相机串行", "Head 与相机都并发"], values2,
         "C  双视角：强基线吃掉大部分收益", [GREY, BLUE, GREEN, ORANGE],
         ["", "省 428.6", "再省 47.2", "未再加速"], ["", "", "//", "//"])
    batch_values = [batch[k]["median_ms"] for k in ("original", "variant")]
    bars(axes[1,0], ["原 B2 图", "B2 恒零 Mask 修复"], batch_values,
         "D  B2 内存修复：不是大速度收益", [GREY, BLUE], ["", "仅省 16.6 ms"])
    stage_keys = ["patch_embedding", "qkv_producers", "attention", "projection_mlp_consumers", "postnorm_connector"]
    stage_labels = ["Patch", "Norm + QKV", "Attention + 等待", "输出投影 + MLP", "Connector"]
    stage = np.asarray([[one[k]["supplemental_stage_medians_ms"][s] for s in stage_keys]
                        for k in ("native_pipeline_full_attention", "native_pipeline_parallel_h4")])
    ax = axes[1,1]
    for i, (name, color) in enumerate(zip(("全头分图", "Head 并行分图"), (BLUE, GREEN))):
        positions = np.arange(5) + (i-.5)*.33
        ax.barh(positions, stage[i], height=.29, label=name, color=color)
        for j, value in enumerate(stage[i]):
            ax.text(value+7, positions[j], f"{value:.1f}", va="center", fontsize=10)
    ax.set_yticks(np.arange(5), stage_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 625)
    ax.set_xlabel("各阶段诊断中位数 / ms（不能相加成总中位数）", fontsize=10)
    ax.set_title("E  Head 并行省在 Attention：约 280 ms", loc="left", pad=13)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    clean(ax, True)
    mixed_keys = ["mixed_attention_consumer_serial_mask7", "mixed_attention_consumer_parallel_mask7",
                  "mixed_attention_consumer_parallel_mask1", "mixed_attention_consumer_parallel_mask3"]
    mixed_values = [mixed[k]["median_ms"] for k in mixed_keys]
    bars(axes[1,2], ["Attention + Consumer 串行", "交错（Consumer 三核）", "交错（Consumer 单核）", "交错（Consumer 两核）"],
         mixed_values, "F  尝试阶段重叠：没有扎实净收益", [GREY, BLUE, ORANGE, ORANGE], ["", "约持平", "反而更慢", "反而更慢"])
    save(fig, "02_vision_progress", "视觉优化历程：单视角约 31% 的优势，双视角强对照下只剩约 4%",
         "A/D 为各自接口专项；B/C 为驻留计算；E 为额外三轮分项诊断；F 为最后一层局部阶段实验。各面板独立，不能串加。\n"
         "斜线柱与原融合 FP16 输出非逐位一致（未降低精度配置）；任务质量门槛仍未完成，未计入上图完整策略。")
    data["vision"] = {"single_total_median_ms": dict(zip(keys, values)), "dual_resident_total_median_ms": dict(zip(keys2, values2)),
                      "single_stage_diagnostic_median_ms": dict(zip(stage_labels, (stage[0]-stage[1]).tolist())),
                      "patch_medians_ms": patch_values, "batch2_medians_ms": batch_values,
                      "mixed_medians_ms": dict(zip(mixed_keys, mixed_values))}


def cross_plot(data):
    record = read("cross_followup_probes/session13.json")
    plans = record["plans"]["10"]
    actual = read("policy_replay_probes/cross_replay_session1.json")
    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .95], hspace=.58, wspace=.55)
    fig.subplots_adjust(left=.17, right=.97, top=.85, bottom=.14)
    ax = fig.add_subplot(grid[0,0])
    keys = ["original_recompute", "ready_joint", "split_parallel_heads", "compact_split_parallel",
            "compact_parallel_cpu_restore", "compact_parallel_cpu_both"]
    labels = ["原块：重复条件准备", "条件准备外提", "+ Head 并行（强对照）", "Compact + 编译器恢复", "Compact + CPU 输出位恢复", "C17：CPU 双向位桥接"]
    values = [plans[k]["median_ms"][-1] for k in keys]
    bars(ax, labels, values, "A  同轮完整 Cross 边界 / 十次消费", [GREY, BLUE, BLUE, ORANGE, GREEN, GREEN],
         ["", "", "", "比强对照慢", "比强对照省 0.60", "比强对照省 1.18"])
    ax = fig.add_subplot(grid[0,1])
    names = list(actual["cases"])
    actual_values = np.asarray([[actual["cases"][name]["plans"][k]["median_ms"][-1]
                                for name in names] for k in ("split_parallel_heads", "compact_parallel_cpu_both")])
    for i, (label, color) in enumerate(zip(("Head 强对照", "C17 双向位桥接"), (BLUE, GREEN))):
        pos = np.arange(3) + (i-.5)*.34
        ax.bar(pos, actual_values[i], width=.3, color=color, label=label)
        for j, value in enumerate(actual_values[i]):
            ax.text(pos[j], value+.12, f"{value:.2f}", ha="center", fontsize=11)
    ax.set_xticks(np.arange(3), ["Frame 0", "Frame 42", "Frame 83"])
    ax.set_ylim(0, 14)
    ax.set_ylabel("完整 Cross 边界中位数 / ms")
    ax.set_title("B  观测驱动重放：仍省约 1.2 ms", loc="left", pad=13)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11,
              bbox_to_anchor=(.97, .925), bbox_transform=fig.transFigure)
    clean(ax)
    ax = fig.add_subplot(grid[1,:])
    split_keys = keys[2:]
    # Keep stage medians grouped, not stacked: their sum is not the total median.
    columns = [0, 3, 4, 5, 2]
    stage = np.asarray([[plans[k]["median_ms"][j] for j in columns] for k in split_keys])
    stage_labels = ["一次条件准备", "Q / RoPE Producer", "Attention + 并行等待", "输出投影 / Consumer", "CPU 位复制 + 同步"]
    legend = ["Head 强对照", "Compact / 编译器恢复", "CPU 仅恢复输出", "CPU 双向位桥接"]
    for i, (name, color) in enumerate(zip(legend, (BLUE, ORANGE, YELLOW, GREEN))):
        pos = np.arange(5) + (i-1.5)*.19
        ax.bar(pos, stage[i], width=.17, color=color, label=name)
        for j, value in enumerate(stage[i]):
            ax.text(pos[j], value+.07, f"{value:.2f}", ha="center", fontsize=10)
    ax.set_xticks(np.arange(5), stage_labels)
    ax.set_ylim(0, 6.1)
    ax.set_ylabel("十次消费的分项中位数 / ms")
    ax.set_title("C  紧凑 Attention 更快，但编译器布局恢复吃掉收益；CPU 位复制绕开它", loc="left", pad=13)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.00), fontsize=11)
    clean(ax)
    save(fig, "03_c17_progress", "C17：多付小规模 CPU 位复制，少做更昂贵的 NPU 布局恢复",
         "A/C：S13，各方案 40 次，合成 checkpoint 输入；B：每帧 60 次，hidden 经 CPU FP32 重建，C17 输出未反馈进 Euler。\n"
         "Q/RoPE + Cross Attention + 输出投影/残差，不含 MLP。准备/提交/同步/最终读回已计；公共原始 Prefix 打包等额排除。")
    data["c17"] = {"source_session": "session13.json", "ten_consumption_total_median_ms": dict(zip(keys, values)),
                   "stage_names": stage_labels, "stage_medians_ms": dict(zip(split_keys, stage.tolist())),
                   "actual_frame_medians_ms": {name: actual_values[:, i].tolist() for i,name in enumerate(names)},
                   "all_s13_variants": {k: v["median_ms"][-1] for k,v in plans.items()}}


def write_report(data):
    lines = ["# SmolVLA 优化进展图与 C17 解释", "", "2026-09-29；只重画已有测量，没有新增性能实验。", "",
             "## C17 是什么", "",
             "C17 优化的是 action expert 的一个 cross-Attention 边界，不是视觉编码，也不是完整 transformer block。",
             "当前几何是 15 个 Q heads、5 个 KV heads、50 个查询、149 个条件 token；包含 Q/RoPE、Attention、输出投影与残差，不含 MLP。", "",
             "原来为了便于执行，会把每组共享的 K/V 展开成三份，供三个 Q heads 消费。紧凑方案保留 5 组 K/V，将共享同一 K/V 的三个 head 的查询打包，数学工作不变。",
             "真正的障碍是：这个紧凑 Attention 的输入/输出排列与原 Q/RoPE、输出投影的原生存储不匹配；让编译器恢复会出现额外 reshape/恢复 Conv，让矩阵权重顺着改又改变浮点累加顺序。", "",
             "最终路径：原 Q/RoPE → CPU 原生位打包 → NPU 紧凑 GQA 并行（2+2+1 KV groups）→ CPU 原生位恢复 → 原输出投影/残差。",
             "CPU 只搬 FP16 存储位，不重新计算数值、不降低精度；每方向复制 96 KB，双向复制和同步十次约 0.397 ms（S13）。条件缓存 583680 → 194560 bytes/层，不是峰值 RSS 或实测 DDR 流量。", "",
             "已测单块的 512 个合成扰动及 30 个观测驱动步骤逐位通过，但不是全输入数学证明。局部比 head 强对照快约 10%，每观测/该块十次省约 1.2 ms；还没集成完整策略，更不能称为 VLA 整体快 10%。", "",
             "## 图与统计口径", "",
             "所谓‘强极限’应改称‘当前已测强基线’，不是理论性能上限。没有充分证据证明 NPU 已达到 Roofline。",
             "完整策略堆叠的是同会话、同帧、同一组调用的分项均值，均值之和核验等于总均值；总中位数另标。",
             "局部阶段使用分组柱画各自中位数，不能相加成总中位数。不混池不同会话、输入、I/O 口径，也不将尚未集成的方案串加。", "",
             "![完整策略](figures/smolvla_optimization_progress/01_policy_progress.png)", "",
             "### 完整策略代表帧的每步分项变化", "",
             "strong2 / frame 0 / 每方案 8 次，单位 ms；正值是省时。Action 包含一次条件准备及十步依赖 FP32 Euler。", "",
             "| 步骤 | 视觉 | Prefill | Action | 其他 | 总均值省时 |", "|---|---:|---:|---:|---:|---:|"]
    for label, delta in zip(("常驻绑定 vs Lite", "C06 vs 常驻绑定"), data["policy"]["step_stage_savings_mean_ms"]):
        lines.append(f"| {label} | " + " | ".join(f"{v:+.2f}" for v in delta) + f" | {sum(delta):+.2f} |")
    lines += ["", "视觉/Prefill 约 0-2 ms 的非目标变化属于本轮观测波动，不归因于优化机制。", "",
              "### 两次会话、全部三帧的完整策略总中位数", "",
              "| 会话 / 帧 | Lite | 常驻绑定 | C06 | 常驻省时 vs Lite | C06 省时 vs 常驻 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for session, value in data["policy"]["all_sessions"].items():
        for frame, case in value["cases"].items():
            values = [case["plans"][k]["median_total_ms"] for k in ("deployed_lite", "original_resident", "hoisted_consumer_ready")]
            lines.append(f"| {session} / {frame} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {values[0]-values[1]:.2f} | {values[1]-values[2]:.2f} |")
    lines += ["", "以上是总中位数之差，不等于逐轮配对差值的中位数。原 Lite 列已经共享优化后的整相机并发视觉，不是最早 3.236 s 闭环部署。", "",
              "![视觉优化](figures/smolvla_optimization_progress/02_vision_progress.png)", "",
              "### 视觉各部分省时", "",
              "额外三轮分项诊断，完整单视角 12 层；全头分图减去 head 并行分图的中位数，不能并入正式总时延中位数。", "",
              "| 部分 | 省时 / ms |", "|---|---:|"]
    for name, value in data["vision"]["single_stage_diagnostic_median_ms"].items():
        lines.append(f"| {name} | {value:+.3f} |")
    lines += ["", "单视角 head 方案相对融合强基线 775.49 → 531.62 ms，省 243.88 ms；双视角最强融合并发 1103.42 → head 方案 1056.27 ms，仅再省 47.15 ms（4.27%）。",
              "这两组不是同一实验，不能相乘。图中 Patch、head 分图相对各自原融合 FP16 均有舍入差异；没有减精度配置，但完整任务质量还没过门槛。",
              "B2 mask 修复大幅减少分配字段，却只省 16.62 ms。跨阶段 overlap 的几个尝试持平或更慢；图中保留失败结果。", "",
              "![C17 优化](figures/smolvla_optimization_progress/03_c17_progress.png)", "",
              "### C17 每个部分的变化", "",
              "S13，同轮十次消费的分项中位数；表内省时为两分项中位数之差。CPU 位复制包含同步，输入 hidden 打包与最终输出读回在总时延中另计。", "",
              "| 部分 | Head 强对照 | Compact / 编译恢复 | CPU 单向恢复 | CPU 双向桥接 | 双向 vs Head 省时 |",
              "|---|---:|---:|---:|---:|---:|"]
    stage = list(data["c17"]["stage_medians_ms"].values())
    for j, name in enumerate(data["c17"]["stage_names"]):
        values = [s[j] for s in stage]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in values) + f" | {values[0]-values[-1]:+.3f} |")
    lines += ["", "最终图中保留全部 head 和条件 token。Compact 本身让 Attention 快约 1.48 ms，却会让上下游更贵；双向 CPU 位桥接保住原 Q/O 算术路径，绕开恢复矩阵。它不是 CPU softmax，也不是已实现的 CPU/NPU overlap。", "",
              "S13 总中位数：head 11.463 → 单向位恢复 10.865 → 双向位桥接 10.281 ms。三轮 S13-S15 比 head 低 10.3%-10.7%；旧传播布局/权重重索引方案未通过扩大数值门槛，不计入合法最优。",
              "真实条件重放中每块只省约 1.2 ms，约为 2.08 s 完整策略的 0.058%。这只是重要性对照，不是已集成策略收益，不能机械外推全部层。尾延迟并非始终改善。", "",
              "### S13 全部 19 个消融（没有只挑赢家）", "",
              "| 实现标识 | 十次消费总中位数 / ms |", "|---|---:|"]
    for name, value in data["c17"]["all_s13_variants"].items():
        lines.append(f"| `{name}` | {value:.3f} |")
    lines += ["", "此表是消融，不是全部方案具有同一数值资格或穷尽全部编译配置。", "", "## 证据与边界", "",
              "- [完整策略验证](SMOLVLA_POLICY_REPLAY_VALIDATION.md)；只含离线模型内推理，不含外部预处理、网络、仿真、动作后处理或图初始化。",
              "- [视觉专项](SMOLVLA_VISION_INVESTIGATION.md)；完整融合相机并发已在实际帧逐位通过，packed/head 分图未作为完整策略质量已通过的实现。",
              "- [C17 续验](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md)；真实条件重放的 hidden 是 CPU FP32 重建，不是捕获的 NPU 内部 hidden。",
              "- [绘图数据](figures/smolvla_optimization_progress/chart_data.json)；图与表直接由原始 JSON 生成，SVG/PDF 同目录。",
              "- 不把旧闭环、合成输入、不同版本模型的数字拼成一条 3.236 → 2.08 s 加速链。Prefill 暂未在这一轮建立独立优化收益，端云也未取得实测胜区。", ""]
    (ROOT / "docs/SMOLVLA_OPTIMIZATION_PROGRESS.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    data = {"scope": "Existing measurements only; no new timing, no cross-scope cumulative speedup", "unit": "ms"}
    policy_plot(data)
    vision_plot(data)
    cross_plot(data)
    (OUT / "chart_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    write_report(data)
    print(f"Saved three PNG/SVG/PDF figures and evidence report to {OUT}")


if __name__ == "__main__":
    main()
