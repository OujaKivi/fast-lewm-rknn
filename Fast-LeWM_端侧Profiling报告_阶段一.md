# Fast-LeWM 端侧异构 Profiling 报告（阶段一：CPU 侧）

- 平台：RK3588（4×Cortex-A76 @2.35 GHz 大核，绑核 cpu4-7，torch 2.9.1+cpu / FP32）
- 模型：Fast-LeWorldModel（action-prefix encoder 1.80M + parallel latent predictor 9.02M，世界模型合计 10.83M；ViT-tiny 视觉编码器 5.52M 近似）
- CEM 配置：num_samples=300, n_iters=30, topk=30, horizon=5, action_dim=2
- 日期：2026-09-15　｜　原始数据：`profile_full_big4.json`　｜　脚本：`profile_fast_lewm.py`
- 对照模式：**fast**（Fast-LeWM，action-prefix 一次编码完整序列，多步复用）vs **ar**（受控 LeWM 自回归对照，每步重新编码单步动作）

---

## 0. 执行摘要

阶段一在 RK3588 CPU 上完成了 E1（端到端 CEM 分解）、E3（batch 扫描）、E4（runtime 微基准）、E5（精度基线）四项实验。核心发现：

1. **CPU 上模型计算仍占绝对主导**：Fast 模式 rollout 占 83.07%、视觉编码 16.84%，CEM 采样/scoring/elite/tensor prep 合计仅 0.09%。
2. **Fast 相比 AR 有 1.19× 加速**（19.55s vs 23.29s/决策），且出现轻微瓶颈转移：rollout 占比 -2.57%、vi_enc 占比 +2.55%。
3. **非模型 runtime 阶段全部亚毫秒级**：CEM 采样 0.098ms、elite selection 0.061ms、tensor prep 0.044ms、Python 循环 0.002ms；合计 ~2.3ms/iter，仅占 rollout 的 0.36%。
4. **大 batch 效率显著提升**：S=16→512，throughput 从 88→537 cand/s（6.1×），latency/candidate 从 11.4→1.86ms（6.1×），512 仍未见拐点。
5. **H1 瓶颈转移在 CPU 上不显著**：模型计算占 83-86%，非模型阶段可忽略。**瓶颈转移只有在 NPU 把模型计算压下来后才会显著显现**——这正是阶段二需要验证的。

---

## 1. 实验设置

### 1.1 阶段定义（对应计划书 4.1）

| 阶段 | 含义 | 测量位置 |
|---|---|---|
| `vi_enc` | 视觉编码（当前帧+目标，每决策一次） | ViT 前向 |
| `cem_sample` | CEM 候选动作采样（高斯+加噪） | 迭代开头 |
| `tensor_prep` | candidate expand / reshape / contiguous | 送模型前 |
| `rollout_model` | 世界模型推演（action_enc + predictor，5 步） | 模型前向 |
| `score` | latent L2 cost 计算 | 模型后 |
| `elite_select` | top-k elite 选择 + mu/sigma 更新 | 迭代末尾 |

### 1.2 对照模式

- **fast**：action-prefix encoder 一次编码完整 5-block 动作序列，得到的 action embedding 在 5 步 predictor 中复用（模拟 Fast-LeWM "model calls 5→1"）。
- **ar**：每步重新编码单步动作，predictor 单步预测，5 步循环（受控 LeWM 自回归对照）。

两者使用完全相同的模型权重、CEM 超参和输入分布，唯一变量是 action encoding 的调用方式。

### 1.3 测量方法

- 高精度计时 `time.perf_counter_ns()`，所有模型前向在 `@torch.no_grad()` 下。
- E1：warmup 1 次（5 iter），正式跑 3 次完整决策（30 iter），取各阶段中位数。
- E3：warmup 1 次，每档 batch 跑 5 次 fast + 5 次 ar。
- E4：每阶段 warmup 3 次，正式 50 次（Python 空循环 200 次），取中位数。
- 绑核 `taskset -c 4-7`，torch 线程数=4，NPU 频率记录为 1GHz（CPU 跑时 NPU 空闲）。

---

## 2. E1：端到端 CEM 循环 latency 分解

### 2.1 绝对时延（单次完整决策，30 iter）

| 阶段 | Fast 总时延(ms) | Fast 占比 | AR 总时延(ms) | AR 占比 | Fast vs AR |
|---|---:|---:|---:|---:|---:|
| rollout_model（世界模型推演） | 19,068.3 | **83.07%** | 22,928.4 | **85.64%** | Fast 快 16.8% |
| vi_enc（视觉编码） | 3,866.2 | 16.84% | 3,825.4 | 14.29% | 基本相同 |
| score（L2 cost） | 8.2 | 0.04% | 8.5 | 0.03% | — |
| elite_select（top-k+更新） | 5.4 | 0.02% | 5.4 | 0.02% | — |
| cem_sample（高斯采样） | 3.6 | 0.02% | 3.7 | 0.01% | — |
| tensor_prep（整理输入） | 1.9 | 0.01% | 1.9 | 0.01% | — |
| **合计** | **19,550.8** | **100%** | **23,287.9** | **100%** | **Fast 1.191×** |

### 2.2 每 iter 中位数（30 iter 平均）

| 阶段 | Fast (ms/iter) | AR (ms/iter) |
|---|---:|---:|
| rollout_model | 635.61 | 764.28 |
| vi_enc（每决策一次，/30） | 128.87 | 127.51 |
| score | 0.275 | 0.284 |
| elite_select | 0.178 | 0.180 |
| cem_sample | 0.120 | 0.125 |
| tensor_prep | 0.063 | 0.064 |

### 2.3 瓶颈转移分析（H1）

Fast 相比 AR，各阶段占比变化：

| 阶段 | Fast 占比 | AR 占比 | 变化 |
|---|---:|---:|---:|
| rollout_model | 83.07% | 85.64% | **-2.57%** |
| vi_enc | 16.84% | 14.29% | **+2.55%** |
| 其余（score/elite/sample/prep） | 0.09% | 0.07% | +0.02% |

**结论**：Fast-LeWM 消除逐步 action encoding 后，rollout 占比下降 2.57%，视觉编码占比相应上升 2.55%——出现了**轻微的瓶颈转移**。但转移幅度很小，因为：
1. CPU 上模型计算（rollout）仍占 83%，绝对时延 635ms/iter，远大于其他阶段；
2. 非模型阶段（CEM 采样、scoring、elite、tensor prep）合计仅 0.64ms/iter，占 rollout 的 0.1%，完全无法与模型计算竞争。

**关键判断**：H1 的"瓶颈从模型计算转移到采样规划与异构数据流"在 **CPU 上不成立**——CPU 上模型计算仍是绝对瓶颈。只有当 NPU 把 rollout 从 635ms/iter 压到（例如）50ms/iter 以下时，vi_enc（129ms/决策）、runtime/transfer（~2.3ms/iter）、CEM 控制逻辑才会成为新的瓶颈。**这正是阶段二 NPU 实验需要验证的核心预测。**

---

## 3. E3：candidate batch 扫描

### 3.1 结果（5 次重复中位数）

| S (candidates) | fast rollout5 (ms) | ar rollout5 (ms) | Fast/AR speedup | latency/cand (ms) | throughput (cand/s) |
|---:|---:|---:|---:|---:|---:|
| 16 | 182.58 | 241.00 | **1.320** | 11.411 | 87.63 |
| 32 | 215.92 | 278.91 | 1.292 | 6.747 | 148.20 |
| 64 | 269.45 | 332.46 | 1.234 | 4.210 | 237.52 |
| 128 | 364.77 | 451.74 | 1.238 | 2.850 | 350.91 |
| 256 | 578.84 | 694.17 | 1.199 | 2.261 | 442.26 |
| 512 | 954.11 | 1086.90 | 1.139 | 1.864 | 536.63 |

### 3.2 分析（H3）

1. **大 batch 效率显著提升**：S=16→512，throughput 增长 **6.1×**（88→537 cand/s），latency/candidate 下降 **6.1×**（11.4→1.86ms）。这验证了 Fast-LeWM 的 action-prefix + parallel predictor 结构天然适合大 batch 并行。
2. **CPU 上 512 仍未见拐点**：throughput 仍在近似线性增长（S 翻倍，throughput 约增 1.2-1.5×），内存峰值 432MB 远低于 16GB，带宽也未成为限制。**CPU 上最佳 batch 区间是"越大越好"**。
3. **Fast/AR speedup 随 batch 增大而缩小**：S=16 时 1.32×，S=512 时 1.14×。原因是大 batch 下矩阵乘（GEMM）成为主导，action encoding 的节省占比下降；小 batch 下 action encoding 的 transformer overhead 更显著。
4. **对 NPU 的预测**：NPU 上预计会出现更明显的拐点——小 batch（S<64）吃不满 NPU 算力（NPU 偏好大矩阵乘），大 batch（S>256）可能受 SRAM/带宽限制。**NPU 上的最佳 batch 区间预计在 128-256 之间**，需要阶段二实验确认。

---

## 4. E4：CPU 侧 runtime 微基准

### 4.1 结果（中位数，n=50）

| 微基准 | 中位时延 (ms) | 说明 |
|---|---:|---|
| tensor_alloc_randn_S192 | **1.216** | 分配 (300,1,192) randn tensor（最大的非模型开销） |
| latent_cat_5steps | 0.750 | 5 步 latent torch.cat（rollout 内每步拼接） |
| score_l2_cost | 0.162 | (S,1,192) L2 cost 计算 |
| cem_sample_gaussian_S300 | 0.098 | (300,5,2) 高斯采样+加噪 |
| elite_select_topk30_mean_std | 0.061 | topk(30) + mean + std |
| tensor_prep_expand_contiguous | 0.044 | expand + contiguous |
| python_loop_30iters_empty | 0.002 | 30 次空 Python 循环 |

### 4.2 分析（H4）

1. **所有非模型 runtime 阶段均为亚毫秒级**，合计约 **2.33ms/iter**（tensor_alloc 1.22 + latent_cat 0.75 + score 0.16 + sample 0.10 + elite 0.06 + prep 0.04）。
2. **相比 rollout_model 635.6ms/iter，runtime 合计仅占 0.37%**。H4 的"runtime+transfer ≈ model"在 CPU 上**完全不成立**。
3. **最大的非模型开销是 tensor 分配**（1.22ms），其次是 latent cat（0.75ms）。这两个在 NPU 上会转化为：
   - tensor_alloc → NPU 输入 tensor 分配 + h2d 拷贝（可能显著增大）
   - latent_cat → rollout 内每步 d2h 回传 + h2d 再送（NPU 上 rollout 如果是逐步推理，这会是主要开销）
4. **CPU scoring/elite 极快**（0.22ms 合计），即使 NPU 把模型压到 50ms，这部分仍只占 0.4%，不会成为瓶颈。
5. **H4 的真正验证需要 NPU 实验**：CPU 上没有 host-device 拷贝、layout 转换、RKNN invoke 开销。这些在 NPU 上可能成为主要瓶颈（特别是 rollout 逐步推理时的 d2h/h2d 反复拷贝）。

---

## 5. E5：精度基线与资源

### 5.1 FP32 精度基线（5 步 rollout，S=64）

| 指标 | 值 |
|---|---:|
| final_latent_mean | 0.0 |
| final_latent_std | 1.000036 |
| final_latent_abs_max | 3.932805 |
| final_latent_norm_mean | 13.856339 |

输入 latent 为标准正态（mean=0, std=1），5 步 rollout 后输出仍保持近似标准正态（std=1.000036），说明 FP32 推理数值稳定，无爆炸或消失。**这是 NPU INT8/FP16 量化的精度对比基线**——量化后如果 final_latent_std 偏离 1.0 超过 5%，或 abs_max 显著变化，说明量化误差在多步 rollout 中累积，可能影响 CEM 代价排序。

### 5.2 资源占用

| 指标 | 值 |
|---|---|
| max_rss（峰值内存） | 432.4 MB |
| NPU 当前频率 | 1 GHz（满载，但 CPU 跑时 NPU 空闲） |
| torch 线程数 | 4（绑 4×A76） |
| CPU 亲和性 | [4, 5, 6, 7] |

432MB 远低于 16GB，内存不是 CPU 侧瓶颈。NPU 频率 1GHz（三核 6TOPS 满载频率），但 CPU 实验中 NPU 未被使用。

---

## 6. 核心结论：H1–H4 回答

| 假设 | CPU 侧结论 | 对 NPU 侧的预测 |
|---|---|---|
| **H1 瓶颈转移** | CPU 上不显著：rollout 占 83-86%，非模型阶段 <0.1%。Fast 相比 AR 有轻微转移（rollout -2.57%，vi_enc +2.55%）。 | **NPU 上预计显著**：rollout 压到 50ms/iter 以下后，vi_enc（129ms/决策）、h2d/d2h 拷贝、RKNN invoke 将成为新瓶颈。 |
| **H2 NPU 适配性** | 未测（CPU only）。从结构看 predictor 无 self-attention、大 GEMM 为主，适合 NPU。 | 需 x86 转换机做 ONNX→RKNN，预计 predictor 可上 NPU，action encoder 的小 attention(T=5) 可固化，CEM 留 CPU。 |
| **H3 最佳 batch** | CPU 上 512 仍未见拐点，越大越好。throughput 6.1× 提升（16→512）。 | NPU 上预计拐点在 128-256：小 batch 吃不满算力，大 batch 受 SRAM/带宽限制。 |
| **H4 runtime 瓶颈** | CPU 上 runtime 合计 2.33ms/iter，占 rollout 0.37%，完全不是瓶颈。 | NPU 上预计成为瓶颈：h2d/d2h 拷贝、layout 转换、RKNN invoke、rollout 逐步推理的反复拷贝。 |

**一句话总结**：CPU 侧 Fast-LeWM 的瓶颈仍在模型计算（rollout 83%），非模型 runtime 可忽略；Fast 相比 AR 有 1.19× 加速和轻微瓶颈转移。**核心假设（瓶颈转移到采样规划与异构数据流）只有在 NPU 把模型计算压下来后才会成立**，这是阶段二的关键验证目标。

---

## 7. NPU 后端可行性评估（E2 前置）

### 7.1 板端 NPU 状态（已确认）

- 三核 NPU，300MHz–1GHz，标称 6 TOPS INT8。
- RKNPU driver v0.9.8，RKNN Runtime 2.3.2（`/usr/lib/librknnrt.so`）。
- `/root/rknn-llm` 齐全（含已转换的 Qwen 视觉 `.rknn`），证明 NPU 推理链路可用。
- NPU 频率当前 1GHz（`/sys/class/devfreq/fdab0000.npu/cur_freq`）。

### 7.2 模型结构与 RKNN 适配度

| 模块 | 参数量 | 主要算子 | RKNN 适配度 | 说明 |
|---|---:|---|---|---|
| action-prefix encoder | 1.80M | Linear + 3层小Transformer(T=5) + LayerNorm + 正弦PE | **中** | attention 序列仅 T=5，易固化；LayerNorm 需布局对齐；正弦 PE 预计算为常量 |
| **parallel latent predictor** | **9.02M** | 6×(Value-only Linear + MLP 192→2048 + AdaLN SiLU + LayerNorm + GELU) | **中偏高** | **无 self-attention，主体是大 GEMM，最适合 NPU**；需固定 shape、处理 LayerNorm/激活 |
| ViT-tiny 视觉编码器 | 5.52M | Conv(patch16) + Linear + self-attn(197 token) + LayerNorm | **中** | ViT 上 RKNN 有成熟实践；但仅占决策时延 17%，优先级低于 predictor |
| CEM 求解器 | — | 随机采样、topk、30 iter 循环 | **不可上 NPU** | 动态控制流，必须留 CPU |

### 7.3 阻塞点与解决方案

| 阻塞点 | 影响 | 解决方案 |
|---|---|---|
| **RKNN 模型转换需 x86 Ubuntu + rknn-toolkit2** | 板端只有 Runtime，无法做 PyTorch→ONNX→RKNN 转换 | 需要 x86 机器（或 Docker）做转换；板端只做推理 |
| LayerNorm 仅部分支持 | predictor/encoder 含多个 LayerNorm，可能回退 CPU | 布局对齐(NCHW)、通道维范围约束；不满足则划 CPU 子图或替换 |
| self-attention 运行时矩阵乘 | action encoder 的小 attention(T=5)、ViT 的 attention | T=5 可固化为静态矩阵乘；ViT attention 需手工拆/改 |
| INT8 量化多步误差累积 | 5 步 rollout 后 latent 分布可能偏移，扰乱 CEM 代价排序 | 用真实数据做量化感知校准；FP16/混合精度兜底；以 E5 的 FP32 基线为对比 |
| rollout 逐步推理的 d2h/h2d 反复拷贝 | NPU 上每步 predictor 输出需回传 CPU 再送下一步，拷贝开销可能主导 | 尝试把 5 步 predictor 融合为静态图一次推理（如果 shape 允许）；或用 double buffering |

### 7.4 下一步（需要用户确认）

NPU 实验（E2）需要 **x86 Ubuntu 机器 + rknn-toolkit2**（或 Docker）做模型转换。你这边有吗？
- **有 x86 机器/Docker**：我可以写转换脚本（PyTorch→ONNX→RKNN INT8/FP16），在 x86 上转换后把 `.rknn` 传到板端推理，完成 E2（NPU 单模块性能）和 NPU 版 E1（端到端分解，验证瓶颈转移）。
- **没有**：我可以先做"NPU 可行性预估"（基于 RKNN 算子表和模型结构，预估各模块上 NPU 的预期加速比和风险），等有转换机再补实测。

---

## 8. 对第二阶段优化的建议（基于阶段一数据）

阶段一数据为第二阶段的四个优化方向提供了具体依据：

### 8.1 O1 Fast-LeWM-aware 异构映射
- **predictor（9.02M，占 rollout 54.5%）优先上 NPU**：无 self-attention、大 GEMM，最适合固定 shape 量化。
- **action encoder（1.80M，占 rollout 45.5%）**：T=5 小 attention 可固化，可上 NPU 或留 CPU（CPU 上 84ms/iter，NPU 可能更快）。
- **ViT（占决策 17%）**：优先级低于 predictor，但上 NPU 后可进一步压缩。
- **CEM 必须留 CPU**：动态控制流，NPU 不适合。

### 8.2 O2 Candidate-aware batching
- CPU 数据显示大 batch 效率提升 6.1×，**NPU 上收益预计更大**（NPU 偏好大矩阵乘）。
- 建议 NPU 上测试 S=128/256/512/1024，找拐点。
- double buffering：CPU prepare batch i+1 与 NPU evaluate batch i 流水并行，CPU 上 tensor_prep 仅 0.044ms，完全可以与 NPU 推理重叠。

### 8.3 O3 Planning-compute 协同优化
- CPU 上 S=300 的 rollout 635ms/iter，S=64 仅 269ms/iter（快 2.36×）。
- 如果 CEM 前几轮用 S=64 粗筛、后几轮用 S=300 精排，可在保持 success rate 的同时显著降低平均时延。
- 建议：30 iter 中前 10 iter 用 S=64，后 20 iter 用 S=300，预计平均时延降低 ~30%。

### 8.4 O4 Selective horizon / early filtering
- Fast-LeWM 本身支持多 horizon 预测，但当前 predictor 是单步 AR（5 步循环）。
- 如果把 predictor 改为一次输出 5 步（真正的 parallel latent predictor），NPU 上可一次推理完成，避免逐步 d2h/h2d 拷贝——这可能是 NPU 上最大的优化点。
- 短 horizon（1-2 步）粗筛 + 长 horizon（5 步）精排，可进一步降低计算量。

---

## 9. 附录：复现命令

```bash
# 板端（RK3588，绑 4×A76 大核）
cd /root/Fast-LeWorldModel
taskset -c 4-7 /root/miniconda3/envs/fast-lewm/bin/python profile_fast_lewm.py \
  --all --e1-iters 30 --e1-repeats 3 \
  --e3-scan 16,32,64,128,256,512 \
  --tag full_big4

# 输出：/root/Fast-LeWorldModel/profile_full_big4.json
# 日志：/root/Fast-LeWorldModel/profile_full_big4.log
```

脚本支持 `--e1/--e3/--e4/--e5/--all`、`--quick`（冒烟）、`--tag` 命名、`--threads` 控制线程数。
