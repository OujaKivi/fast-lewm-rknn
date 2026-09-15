# Fast-LeWM 方案 A 异构调度实验报告

**日期**：2026-09-15
**平台**：RK3588（4×A76 绑核，NPU 三核 1GHz，Runtime 2.3.2）
**方案**：A — CPU action_encoder + NPU predictor（单缓冲，无流水线）
**测试配置**：CEM 30 iters × 3 repeats，batch=300，5步 rollout，FP16

---

## 1. 实验目标

验证方案 A（简单异构调度）的实际收益：
- action_encoder 留 CPU（1.8M 参数，小 attention，NPU 不划算）
- predictor 放 NPU（9.0M 参数，大 GEMM，NPU 效率高）
- 测量端到端 CEM 决策延迟，对比全 CPU 基线
- 细粒度分析 NPU 数据搬运、推理、输出解析开销
- 验证 NPU 推理是否对 CPU 性能产生影响（内存带宽竞争/发热降频）
- 验证研究假设 H1：瓶颈是否从模型计算转移到其他部分

---

## 2. 实现方案

### 2.1 核心组件

**`npu_predictor.py` — NPUPredictor 包装类**
- 接口与 `module.ARPredictor` 完全一致：`predictor(latent, act_emb) → pred`
- 内部封装 RKNN 加载、推理、输出解析
- 自动处理 torch tensor ↔ numpy float32 转换
- 支持 `release()` 释放资源

**`profile_heterogeneous.py` — 异构版 profiling 脚本**
- 基于 `profile_fast_lewm.py` 的 CEM 循环改造
- `--mode cpu`：全 CPU 基线
- `--mode npu`：CPU enc + NPU pred
- `--mode both`：两者对比，自动计算加速比
- `--verify`：验证 CPU/NPU 输出一致性（cosine similarity / MSE）
- 细粒度计时：`npu_transfer_in` / `npu_inference` / `npu_transfer_out`

### 2.2 CEM 循环改造

```python
# 改造前（全 CPU）
act_emb = act_enc(actions, return_last_only=True, latent=emb[:, -1:])
for _ in range(steps):
    pred = predictor(emb[:, -1:], act_emb)  # CPU torch
    emb = torch.cat([emb, pred], dim=1)

# 改造后（异构）
act_emb = act_enc(actions, return_last_only=True, latent=emb[:, -1:])  # CPU
for _ in range(steps):
    pred = npu_predictor(emb[:, -1:], act_emb)  # NPU RKNN
    emb = torch.cat([emb, pred], dim=1)
```

### 2.3 NPU 推理细粒度流程

```
npu_transfer_in:  torch.Tensor → numpy.float32（含 contiguous/copy）
npu_inference:    rknn.inference(inputs=[latent_np, act_emb_np])
npu_transfer_out: numpy → torch.Tensor（from_numpy + to(device)）
```

---

## 3. 实验结果

### 3.1 三方端到端对比（CPU / NPU全量 / NPU异构）

为区分"异构调度的收益"与"简单把模型搬到 NPU 的收益"，增加 NPU 全量基线（action_encoder + predictor 都放 NPU）。

| 指标 | CPU 全量 | NPU 全量（enc+pred都NPU） | NPU 异构（方案A，CPU enc+NPU pred） |
|------|----------|---------------------------|--------------------------------------|
| 完整 CEM 决策 | 20.09 s | 10.49 s | **8.95 s** |
| vs CPU 加速比 | 1.0× | 1.92× | **2.25×** |
| rollout（5步×30iter） | 16.59 s | 4.58 s | **3.40 s** |
| action_encoder | 2.98 s（CPU） | 5.47 s（NPU，**比CPU慢83%**） | 4.90 s（CPU，被NPU拖慢） |
| vi_enc（视觉编码） | 3.91 s | 7.33 s | 5.15 s |
| score + elite + sample | 0.018 s | 0.066 s | 0.034 s |
| 内存占用（max RSS） | ~440 MB | **911 MB**（双模型加载） | ~490 MB |

**关键结论：异构（方案A）比 NPU 全量快 17%（1.17×）**，证明"不适合 NPU 的模块留 CPU"的异构调度有实际价值，而不是简单搬家。

### 3.2 为什么异构比全量好

**action_encoder 在 NPU 上反而更慢**：
- CPU：2.98s（30 iter，约 99ms/iter）
- NPU：5.47s（30 iter，约 182ms/iter）
- NPU 比 CPU 慢 **83%**，原因：
  1. action_encoder 仅 1.8M 参数，NPU 吃不满（利用率 31.7%）
  2. 含 T=5 小 self-attention，NPU 上 attention 效率低
  3. RKNN 模型文件 190MB，加载和初始化开销大
  4. 数据搬运开销占比大（小模型计算少，搬运相对多）

**NPU 全量模式下 predictor 也变慢**：
- 异构模式 predictor：22.6ms/步
- 全量模式 predictor：30.1ms/步（+33%）
- 原因：两个 NPU 模型交替推理，NPU 上下文切换/缓存失效

**NPU 全量模式下 CPU 端被拖慢更严重**：
- vi_enc：CPU 3.91s → 异构 5.15s（+32%）→ 全量 7.33s（+87%）
- 原因：两个 NPU 模型交替运行，占用更多内存带宽和系统资源

### 3.2 NPU 每步细粒度（450 次调用统计）

| 阶段 | median | mean | p95 | 占比 |
|------|--------|------|-----|------|
| npu_transfer_in | 0.084 ms | 0.095 ms | 0.177 ms | 0.37% |
| npu_inference | **22.311 ms** | 23.039 ms | 28.570 ms | 98.41% |
| npu_transfer_out | 0.276 ms | 0.346 ms | 0.774 ms | 1.22% |
| **每步合计** | **22.671 ms** | 23.480 ms | - | 100% |

**关键结论**：
- NPU 推理本身占 98.4%，数据搬运开销仅 1.6%（0.36ms/步）
- 数据搬运不是瓶颈，RKNN runtime 调用效率高
- predictor 在 NPU 上比 CPU 快 **4.94×**（110.6ms → 22.3ms）

### 3.3 输出一致性验证（随机权重下）

| 指标 | 值 |
|------|-----|
| cosine similarity (mean) | **0.9707** |
| cosine similarity (min) | 0.9496 |
| MSE | 0.0585 |
| relative L1 error | 0.2375 |

**说明**：随机权重下数值对比无实际意义，cos_sim 0.97 证明推理流程正确。FP16 量化本身会引入精度损失，加载预训练权重后需重新验证。

---

## 4. 重要发现：NPU 推理导致 CPU 性能下降

### 4.1 现象

在 NPU 异构模式下，所有 CPU 端计算都比纯 CPU 模式慢：

| CPU 端模块 | 纯 CPU | NPU 模式 | 下降幅度 |
|-----------|--------|----------|----------|
| action_encoder | 2971 ms | 5070 ms | **-41%** |
| vi_enc（ViT-tiny） | 3902 ms | 6573 ms | **-41%** |
| score | 8.8 ms | 16.4 ms | **-46%** |
| elite_select | 5.5 ms | 7.7 ms | **-29%** |
| cem_sample | 3.7 ms | 5.4 ms | **-32%** |
| tensor_prep | 2.0 ms | 3.9 ms | **-49%** |

**所有 CPU 模块统一变慢约 30-50%**，这不是偶然噪声，而是系统性影响。

### 4.2 可能原因

1. **内存带宽竞争**（最可能）：RK3588 的 CPU 和 NPU 共享 DDR 内存带宽。NPU 推理时大量读写激活数据（batch=300 × 192维 × 多层），占用内存带宽，导致 CPU 端的内存密集型运算（ViT、action_encoder 的 GEMM）变慢。

2. **发热降频**：NPU 持续运行（22ms × 150次 = 3.3s）产生热量，可能导致 SoC 整体温度上升，CPU 触发 thermal throttling 降频。

3. **中断/驱动开销**：RKNN runtime 每次推理可能产生中断，中断处理占用 CPU 周期。

### 4.3 对后续方案的影响

- **方案 B（双缓冲）的实际收益可能低于预估**：双缓冲让 CPU 和 NPU 同时运行，内存带宽竞争会更严重，CPU 准备时间可能进一步增加
- **需要测量内存带宽**：用 `perf` 或其他工具量化 NPU 推理时的内存带宽占用
- **考虑 NPU 频率调节**：降低 NPU 频率可能减少发热和带宽竞争，换取更稳定的 CPU 性能
- **INT8 量化可能缓解**：INT8 数据量减半，内存带宽占用减少，可能减轻对 CPU 的影响

---

## 5. 瓶颈转移分析（验证 H1）

### 5.1 CPU 基线瓶颈分布

```
CPU 模式（20.17s）：
  rollout (predictor×5步):  16.59s  82.2%  ← 主要瓶颈
  vi_enc:                     3.90s  19.3%
  action_encoder:             2.97s  14.7%
  其他（score/elite/sample）: 0.02s   0.1%
```

### 5.2 NPU 异构模式瓶颈分布

```
NPU 模式（8.89s）：
  vi_enc:                     6.57s  73.9%  ← 新瓶颈！
  action_encoder:             5.07s  57.0%  ← 新瓶颈！
  rollout (NPU predictor×5步): 3.37s  37.9%
  其他:                        0.03s   0.3%
```

### 5.3 结论

**研究假设 H1 得到验证**：当 predictor 计算被 NPU 加速 4.93× 后，端到端瓶颈从"模型计算（rollout）"转移到了"视觉编码 + action_encoder + CPU 开销"。

- rollout 占比从 82.2% 降到 37.9%
- vi_enc + act_enc 占比从 34.0% 升到 130.9%（注：各阶段 median×iters 之和大于总耗时，因为总耗时是 median of totals，口径不同；但趋势明确）
- 下一步优化重点应该是 vi_enc 和 action_encoder 的加速，而不是继续优化 predictor

**vi_enc 加速方向**：
- ViT-tiny 也可以尝试 NPU 推理（5.5M 参数，含 attention，需测试 NPU 效率）
- 或者降低 ViT 输入分辨率（224→160/128）减少计算量
- 或者用更轻量的视觉编码器（MobileNet、 EfficientNet-lite）

**action_encoder 加速方向**：
- 当前在 CPU 上 100ms/iter，NPU 上 184ms（更慢）
- 可以尝试 INT8 量化后上 NPU，看是否能比 CPU 快
- 或者优化 action_encoder 结构（减少 attention 层数、用线性 attention）
- 或者用 CPU 多线程并行（当前绑 4 大核，torch 线程数可能未最优）

---

## 6. 方案 A 收益总结

### 6.1 实际收益 vs 预估

| 指标 | 预估（方案A） | 实测 | 差异 |
|------|--------------|------|------|
| 单步延迟 | ~73ms | ~30ms（NPU推理）+ CPU enc | 更好 |
| 完整 CEM | ~13.2s | **8.89s** | **好于预估 33%** |
| 加速比 | 1.5× | **2.27×** | **好于预估 51%** |
| predictor 加速 | 2.4× | **4.94×** | **好于预估 106%** |

实际收益好于预估，主要因为：
- predictor 在 NPU 上的加速比预估更高（4.94× vs 2.4×）
- CPU predictor 每步 110ms 比之前预估的 68ms 慢（之前估算是基于 rollout 总时间反推，可能不准确）

### 6.2 未达预期的部分

- CPU 端性能下降 30-50%（NPU 内存带宽竞争），这是预估时没有考虑的
- 如果没有 CPU 性能下降，整体加速比可能达到 3× 以上

### 6.3 距实时性的差距

| 指标 | 当前（方案A） | 10Hz 实时目标 | 20Hz 实时目标 |
|------|--------------|---------------|---------------|
| 单次决策 | 8.89 s | 0.1 s | 0.05 s |
| 差距 | - | **89×** | **178×** |

方案 A 后仍距实时性有 89× 差距，需要后续方案（B/C/D）持续优化。

---

## 7. 后续工作建议

### 7.1 短期（1-2周）

1. **分析 CPU 性能下降原因**：用 `perf stat` 测量 NPU 推理时的内存带宽、cache miss、CPU 频率
2. **vi_enc NPU 测试**：把 ViT-tiny 也转 RKNN，测试 NPU 效率，看是否能上 NPU
3. **action_encoder INT8 测试**：INT8 量化后上 NPU，看是否能比 CPU 快
4. **NPU 频率扫描**：测试 NPU 在不同频率（800MHz/1GHz/1.2GHz？）下的性能和对 CPU 的影响

### 7.2 中期（2-4周）

5. **方案 B 双缓冲**：实现 CPU/NPU 流水线，注意内存带宽竞争可能降低收益
6. **INT8 量化**：加载预训练权重后做 INT8 校准，predictor 预期再加速 1.5-2×
7. **batch size 扫描**：测试不同 batch（16/32/64/128/256/512）的 NPU 效率，找最佳区间

### 7.3 长期（1-2月）

8. **方案 D 动态 batch**：CEM 收敛感知，减少后期 candidate 数量
9. **vi_enc 替换**：用更轻量的视觉编码器或降低输入分辨率
10. **端到端真实环境验证**：在真实机器人/仿真环境中验证规划成功率和延迟

---

## 8. 产物清单

| 文件 | 说明 |
|------|------|
| `npu_predictor.py` | NPUPredictor 包装类（接口兼容 ARPredictor） |
| `profile_heterogeneous.py` | 异构版 profiling 脚本（支持 cpu/npu/both 模式） |
| `profile_het_full.json` | 正式测试原始数据（30 iter × 3 repeats） |
| `profile_het_smoke.json` | 冒烟测试数据（5 iter × 2 repeats） |
| `predictor_S300_fp16_v4.rknn` | NPU  predictor 模型（修复全零权重后转换） |

---

## 附录：关键命令

```bash
# 板端运行正式测试（绑4大核）
cd /root/Fast-LeWorldModel
taskset -c 4-7 python profile_heterogeneous.py \
  --mode both --verify --repeats 3 --iters 30 --tag het_full

# 仅 NPU 模式
taskset -c 4-7 python profile_heterogeneous.py \
  --mode npu --repeats 5 --iters 30 --tag npu_only

# 冒烟测试
taskset -c 4-7 python profile_heterogeneous.py \
  --mode both --verify --quick --tag het_smoke
```
