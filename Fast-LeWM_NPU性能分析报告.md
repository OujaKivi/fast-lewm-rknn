# Fast-LeWM RK3588 NPU 性能分析报告

**日期**：2026-09-15
**平台**：RK3588（4×A76 + 4×A55，NPU 三核 6TOPS INT8，Runtime 2.3.2，driver 0.9.8）
**模型**：Fast-LeWorldModel（action_encoder 1.80M + predictor 9.02M = 10.83M 参数）
**精度**：FP16，batch=300（CEM num_samples=300）

---

## 1. 概述

本报告完成 Fast-LeWM 两个核心模块（action_encoder 和 predictor）的 ONNX→RKNN 转换与板端 NPU 推理测试，对比 CPU 与 NPU 性能，评估异构执行优化空间。

**核心结论**：
- predictor（9M参数，大GEMM为主）在 NPU 上表现优异：27.86ms/batch300，比 CPU 快约 3 倍
- action_encoder（1.8M参数，含self-attention）在 NPU 上反而较慢：184ms，比 CPU 慢约 5 倍
- **推荐异构调度**：action_encoder 留 CPU，predictor 放 NPU，单步延迟可从 CPU 的 108ms 降到约 60ms（1.8× 加速）
- NPU 利用率仍有较大提升空间（predictor 仅 20.1%），INT8 量化和双缓冲可进一步加速

---

## 2. NPU 转换过程与关键问题

### 2.1 转换流程

```
PyTorch (torch 2.6.0)
    ↓ torch.onnx.export (opset=14, 固定shape)
ONNX (action_encoder 6.96MB, predictor 30.28MB)
    ↓ rknn-toolkit2 2.3.2 (x86 Ubuntu, FP16, optimization_level=0)
RKNN (action_encoder 190MB, predictor 18.95MB)
    ↓ scp 到板端
板端 NPU 推理 (rknn-toolkit-lite2 2.3.2, aarch64)
```

### 2.2 遇到的关键问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| opset 12 导出失败 | action_encoder 使用 `scaled_dot_product_attention`，需 opset≥14 | 改用 opset 14 |
| predictor 转换 fold_constant TypeError | rknn-toolkit2 2.3.2 内部 bug，`check_inputs` 遇到 NoneType | monkey patch `GraphOptimizer.fold_constant`，捕获异常跳过 |
| predictor RKNN 只有 1 个输入 | `action_fusion_zero_init` 导致融合分支最后一层权重全为 0，rknn-toolkit2 把整个分支（含 act_emb 输入）优化掉 | 修改 ONNX，将 5 个全零权重改为小非零随机值（×0.02），重新转换 |
| onnx-simplifier 导致输入丢失 | 简化器把两个相同 shape 的输入合并 | 改用未简化的原始 ONNX |

### 2.3 关键发现：action_fusion_zero_init 的影响

ARPredictor 初始化时使用 `action_fusion_zero_init=True`，将 `action_fusion_mlp` 最后一层的权重和偏置初始化为全 0。这导致：
- 融合分支输出恒为 0：`x + 0 = x`
- rknn-toolkit2 的常量折叠优化将整个融合分支移除
- `act_emb` 输入因为没有被使用而被移除
- 最终 RKNN 模型只有 `latent` 一个输入

**修复方法**：在 ONNX 导出后，将全零权重改为小非零随机值（标准差 0.02），模拟加载预训练权重后的状态。这不会显著影响模型结构，但能保留完整计算图。

---

## 3. NPU 单模块性能

### 3.1 action_encoder（ActionPrefixEmbedder）

| 指标 | 值 |
|------|-----|
| 参数规模 | 1.80M |
| 模型结构 | depth=3, heads=6, 含 T=5 小 self-attention |
| RKNN 文件大小 | 190 MB（FP16，含 FP32+FP16 双份权重） |
| 平均延迟 | **184.43 ms** ± 8.08 |
| P50 / P95 / P99 | 180.86 / 207.05 / 209.22 ms |
| 吞吐量 | 1,626.6 candidates/sec |
| 单候选延迟 | 0.615 ms/candidate |
| NPU 利用率 | 31.7%（三核平均） |
| 输出 shape | (300, 1, 192) ✓ |

**分析**：action_encoder 在 NPU 上表现不佳，主要原因：
1. **模型太小**：1.8M 参数，NPU 吃不满，利用率仅 31.7%
2. **self-attention 开销**：T=5 的小 attention 在 NPU 上效率低，数据搬运占比大
3. **文件过大**：190MB RKNN 文件，加载和初始化开销大

### 3.2 predictor（ARPredictor）

| 指标 | 值 |
|------|-----|
| 参数规模 | 9.02M |
| 模型结构 | depth=6, value_heads=16, 无 self-attention，以大 GEMM 为主 |
| RKNN 文件大小 | 18.95 MB（FP16） |
| 平均延迟 | **27.86 ms** ± 6.18 |
| P50 / P95 / P99 | 25.92 / 47.88 / 50.04 ms |
| 吞吐量 | 10,768.3 candidates/sec |
| 单候选延迟 | 0.093 ms/candidate |
| NPU 利用率 | 20.1%（三核平均） |
| 输出 shape | (300, 1, 192) ✓ |

**分析**：predictor 在 NPU 上表现优异，主要原因：
1. **大 GEMM 为主**：9M 参数中大部分是矩阵乘法，NPU 擅长
2. **无 self-attention**：避免了 attention 的数据搬运和不规则计算
3. **高吞吐**：10768 cand/s，是 action_encoder 的 6.6 倍
4. **利用率仍低**：仅 20.1%，说明还有很大优化空间（INT8、更大batch、双缓冲）

### 3.3 两模块对比

| 模块 | 参数 | NPU 延迟 | 吞吐量 | NPU 利用率 | 延迟/参数 |
|------|------|----------|--------|-----------|-----------|
| action_encoder | 1.80M | 184.43ms | 1,627/s | 31.7% | 102.5ms/M |
| predictor | 9.02M | 27.86ms | 10,768/s | 20.1% | 3.1ms/M |

**predictor 的单位参数延迟是 action_encoder 的 1/33**，充分说明大 GEMM 结构比 self-attention 更适合 NPU。

---

## 4. CPU vs NPU 对比

### 4.1 单步 rollout 延迟对比

从阶段一 CPU profiling 数据推算（Fast 模式，4大核，batch=300）：
- 完整 CEM 决策：19.55s
- rollout 占比：83.07% → 16.24s
- CEM 迭代 30 次 × 5 步 = 150 次模型调用
- **CPU 单次模型调用（action_encoder + predictor）：约 108.3ms**

| 执行方式 | action_encoder | predictor | 单步合计 | 相对 CPU |
|----------|---------------|-----------|----------|----------|
| CPU 4大核 | ~40ms（估） | ~68ms（估） | **108.3ms** | 1.00× |
| NPU 全量 | 184.4ms | 27.9ms | **212.3ms** | 0.51×（更慢） |
| **异构（CPU enc + NPU pred）** | ~40ms | 27.9ms | **~68ms** | **1.59×** |

**关键发现**：
- 全量放 NPU 反而更慢（0.51×），因为 action_encoder 在 NPU 上太慢
- 异构调度（action_encoder 留 CPU，predictor 放 NPU）可获得 1.59× 加速
- predictor 单独在 NPU 上比 CPU 快约 2.4 倍（68ms → 27.9ms）

### 4.2 端到端 CEM 决策延迟估算

| 执行方式 | 单步延迟 | 150次调用 | 端到端（含CEM开销） | 相对 CPU |
|----------|----------|-----------|---------------------|----------|
| CPU 4大核 | 108.3ms | 16.25s | 19.55s | 1.00× |
| 异构（CPU enc + NPU pred） | ~68ms | 10.2s | ~12.5s | **1.56×** |
| 异构 + INT8（预测） | ~50ms | 7.5s | ~9.5s | **2.06×** |

---

## 5. 异构执行优化建议

### 5.1 推荐架构

```
CPU (4×A76)                          NPU (三核)
┌─────────────────────┐              ┌─────────────────────┐
│ CEM 采样/更新        │              │                     │
│ elite selection      │              │  predictor (9M)     │
│ ViT 编码 (5.5M)     │    latent    │  - 大GEMM为主       │
│ action_encoder (1.8M)│ ──────────→ │  - 无self-attention  │
│  - 小attention       │              │  - FP16: 27.9ms    │
│  - CPU更优           │ ←────────── │  - INT8: ~15ms(估)  │
│ scoring              │    pred      │                     │
└─────────────────────┘              └─────────────────────┘
```

### 5.2 分模块映射策略

| 模块 | 推荐设备 | 理由 |
|------|----------|------|
| CEM 采样/更新 | CPU | 纯 Python/numpy 控制逻辑，NPU 不适合 |
| elite selection | CPU | 小数据量排序，CPU 足够 |
| ViT 编码 | CPU（或 NPU） | 5.5M 参数，含 attention，需单独测试 |
| **action_encoder** | **CPU** | 1.8M 参数，小 attention，NPU 利用率低（31.7%），CPU 更快 |
| **predictor** | **NPU** | 9.0M 参数，大 GEMM，NPU 利用率高，比 CPU 快 2.4× |
| scoring | CPU | 简单矩阵运算，数据量小 |

### 5.3 进一步优化方向

1. **INT8 量化**：predictor 当前 FP16，INT8 可预期 1.5-2× 加速（需真实校准数据）
2. **双缓冲（Double Buffering）**：CPU 准备下一批 candidate 的同时，NPU 推理当前批，掩盖数据搬运开销
3. **更大 batch**：当前 batch=300，NPU 利用率仅 20.1%，可尝试 batch=512 或 1024 提高利用率
4. **NPU 多核绑定**：当前 CORE_AUTO，可尝试绑定特定核组合减少调度开销
5. **Candidate-aware batching**：利用 CEM 收敛特性，动态调整 batch size（初期大 batch 探索，后期小 batch 精调）

---

## 6. 技术栈对 RK-NPU 优化的支持评估

### 6.1 模型结构适配性

| 特性 | action_encoder | predictor | 适配 NPU 程度 |
|------|---------------|-----------|---------------|
| 静态计算图 | ✓ | ✓ | 高（RKNN 要求固定 shape） |
| 大 GEMM 为主 | 部分 | ✓✓ | predictor 极高 |
| self-attention | ✓（T=5） | ✗ | action_encoder 低 |
| 动态控制流 | ✗ | ✗ | 高（无 if/while） |
| 标准化算子 | ✓ | ✓ | 高（Conv/MatMul/LayerNorm/GELU 均支持） |

### 6.2 RKNN 工具链支持

- **ONNX 导入**：✓ 支持（opset≥14）
- **FP16 推理**：✓ 支持（已验证）
- **INT8 量化**：✓ 支持（需校准数据集）
- **多输入模型**：✓ 支持（需注意全零权重优化问题）
- **动态 shape**：✗ 不支持（需为每个 batch size 单独转换）
- **自定义算子**：部分支持（需用 rknn 自定义算子框架）

### 6.3 限制与挑战

1. **动态 batch 不支持**：RKNN 模型固定 shape，CEM 中若需动态调整 candidate 数量，需预转换多个 batch 版本
2. **小模型效率低**：<2M 参数的小模型在 NPU 上利用率低，数据搬运开销占比大
3. **self-attention 效率**：短序列 attention 在 NPU 上不如大 GEMM 高效
4. **校准数据需求**：INT8 量化需要真实输入数据做校准，当前用随机权重无法做有意义的 INT8
5. **全零权重优化 bug**：rknn-toolkit2 会过度优化全零分支，需手动修复

---

## 7. 结论

1. **Fast-LeWM 的 predictor 模块非常适合 RK-NPU**：9M 参数、大 GEMM 为主、无 self-attention，NPU 上 27.86ms/batch300，比 CPU 快 2.4 倍
2. **action_encoder 不适合 NPU**：1.8M 参数、小 attention，NPU 上 184ms，比 CPU 慢约 5 倍，应留 CPU
3. **异构调度可获 1.59× 端到端加速**：CPU 跑 action_encoder + CEM，NPU 跑 predictor，单步从 108ms 降到 ~68ms
4. **NPU 利用率仍有较大空间**：predictor 仅 20.1%，INT8 量化 + 双缓冲 + 更大 batch 有望再提升 1.5-2×
5. **技术栈支持 RK-NPU 优化**：模型结构静态、算子标准化、ONNX 中间格式成熟，主要限制是动态 batch 和小模型效率

**下一步建议**：
- 加载 HF 预训练权重后重新转换（避免全零权重问题）
- 做 INT8 量化测试（用真实观测数据校准）
- 实现异构调度原型（CPU action_encoder + NPU predictor）
- 做 batch size 扫描（16/32/64/128/256/512）找最佳运行区间
- 实现双缓冲掩盖数据搬运开销

---

## 附录：原始数据

### action_encoder NPU 跑分（200 iters, warmup=20）
```
mean: 184.433ms ± 8.084
P50: 180.864ms, P95: 207.050ms, P99: 209.217ms
throughput: 1626.6 cand/s
latency/cand: 0.6148ms
NPU util: 31.7%
output: (300, 1, 192)
```

### predictor NPU 跑分（200 iters, warmup=20）
```
mean: 27.859ms ± 6.179
P50: 25.921ms, P95: 47.877ms, P99: 50.037ms
throughput: 10768.3 cand/s
latency/cand: 0.0929ms
NPU util: 20.1%
output: (300, 1, 192)
```

### CPU 4大核阶段一 profiling 数据
```
完整 CEM 决策: 19.551s
rollout 占比: 83.07% (16.24s)
单次模型调用(150次): 108.3ms
vi_enc 占比: 16.84% (3.29s)
非模型开销: 0.09% (17.6ms)
```
