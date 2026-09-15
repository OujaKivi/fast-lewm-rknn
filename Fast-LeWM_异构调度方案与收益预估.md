# Fast-LeWM 异构调度方案与收益预估

**日期**：2026-09-15
**平台**：RK3588（4×A76 + 4×A55，NPU 三核 6TOPS INT8 / 3TOPS FP16）
**基线**：CPU 4大核完整 CEM 决策 19.55s（单步 rollout 108ms，30 iter × 5 step）

---

## 1. 当前状态说明

### 1.1 权重状态：随机初始化，未加载预训练

目前所有 ONNX 导出、RKNN 转换和 NPU 跑分使用的是**模型默认随机初始化权重**，未加载 HF 预训练权重（`https://huggingface.co/naiverer/fast-leworldmodel`）。

- `export_onnx.py` 直接 `build_models()` 后导出，无 `load_state_dict`
- `bench_npu.py` 输入用 `np.random.randn`，仅验证 shape 不验证数值
- 之前遇到的 `action_fusion_zero_init` 全零权重问题，正是随机初始化 + zero_init 导致融合分支最后一层恒为 0

**影响**：NPU 延迟数据在算子层面可信（计算图结构正确），但数值精度无意义。加载预训练权重后需重新转换和跑分，预期延迟变化不大（权重值不影响计算量），但可避免全零权重补丁，且能做 INT8 校准。

### 1.2 NPU 当前性能（FP16, batch=300）

| 模块 | 参数 | NPU 延迟 | 吞吐量 | NPU 利用率 | 适合 NPU |
|------|------|----------|--------|-----------|----------|
| action_encoder | 1.80M | 184.4ms | 1,627/s | 31.7% | ❌ 太小，attention 效率低 |
| predictor | 9.02M | 27.9ms | 10,768/s | 20.1% | ✅ 大GEMM，比CPU快2.4× |

---

## 2. INT8 量化分析

### 2.1 原理

RK3588 NPU 的 **6TOPS 标称算力是 INT8**，FP16 只有约 3TOPS。INT8 收益来自三方面：

| 收益来源 | 效果 |
|----------|------|
| 算力翻倍 | INT8 MAC 单元是 FP16 的 2 倍 |
| 内存带宽减半 | 权重和激活体积减半，对 memory-bound 的小模型尤其有利 |
| NPU 算子更友好 | 大 GEMM 的 INT8 实现通常比 FP16 更高效 |

### 2.2 分模块预估

| 模块 | FP16 当前 | INT8 预估 | 加速比 | 理由 |
|------|-----------|-----------|--------|------|
| predictor（大GEMM为主） | 27.9ms | **14-19ms** | 1.5-2.0× | 9M参数全是矩阵乘法，INT8 收益最大 |
| action_encoder（含attention） | 184ms | **120-160ms** | 1.15-1.5× | attention 的 softmax/归一化对量化敏感，部分算子仍跑 FP16 |

### 2.3 前提条件与风险

- **必须先加载 HF 预训练权重 + 准备校准数据集**：INT8 需要真实校准数据（用真实观测/动作输入跑几十到几百组样本统计激活分布）。随机权重下做 INT8 没有意义
- **精度风险**：INT8 可能带来 1-3% 的精度损失，需验证 rollout 输出与 FP16 的 cosine similarity / MSE，以及最终 CEM 规划成功率是否下降
- **action_encoder 的 INT8 收益有限**：含 attention 的模型对量化敏感，可能需要混合精度（attention 部分 FP16，GEMM 部分 INT8）

---

## 3. 异构调度方案设计

### 3.1 核心思路

把 Fast-LeWM 的 CEM 规划循环拆成 CPU 段和 NPU 段，让两者流水线重叠：

```
当前（CPU 全量）：
  [CEM采样] → [action_enc] → [predictor×5步] → [scoring] → [elite更新]
  CPU 4大核  108ms/步（enc+pred合计），完整CEM 19.55s

目标（异构流水线）：
  CPU: [采样+enc+scoring+elite]  ≈ 45ms/步
  NPU:        [predictor×5步]     ≈ 28ms/步（FP16）
  流水线重叠后每步 ≈ max(45,28) + sync ≈ 50ms
```

**分模块映射策略**：

| 模块 | 推荐设备 | 理由 |
|------|----------|------|
| CEM 采样/更新 | CPU | 纯 Python/numpy 控制逻辑，NPU 不适合 |
| elite selection | CPU | 小数据量排序，CPU 足够 |
| ViT 编码 | CPU（或NPU待验证） | 5.5M 参数，含 attention，需单独测试 |
| action_encoder | **CPU** | 1.8M 参数，小 attention，NPU 利用率低（31.7%），CPU 更快 |
| predictor | **NPU** | 9.0M 参数，大 GEMM，NPU 利用率高，比 CPU 快 2.4× |
| scoring | CPU | 简单矩阵运算，数据量小 |

---

### 3.2 方案 A：简单异构（单缓冲，无流水线）

**设计**：
- 每步：CPU 跑 action_encoder → 结果搬 NPU → NPU 跑 predictor → 结果搬回 CPU → CPU scoring
- 串行执行，NPU 推理时 CPU 空闲
- 改动最小，只需把 predictor 调用替换为 NPU 推理

**实现要点**：
- 写 `NPUPredictor` 包装类，封装 RKNN 加载/推理/输出解析，接口与 PyTorch 版 predictor 一致
- 修改 CEM 循环：action_encoder 仍用 CPU torch，predictor 替换为 NPU 调用
- 处理 CPU↔NPU 数据格式转换（torch tensor ↔ numpy，layout 对齐）

**预估**：
- 单步：CPU enc ~40ms + NPU pred 27.9ms + overhead ~5ms = **~73ms**
- 完整 CEM：**~13.2s**
- 加速比：**1.48×**
- 实现难度：低（1-2天）

---

### 3.3 方案 B：双缓冲流水线

**设计**：
- CEM 拆成两阶段：阶段1 CPU 采样+action_encoder 生成候选，阶段2 NPU 批量推理 predictor
- 用双缓冲：CPU 准备第 i+1 批候选的同时，NPU 推理第 i 批
- 掩盖 CPU 准备时间，NPU 利用率从 20% 提升到 50%+

**实现要点**：
- CEM 循环重构为生产者-消费者模式：CPU 线程生产候选 batch，NPU 线程消费推理
- 双缓冲数组（ping-pong），避免数据竞争
- 同步点：每轮 CEM 结束时等待 NPU 完成当前 batch

**预估**：
- 假设 CPU 准备+enc 约 45ms，NPU pred 27.9ms
- 流水线后每步延迟 = max(45, 27.9) + sync overhead ≈ **50ms**
- 完整 CEM：**~9.0s**
- 加速比：**2.17×**
- 实现难度：中（3-5天）

---

### 3.4 方案 C：INT8 + 双缓冲

**设计**：
- 在方案 B 基础上，predictor 用 INT8（14-19ms）
- NPU 推理进一步缩短，但 CPU 准备时间（45ms）成为瓶颈

**预估**：
- 单步：**~48ms**（CPU 瓶颈）
- 完整 CEM：**~8.6s**
- 加速比：**2.27×**
- 实现难度：中高（需校准数据 + 精度验证）

**注意**：INT8 的额外收益被 CPU 瓶颈吃掉了一部分。如果同时优化 CPU 端（如用 numba 加速 CEM 采样、用多线程并行 action_encoder），可以进一步降低 CPU 段时间，释放 INT8 收益。

---

### 3.5 方案 D：动态 batch + CEM 收敛感知

**设计**：
- CEM 初期（前 10 轮）用大 batch（512）广泛探索，后期（后 20 轮）收敛后用小 batch（64-128）精调
- 利用 NPU batch 效率曲线（batch 越大单候选越便宜），但收敛后不需要那么多候选
- 平均有效 batch 从 300 降到 ~180-200
- 需为不同 batch size 预转换多个 RKNN 模型（RKNN 固定 shape）

**预估**：
- 完整 CEM：**~6-7s**
- 加速比：**2.8-3.3×**
- 实现难度：高（需改 CEM 逻辑 + 多 batch 模型管理）

---

## 4. 收益区间汇总

| 方案 | 单步延迟 | 完整CEM(30iter×5step) | 加速比 | 实现难度 | 预计工时 |
|------|----------|------------------------|--------|----------|----------|
| CPU 基线（当前） | 108ms | 19.55s | 1.0× | - | - |
| A 简单异构 | ~73ms | ~13.2s | **1.5×** | 低 | 1-2天 |
| B 双缓冲流水线 | ~50ms | ~9.0s | **2.2×** | 中 | 3-5天 |
| C INT8+双缓冲 | ~48ms | ~8.6s | **2.3×** | 中高 | 5-7天 |
| D 动态batch+收敛感知 | ~35-40ms | ~6-7s | **2.8-3.3×** | 高 | 7-10天 |

---

## 5. 推荐路径

### 第一阶段：方案 A 验证（1-2天）

先做方案 A 验证异构可行性，拿到真实数据后再决定是否投入方案 B。

核心工作：
1. 写 `NPUPredictor` 包装类，封装 RKNN 加载/推理/输出解析，接口与 PyTorch 版 predictor 一致
2. 修改 CEM 循环：action_encoder 仍用 CPU torch，predictor 替换为 NPU 调用
3. 处理 CPU↔NPU 数据格式转换（torch tensor ↔ numpy，layout 对齐）
4. 跑分对比 CPU 基线，验证输出一致性（cosine similarity）

### 第二阶段：方案 B 优化（3-5天）

如果方案 A 验证通过且收益符合预期，投入双缓冲流水线优化。

### 第三阶段：INT8 + 动态 batch（按需）

在方案 B 基础上，根据精度验证结果决定是否做 INT8；根据 CEM 收敛特性决定是否做动态 batch。

---

## 6. 关键风险点

### 6.1 数据搬运开销
每次 NPU 推理需要把 latent+act_emb 从 CPU 搬到 NPU，结果搬回。batch=300 时每次搬运约 300×1×192×4×2 ≈ 460KB，PCIe 带宽下约 0.1-0.3ms，可忽略。

### 6.2 NPU 推理调用开销
`rknn.inference` 本身有 ~0.5-1ms 的 Python/runtime 开销，对 28ms 的 predictor 影响约 2-4%。5 步 rollout 中每步都要调用 NPU，5 次调用的 runtime 开销累计约 2.5-5ms。

### 6.3 预训练权重依赖
方案 A/B/C 都需要先加载 HF 预训练权重，否则：
- 全零权重问题需要手动补丁
- INT8 校准无法进行
- 输出精度无法验证

### 6.4 RKNN 固定 shape 限制
RKNN 模型固定输入 shape，方案 D（动态 batch）需要为每个 batch size 预转换一个 RKNN 模型，并在运行时根据当前 batch 选择对应模型。模型切换有加载开销（~100-500ms），需要预加载所有模型到内存。

### 6.5 NPU 利用率上限
当前 predictor NPU 利用率仅 20.1%，理论上还有 5× 提升空间。但实际受限于：
- 模型本身计算量（9M 参数，batch=300 时约 5.2 GFLOPs）
- NPU 频率（1GHz，三核约 3TOPS FP16）
- 理论最低延迟 = 5.2 GFLOPs / 3 TOPS ≈ 1.7ms，当前 27.9ms，说明内存带宽和 runtime 开销占主导
