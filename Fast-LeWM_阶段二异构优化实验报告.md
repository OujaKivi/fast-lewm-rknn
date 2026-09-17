# Fast-LeWM 端侧异构执行优化 — 阶段二实验报告

> 实验平台：RK3588（4×A76@2.35GHz + 4×A55@1.8GHz + NPU 三核 6TOPS@1GHz，16GB LPDDR4）
> 实验时间：2026-09-15 ~ 2026-09-17
> 锁频状态：所有实验均锁最高频（CPU performance governor + NPU 1GHz + DMC 2112MHz）

---

## 1. 实验背景

### 1.1 研究问题

Fast-LeWM 通过 action-prefix prediction 将 autoregressive latent rollout 改为直接多步预测，dynamics time 从 31.4s 降到 8.0s。但完整 CEM solve time 仍为 28.3s，说明 dynamics 之外存在较大端到端开销。

核心假设：**Fast-LeWM 消除 autoregressive rollout 后，端侧瓶颈可能从模型计算转移到采样规划与异构数据流。**

### 1.2 模型结构

| 模块 | 参数量 | 输入 | 输出 |
|---|---|---|---|
| ActionPrefixEncoder | ~5.5M | actions[300,5,2] + latent[300,1,192] | act_emb[300,1,192] |
| ParallelLatentPredictor | ~5.3M | latent[300,1,192] + act_emb[300,1,192] | pred[300,1,192] |
| ViT-tiny (vi_enc) | ~5M | image[1,3,224,224] | embedding[1,192] |
| **总计** | **~15.8M** | | |

### 1.3 当前权重状态

**当前所有 NPU 模型均使用随机初始化权重**，未加载 HuggingFace 预训练权重。影响：
- 推理延迟/加速比数据可信（不依赖权重值）
- 量化精度测试参考价值有限（随机权重分布与预训练不同）
- 任务成功率无法评估

---

## 2. 方案 A：异构调度（CPU enc + NPU pred）

### 2.1 方案设计

将 CEM 规划中的两个核心模块映射到不同后端：
- **ActionPrefixEncoder → CPU（4×A76）**：已验证 NPU 上比 CPU 慢 14%（FP16）/ 2%（INT8），不适合上 NPU
- **ParallelLatentPredictor → NPU**：大 batch 矩阵计算，适合 NPU 并行

### 2.2 三方对比（锁频后，30 iter × 3 rep，batch=300，FP16）

| 模式 | 总时间 | enc | pred | 加速比（vs CPU） |
|---|---|---|---|---|
| CPU 全量 | 19.29s | 2.94s | 16.0s | 1.00x |
| NPU 全量 | 8.58s | 5.31s | 3.14s | 2.25x |
| **NPU 异构（方案A）** | **6.27s** | **2.94s** | **3.14s** | **3.08x** |

**关键发现**：
- 异构比 NPU 全量快 37%（1.37x），因为 action_encoder 留 CPU 避免了 NPU 上的低效执行
- action_encoder 在 NPU 上比 CPU 慢 76%（5.31s vs 2.94s），确认不适合上 NPU
- predictor 在 NPU 上单步仅 20.9ms，5 步 rollout 约 105ms/iter

### 2.3 单 iteration 分解

| 阶段 | 时间 | 占比 |
|---|---|---|
| CPU action_encoder | 111ms | 50.2% |
| NPU predictor (5步) | 110ms | 49.8% |
| **合计** | **221ms** | **100%** |

enc 和 pred 时间几乎相等，**理论完美并行可从 221ms 降到 111ms，加速 ~2x**。

---

## 3. 方案 B：双缓冲流水线

### 3.1 方案设计

同一 iteration 内将 300 candidate 分两组（各 150）：
1. CPU 编码 A 组（78ms）
2. 启动线程 NPU 推理 A 组，同时主线程 CPU 编码 B 组（64ms，与 NPU 并行）
3. 等待 NPU A 完成，NPU 推理 B 组（58ms）

需要 batch=150 的 predictor RKNN 模型（已在 x86 上重新转换）。

### 3.2 实验结果（30 iter × 3 rep）

| 指标 | 方案 A（串行 S300） | 方案 B（双缓冲 S150×2） |
|---|---|---|
| **端到端** | **6.79s** | **6.35s** |
| enc（CPU编码） | 112.6ms/iter | enc0=78.1ms + enc_overlap=63.9ms |
| pred（NPU推理） | 111.1ms/iter（5步×300） | pred0=71.8ms + pred1=58.0ms（5步×150×2） |
| **加速比** | — | **1.07x（+6.9%）** |

### 3.3 收益有限的原因分析

1. **小 batch NPU 效率低**：batch=150 单步 ~14.4ms，batch=300 单步 ~22.2ms。batch 减半但时间只减 35%，NPU 在小 batch 下吃不满算力。

2. **分组 overhead**：
   - enc 分组后总时间从 113ms 增到 142ms（+26%）
   - pred 分组后总时间从 111ms 增到 130ms（+17%）

3. **并行度不够**：enc_overlap=64ms < pred0=72ms，CPU 编码第二组先完成，NPU 第一组还在跑，CPU 有 ~8ms 空等。

### 3.4 结论

方案 B 双缓冲在当前实现下只有 **7%** 的收益，主要瓶颈是**小 batch NPU 效率低**和**分组 overhead**。该方向收益不如 predictor INT8 量化。

---

## 4. Predictor INT8 量化

### 4.1 转换配置

- 平台：x86 Ubuntu + rknn-toolkit2 2.3.2
- 校准数据：20 组随机数据（batch=300，latent + act_emb）
- 量化类型：asymmetric_quantized-8
- 模型体积：FP16 18.95MB → INT8 10.30MB（-47%）

### 4.2 量化精度（随机权重，10 组测试）

| 指标 | 值 |
|---|---|
| 平均 MAE | 0.0333 |
| 最大误差 | 0.192 |
| FP16 输出 std | 1.0 |
| 信噪比（std/MAE） | ~30:1 |

> 注意：当前为随机权重，预训练权重的量化误差可能不同。加载预训练权重后需重新评估。

### 4.3 单步推理延迟（batch=300，warmup=5，repeats=20）

| 精度 | mean | p99 | 加速比 |
|---|---|---|---|
| FP16 | 17.46ms | 17.62ms | — |
| **INT8** | **13.14ms** | **13.34ms** | **1.33x** |

### 4.4 端到端 CEM 规划（30 iter × 3 rep）

| 精度 | 总时间 | enc(CPU) | pred(NPU) | 端到端加速 |
|---|---|---|---|---|
| FP16 | 6.53s | 114ms | 101.5ms | — |
| **INT8** | **5.84s** | 114ms | **78.7ms** | **1.12x** |

### 4.5 结论

- Predictor INT8 量化**单步加速 1.33x**，模型体积减小 47%
- 端到端加速 **1.12x**，受限于 action_encoder（CPU 瓶颈，占 59%）
- 量化损失可控（MAE=0.033，信噪比 ~30:1），但需预训练权重验证

---

## 5. 综合对比与瓶颈分析

### 5.1 各方案端到端对比

| 方案 | 总时间 | 加速比（vs CPU全量） | 关键优化 |
|---|---|---|---|
| CPU 全量 | 19.29s | 1.00x | 基线 |
| NPU 全量 | 8.58s | 2.25x | predictor + encoder 上 NPU |
| 方案 A（异构） | 6.27s | 3.08x | encoder 留 CPU，predictor 上 NPU |
| 方案 A + INT8 | **5.84s** | **3.30x** | predictor INT8 量化 |
| 方案 B（双缓冲） | 6.35s | 3.04x | 分组并行（收益有限） |

### 5.2 当前瓶颈分解（方案 A + INT8，5.84s）

| 模块 | 时间/iter | 占比 | 后端 | 优化空间 |
|---|---|---|---|---|
| action_encoder | 114ms | 59.1% | CPU | 已验证不适合 NPU；可尝试 torch.compile / ONNX Runtime |
| predictor (5步) | 78.7ms | 40.9% | NPU INT8 | 已量化；可尝试混合精度 / 算子融合 |
| 其他（采样/评分/更新） | <1ms | <0.1% | CPU | 可忽略 |
| **合计** | **193ms** | **100%** | | |

**瓶颈已从模型计算转移到 action_encoder（CPU）**，占总时间 59%。

### 5.3 理论上限估算

如果 action_encoder 能完美优化到与 predictor 相同的效率（78.7ms），端到端可达：
- 方案 A + INT8：78.7 + 78.7 = 157.4ms/iter → 4.72s（当前 5.84s 的 0.81x）
- 方案 A + INT8 + 完美双缓冲：max(78.7, 78.7) = 78.7ms/iter → 2.36s（当前 5.84s 的 0.40x）

但 action_encoder 已验证不适合 NPU，CPU 优化空间有限（torch.compile 可能 1.2-1.5x）。

---

## 6. 已验证做不通的方向

| 方向 | 结论 | 原因 |
|---|---|---|
| action_encoder 上 NPU（FP16） | 比 CPU 慢 14% | NPU 小 batch + 控制流开销 |
| action_encoder 上 NPU（INT8） | 比 CPU 慢 2% | INT8 比 FP16 快 12%，但仍未超 CPU |
| vi_enc 上 NPU | NPU 加速 3.98x，但端到端仅省 1.4% | vi_enc 单次仅占总时间 2% |
| 方案 B 双缓冲（2组） | 仅加速 7% | 小 batch NPU 效率低 + 分组 overhead |
| predictor 静态 batch=300 用 batch=150 推理 | 不可行 | RKNN 静态 shape 不支持动态 batch |

---

## 7. 下一步建议

### 7.1 高优先级

1. **加载预训练权重**：当前所有模型用随机权重，需加载 HuggingFace 预训练权重后重新：
   - 转换 RKNN（FP16 + INT8）
   - 评估量化精度（用真实数据校准）
   - 评估任务成功率

2. **action_encoder CPU 优化**：当前瓶颈（59%），可尝试：
   - torch.compile（inductor backend）
   - ONNX Runtime + CPU EP
   - 算子融合（Linear + GELU + LayerNorm）
   - 预期收益：1.2-1.5x

### 7.2 中优先级

3. **Predictor 混合精度**：对精度敏感的层用 FP16，其余用 INT8，可能在精度损失可控的前提下进一步加速

4. **动态 candidate batch**：CEM 收敛后减少 candidate 数量（256→128→64），减少无效计算

5. **方案 B 优化**：尝试 3 组/4 组分组，或优化线程同步减少 overhead

### 7.3 低优先级

6. **vi_enc NPU 流水线**：vi_enc 单次仅占 2%，但可以和 predictor 并行（视觉编码与动力学预测无数据依赖）

---

## 8. 附录

### 8.1 实验环境

- **RK3588 板子**：Ubuntu 22.04 / aarch64 / 16GB
- **conda 环境**：fast-lewm（torch 2.9.1+cpu, rknn-toolkit-lite2 2.3.2）
- **x86 转换机**：Ubuntu 24.04 / 52核 / 503GB，conda 环境 rknn-py310（rknn-toolkit2 2.3.2）
- **锁频命令**：CPU performance governor + 大核 2352MHz / 小核 1800MHz + NPU 1GHz + DMC 2112MHz

### 8.2 NPU 模型文件

| 模型 | 文件 | 大小 | batch | 精度 |
|---|---|---|---|---|
| predictor | predictor_S300_fp16_v4.rknn | 19MB | 300 | FP16 |
| predictor | predictor_S300_int8.rknn | 10MB | 300 | INT8 |
| predictor | predictor_S150_fp16.rknn | 19MB | 150 | FP16 |
| action_encoder | action_encoder_S300_fp16.rknn | 190MB | 300 | FP16 |
| action_encoder | action_encoder_S300_int8.rknn | 220MB | 300 | INT8 |
| vi_enc | vit_tiny_b2_fp16.rknn | 14MB | 2 | FP16 |

### 8.3 关键脚本

- `npu_predictor.py`：NPUPredictor + NPUActionEncoder 包装类
- `profile_heterogeneous.py`：五种模式 profiling（cpu/npu/npu_full/both/all）
- `profile_het_v2.py`：方案 A vs 方案 B 双缓冲对比
- `bench_pred_int8.py`：predictor FP16 vs INT8 对比（精度+延迟+端到端）
- `bench_action_encoder.py`：action_encoder 三方对比（CPU/NPU FP16/NPU INT8）
- `bench_vi_enc.py`：vi_enc CPU vs NPU 对比
- `diagnose_root_cause.py`：DVFS 根因诊断脚本

---

*报告生成时间：2026-09-17*
