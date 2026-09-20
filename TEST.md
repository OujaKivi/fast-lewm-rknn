# Fast-LeWM RK3588 端侧优化测试说明

## 一、对齐说明

### 1.1 与源 repo 对齐

**源 repo**: https://github.com/Yuntian-Gao/Fast-LeWorldModel

**正确实现（Fast-LeWM action-prefix prediction）：**
- 一次 `action_encoder`：输入完整 5 个 action blocks
- 一次 `predictor`：一次 forward 预测所有 horizon
- 总计算量：1 次 action_encoder + 1 次 predictor

**之前错误实现（已修正）：**
- 错误地分两步自回归：[2, 3] blocks
- 总计算量：2 次 action_encoder + 2 次 predictor
- 多算了一倍工作量

---

## 二、测试配置

### 2.1 硬件环境
- **板子**: RK3588（16GB RAM）
- **NPU**: 三核 6TOPS INT8，Runtime 2.3.2
- **CPU**: 4×A76 + 4×A55，绑大核 taskset -c 4-7
- **锁频**: 小核 1800MHz，大核 2352MHz，NPU 1GHz

### 2.2 模型配置
- **ViT**: tiny，hidden_size=192, heads=3, layers=12
- **ActionPrefixEmbedder**: input_dim=10, emb_dim=192, transformer_depth=3
- **ARPredictor**: depth=6, mlp_dim=2048, heads=8, dim_head=128
- **Projector/PredProj**: MLP 192→2048→192, BatchNorm1d

### 2.3 CEM 配置
- num_samples = 300
- n_steps = 30
- topk = 30
- horizon = 1
- action_block = 25
- action_num_blocks = 5

---

## 三、测试项目

### 3.1 单模块性能测试

| 模块 | CPU | NPU FP16 | NPU INT8 | 说明 |
|------|-----|----------|----------|------|
| ViT Encoder | 138.43 ms | 58.04 ms | - | NPU 加速 2.39x |
| Action Encoder (5 blocks) | 93.67 ms | - | - | 只在 CPU 上 |
| Predictor + PredProj | ??? | ??? | ??? | 对齐后重新测试 |
| Data Transfer | - | 0.15 ms | 0.15 ms | 可以忽略 |

### 3.2 Batch Size 性能曲线（对齐后正确结果）

![对齐后性能曲线](aligned_batch_performance_curve.png)

**关键数据：**

| Batch Size | CPU (ms) | NPU FP16 (ms) | Speedup (NPU/CPU) |
|------------|----------|---------------|-------------------|
| 1 | 29.80 | 8.46 | **3.52x** ✅ |
| 2 | 30.57 | 6.29 | **4.86x** ✅ |
| 4 | 33.10 | 9.19 | **3.60x** ✅ |
| 8 | 39.96 | 15.30 | **2.61x** ✅ |
| 16 | 51.77 | 27.50 | **1.88x** ✅ |
| 32 | 72.74 | 52.00 | **1.40x** ✅ |
| 64 | 127.06 | 102.39 | **1.24x** ✅ |
| 128 | 217.80 | 198.98 | **1.09x** ✅ |
| **192** | **296.53** | **295.55** | **1.00x** ← 拐点！ |
| 256 | 374.38 | 397.85 | **0.94x** ❌ |
| 300 | 424.15 | 459.65 | **0.92x** ❌ |

**关键发现：**
- ✅ **拐点在 batch=192**
- ✅ **batch ≤ 192：NPU 更快**
- ✅ **batch=300：NPU 只比 CPU 慢 8%**
- ✅ **小 batch 时 NPU 快 2-5 倍**

### 3.3 端到端规划测试

| 模式 | 成功率 | 平均规划时间 | 说明 |
|------|--------|-------------|------|
| CPU 模式 | 60% | 17.92s | 基准 |
| NPU FP16 模式 | ??? | ??? | 对齐后重新测试 |
| 异构模式（ViT NPU + Predictor CPU） | ??? | ??? | 最优组合？ |

---

## 四、关键结论（对齐后正确版本）

### 4.1 已确认结论
- ✅ ViT NPU 加速 2.39x
- ✅ 数据传输开销仅 0.15ms，不是瓶颈
- ✅ **对齐后 NPU 性能其实很好！**
  - 小 batch（≤16）：NPU 快 2-5 倍
  - 中 batch（≤192）：NPU 快 1-2 倍
  - 大 batch（300）：NPU 只慢 8%

### 4.2 待确认结论
- 🔄 端到端规划测试（对齐后重新跑）
- 🔄 INT8 量化后的性能提升
- 🔄 异构模式最优组合

---

## 五、测试脚本

| 脚本 | 位置 | 说明 |
|------|------|------|
| rk3588_planner_server.py | 板端 /root/Fast-LeWorldModel/ | 完整规划 server（已修正） |
| board_eval_v2.py | 板端 /root/Fast-LeWorldModel/ | 端到端评估脚本 |
| latency_breakdown_v2.py | /tmp/ | 单模块 latency breakdown |
| aligned_batch_test.py | /tmp/ | 对齐的 batch size 性能测试 |

---

**更新时间**: 2026-09-20
