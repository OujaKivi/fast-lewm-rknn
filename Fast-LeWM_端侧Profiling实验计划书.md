# Fast-LeWM 端侧 Profiling 实验计划书

## 1. 背景与目标

### 1.1 背景
Fast-LeWM 通过 action-prefix prediction 把 autoregressive latent rollout 改成直接多步预测，在 RK3588 端侧平台上已实现：
- 纯 CPU：19.09s/决策
- 异构 NPU（CPU enc + NPU INT8 pred）：5.20s/决策，加速 3.67x

但之前的 profiling 都是**基于随机权重的单模块延迟测试**，尚未在**真实任务成功率**下验证端到端性能。

近期已完成 Mac CPU 官方原生流程对齐，成功率达到 **100% (5/5)**，为端到端 profiling 奠定了基础。

### 1.2 目标
通过 3 组对比实验，回答以下核心问题：
1. **稳定成功率**：Mac CPU 上 50 个 episode 的稳定成功率是多少？和论文结果是否对齐？
2. **NPU offload 收益**：把 predictor 推理 offload 到 RK3588 NPU 后，端到端延迟降低多少？成功率是否保持？
3. **INT8 量化影响**：INT8 量化在真实任务上的成功率损失是多少？是否可接受？

最终输出一份完整的端侧 profiling 报告。

---

## 2. 实验设计

### 实验 1：Mac CPU 50 Episode 稳定成功率

**目的**：建立 CPU 基线，验证官方流程的稳定成功率

**配置**：
- 平台：Mac M5 Pro CPU
- 模型：预训练权重（FP32）
- 环境：官方 stable-worldmodel PushT-v1
- CEM：num_samples=300, n_steps=30, topk=30
- episode 数：50（和官方 eval.num_eval 一致）
- goal_offset_steps：25

**指标**：
- 成功率（%）
- 平均成功步数
- 平均 episode reward
- 平均 CEM solve time
- CEM solve time 分布（P50/P95/P99）

### 实验 2：RK3588 NPU Offload 端到端对比

**目的**：验证 NPU offload 在真实任务上的端到端收益

**架构**：
```
Mac (CPU)                          RK3588 (NPU)
┌─────────────────┐               ┌─────────────────┐
│ PushT 环境      │   JSON RPC    │ ViT encoder     │
│ Episode 管理    │ ◄───────────► │ action_encoder  │
│ 可视化          │  (base64 PNG) │ predictor (NPU) │
│ CEM elite 选择  │               │ CEM 采样        │
└─────────────────┘               └─────────────────┘
```

**对比组**：
| 组别 | ViT | action_encoder | predictor | 说明 |
|------|-----|---------------|-----------|------|
| A (Mac CPU) | CPU | CPU | CPU | 实验1基线 |
| B (RK3588 CPU) | CPU | CPU | CPU | 板端CPU基线 |
| C (RK3588 NPU FP16) | CPU | CPU | NPU FP16 | 异构方案A |
| D (RK3588 NPU INT8) | CPU | CPU | NPU INT8 | 异构+量化 |

**指标**：
- 成功率（%）
- 端到端延迟（s/决策）
  - 总延迟
  - 图像传输延迟
  - ViT 编码延迟
  - action_encoder 延迟
  - predictor 延迟
  - CEM 采样/elite 选择延迟
- 各模块占比
- 通信开销占比

### 实验 3：INT8 量化对任务成功率的影响

**目的**：量化 INT8 量化在真实任务上的精度损失

**对比组**：
| 组别 | predictor 精度 | 信噪比 | 单步延迟 |
|------|--------------|--------|---------|
| FP16 | FP16 | 478:1 | 17.4ms |
| INT8 | INT8 | 21.3:1 | 13.1ms |

**指标**：
- 成功率差异（FP16 vs INT8）
- 成功步数差异
- reward 差异
- 失败案例分析（INT8 失败但 FP16 成功的 episode）

---

## 3. 技术方案

### 3.1 RK3588 Planner Server 重写

**问题**：之前的 rk3588_planner_server.py 是自实现的简化 CEM，与官方流程不对齐（action 格式、cost function、标准化等）。

**方案**：重写 server，完全对齐官方流程：
1. 复用官方 `AutoCostModel` 加载模型（ViT + action_encoder + predictor）
2. predictor 用 RKNN NPU 推理（替换 torch forward）
3. CEM 用官方 `CEMSolver`
4. action 标准化用从数据集拟合的 StandardScaler
5. 输入/输出格式与官方 `WorldModelPolicy.get_action` 完全一致

**Server 接口**：
```json
// 请求
{
  "pixels": "base64 PNG",
  "goal": "base64 PNG",
  "state": [agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y],
  "proprio": [...],
  "mode": "cpu" | "npu_fp16" | "npu_int8"
}

// 响应
{
  "action": [action_x, action_y],
  "timings": {
    "vit": 0.035,
    "action_enc": 0.097,
    "predictor": 0.073,
    "cem_total": 1.2,
    "total": 1.5
  }
}
```

### 3.2 Mac↔RK3588 通信优化

**传输内容**：
- pixels/goal：224×224×3 PNG，~2-5KB
- state/proprio：小数组，<100B
- action：2 个 float，<100B

**优化措施**：
1. SSH ControlMaster 复用连接（已配置）
2. stdin/stdout JSON 通信（避免 HTTP 开销）
3. base64 PNG 编码（~33% 膨胀，但简单可靠）
4. 可选：后续可改用共享内存或零拷贝

### 3.3 实验自动化

**脚本**：
- `run_mac_cpu_50ep.py`：Mac CPU 50 episode 实验
- `run_rk3588_offload.py`：RK3588 offload 实验
- `run_int8_comparison.py`：INT8 vs FP16 对比实验
- `aggregate_results.py`：汇总结果，生成 profiling 报告

---

## 4. 时间安排

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| 1 | 写计划书 | 5min |
| 2 | 实验1：Mac CPU 50 episode | 15-20min（50 ep × ~1.5s/step × ~30步） |
| 3 | 重写 RK3588 planner server | 20-30min |
| 4 | 实验2：RK3588 NPU offload | 20-30min |
| 5 | 实验3：INT8 对比 | 15-20min |
| 6 | 汇总结果，生成 profiling 报告 | 15-20min |
| **总计** | | **~2h** |

---

## 5. 预期成果

### 5.1 数据产物
- `mac_cpu_50ep_results.json`：Mac CPU 50 episode 详细结果
- `rk3588_offload_results.json`：RK3588 offload 详细结果
- `int8_comparison_results.json`：INT8 vs FP16 对比结果

### 5.2 代码产物
- `rk3588_planner_server_v2.py`：对齐官方流程的 RK3588 planner server
- `run_mac_cpu_50ep.py`：Mac CPU 50 episode 实验脚本
- `run_rk3588_offload.py`：RK3588 offload 实验脚本
- `run_int8_comparison.py`：INT8 对比实验脚本

### 5.3 最终报告
- `Fast-LeWM_端侧Profiling综合报告.md`：完整 profiling 报告
  - 实验背景与目标
  - 实验配置与方法
  - 三组实验结果（含图表）
  - 瓶颈分析与优化建议
  - 结论与后续工作

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| RK3588 连接不稳定 | 中 | 高 | SSH ControlMaster 复用，失败重试，备用纯板端实验 |
| NPU 推理精度问题导致成功率下降 | 中 | 中 | 先验证单步精度，再跑完整 episode；FP16 作为对照 |
| 通信开销过大掩盖 NPU 收益 | 中 | 中 | 详细 profiling 各阶段时间；必要时优化传输（如共享内存） |
| 50 episode 运行时间过长 | 低 | 低 | 估算 ~15min，可接受；必要时减少到 30 episode |
| RK3588 板端环境依赖问题 | 低 | 中 | 已有 fast-lewm conda 环境，提前验证 |

---

## 7. 关键假设

1. **Mac CPU 基线可靠**：实验1的 50 episode 结果作为基线，假设官方流程正确
2. **NPU 模型精度足够**：predictor FP16 信噪比 478:1，INT8 信噪比 21.3:1，假设对成功率影响可测但不致命
3. **通信延迟稳定**：SSH 局域网通信延迟 <5ms，假设不随时间显著变化
4. **板端锁最高频**：所有 RK3588 实验都锁最高频（CPU 2.35GHz, NPU 1GHz, DMC 2.1GHz）

---

**计划书完成，开始执行实验 1。**
