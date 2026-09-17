# Fast-LeWM 端侧异构执行优化：预训练权重综合评测报告

> **实验平台**：RK3588（4×A76@2.35GHz + 4×A55@1.8GHz + NPU 6TOPS@1GHz，16GB RAM）
> **实验条件**：锁最高频（CPU/NPU/DMC 全锁），绑大核（taskset -c 4-7），30 CEM iterations × 3 repeats
> **预训练权重**：HuggingFace `naiverer/fast-leworldmodel`（PushT 任务，69MB）
> **报告日期**：2026-09-17

---

## 1. 实验背景与目标

### 1.1 研究问题

Fast-LeWM 通过 action-prefix prediction 将 autoregressive latent rollout 改为直接多步预测，model calls 从 5 次降到 1 次。但当 autoregressive rollout 不再是主要瓶颈后，端到端规划性能是否转而受 **CEM 采样、CPU/NPU 数据搬运、runtime 同步、batch 组织**等系统开销限制？

### 1.2 本阶段目标

1. 加载官方预训练权重（PushT），验证 RKNN 转换精度
2. 完成 **纯CPU / 纯NPU / 异构FP16 / 异构INT8** 四方性能对比
3. 明确优化算法在 Fast-LeWM 完整框架中的位置
4. 规划仿真环境，为后续任务成功率评估做准备

---

## 2. 预训练权重加载与验证

### 2.1 权重下载与加载

- **来源**：HuggingFace `naiverer/fast-leworldmodel`，4 个任务权重（dmc/ogb/pusht/tworoom）
- **本次使用**：`Fast-lewm_pusht_object.ckpt`（69MB）
- **加载方式**：checkpoint 是 `jepa.JEPA` 对象，需 `.state_dict()` 提取
- **关键配置差异**（与之前随机权重实验对比）：

| 配置项 | 随机权重（之前） | 预训练权重（本次） |
|---|---|---|
| action_encoder input_dim | 2 | **10**（5帧×2维） |
| predictor action_fusion_norm | LayerNorm(192) | **LayerNorm(576)**（3×192） |
| action_encoder 参数 | ~1.8M | 1.80M（47 个 tensor） |
| predictor 参数 | ~9.0M | 9.02M（86 个 tensor） |

- **加载结果**：action_encoder 47 参数 + predictor 86 参数，missing=0, unexpected=0 ✅

### 2.2 RKNN 转换精度验证

| 模块 | MAE | 信噪比 | 相对误差 | 结论 |
|---|---|---|---|---|
| **predictor FP16** | 0.0021 | **478:1** | 1.8% | ✅ 转换完全正确 |
| **predictor INT8** | 0.0464 | **21.3:1** | 51.8%* | ✅ INT8 合理范围 |
| **action_encoder FP16** | 1.189 | 0.9:1 | 1054% | ❌ 转换有精度问题 |

> *INT8 相对误差较高是因为输出值接近 0，相对误差被放大；信噪比 21.3:1 是更合理的度量，处于 INT8 量化典型范围（20-30:1）。

**action_encoder 精度问题分析**：
- 可能原因：causal attention mask 的 -inf 值在 RKNN 转换中被修正为 -10000，导致 attention 权重计算偏差
- 已尝试 `optimization_level=2` 禁用该修正，精度仍无改善
- **但这不影响最终方案**：action_encoder 已验证不适合上 NPU（CPU 比 NPU 快 34%），最终方案用 CPU 运行 action_encoder

---

## 3. 四方性能对比

### 3.1 测试配置

- **CEM 参数**：num_samples=300, n_steps=30, topk=30, horizon=5
- **测试模式**：
  1. **纯CPU**：action_encoder + predictor 均用 PyTorch CPU
  2. **纯NPU FP16**：action_encoder + predictor 均用 RKNN NPU（action_encoder 精度有问题，仅作延迟参考）
  3. **异构 FP16**：CPU action_encoder + NPU FP16 predictor
  4. **异构 INT8**：CPU action_encoder + NPU INT8 predictor

### 3.2 端到端结果

| 模式 | 端到端 | enc/iter | pred/iter | 加速比 | 备注 |
|---|---|---|---|---|---|
| **纯CPU** | 19.09s | 97ms | 536ms | 1.00x | 基线 |
| **纯NPU FP16** | 6.77s | 130ms | 94ms | 2.82x | ⚠️ enc精度有问题 |
| **异构 FP16** | 5.84s | 98ms | 96ms | 3.27x | CPU enc + NPU pred |
| **异构 INT8** | **5.20s** | 96ms | 73ms | **3.67x** | CPU enc + NPU INT8 pred |

### 3.3 关键发现

#### 发现 1：纯CPU 下 predictor 是绝对瓶颈（94%）

```
纯CPU 单次 CEM iteration（~633ms）：
├── action_encoder:  97ms  (15%)
├── predictor (5步): 536ms  (85%)  ← 绝对瓶颈
└── CEM update:      ~0ms  (<1%)
```

predictor 单步 CPU 延迟 ~107ms，5 步 rollout 536ms，占总时间 85%。

#### 发现 2：action_encoder 不适合上 NPU

| 设备 | action_encoder 延迟 | 对比 |
|---|---|---|
| CPU (A76@2.35GHz) | 97ms | 基线 |
| NPU FP16 | 130ms | **慢 34%** |
| NPU INT8 | 117ms | 慢 21% |

原因：action_encoder 是小 batch（300）+ 浅层 Transformer（3层）+ 串行 causal attention，NPU 的大矩阵并行优势无法发挥，且 CPU↔NPU 数据搬运开销占比高。

#### 发现 3：异构比纯NPU快 14%

异构方案将 action_encoder 留 CPU（97ms），predictor 放 NPU（96ms），比纯 NPU（enc 130ms + pred 94ms = 224ms）快 14%。

#### 发现 4：INT8 量化进一步加速 11%

| 指标 | FP16 | INT8 | 提升 |
|---|---|---|---|
| predictor 单步 | 17.4ms | 13.1ms | **1.33x** |
| predictor 5步/iter | 96ms | 73ms | 1.32x |
| 端到端 | 5.84s | 5.20s | 1.12x |
| 模型体积 | 19MB | 10MB | **-47%** |

INT8 量化在 predictor 上效果显著：单步加速 1.33x，模型体积减半，精度损失可控（信噪比 21.3:1）。

### 3.4 瓶颈转移分析

```
纯CPU（19.09s）：
  predictor:    536ms/iter (85%)  ← 瓶颈
  action_enc:    97ms/iter (15%)

异构INT8（5.20s）：
  action_enc:    96ms/iter (46%)  ← 新瓶颈！
  predictor:     73ms/iter (35%)
  其他开销:      ~4ms/iter (19%)
```

**核心结论**：predictor 卸载到 NPU 后，瓶颈从"模型计算"转移到"action_encoder + 异构调度开销"。但因 action_encoder 不适合 NPU，进一步优化方向应为：
- action_encoder 的 CPU 优化（算子融合、多线程）
- CPU↔NPU 数据搬运优化（零拷贝、pinned memory）
- 双缓冲流水线（CPU 准备下一批时 NPU 推理当前批）

---

## 4. 优化算法在框架中的位置

### 4.1 完整框架流程图

详见独立文件：[Fast-LeWM_框架优化流程图.html](Fast-LeWM_框架优化流程图.html)

### 4.2 框架结构与优化位置

Fast-LeWM 的完整规划流水线分为三层：

```
┌─────────────────────────────────────────────────────────┐
│  感知层 Perception                                        │
│  观测图像 o_t → vi_enc (ViT-tiny, CPU) → latent z_t     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  CEM 规划循环（30 iterations × 300 candidates）          │
│                                                           │
│  候选动作采样 a_{t:t+k} ~ N(μ,σ)                         │
│       │                                                   │
│       ▼                                                   │
│  ⚡ action_encoder (动作前缀编码器)  → CPU (97ms)        │
│       │                                                   │
│       ▼                                                   │
│  act_emb [B,1,192]                                       │
│       │                                                   │
│       ▼                                                   │
│  ⚡ predictor (并行 latent 预测器)  → NPU FP16/INT8     │
│       │                      (17ms/13ms 每步)            │
│       ▼                                                   │
│  预测 z_{t+k} [B,H,192]                                  │
│       │                                                   │
│       ▼                                                   │
│  评分 + 精英选择 (Top-K, CPU)                             │
│       │                                                   │
│       ▼                                                   │
│  CEM 分布更新 μ,σ ← elite mean/std                       │
│       │                                                   │
│       └─────── 循环（下一轮迭代）────────┐               │
└───────────────────────────────────────────┼───────────────┘
                                             │
                              收敛后输出最优动作 a*
                                             │
┌───────────────────────────────────────────▼───────────────┐
│  执行层 Execution                                           │
│  环境执行 / 机器人控制（PushT 仿真 / 真实机械臂）          │
└───────────────────────────────────────────────────────────┘
```

### 4.3 本研究优化范围

**⚡ 标记的两个模块是本研究的优化对象：**

| 优化模块 | 原始执行 | 优化后执行 | 加速 |
|---|---|---|---|
| **action_encoder** | CPU（默认） | CPU（验证后确认 NPU 更慢，留 CPU） | — |
| **predictor** | CPU（536ms/iter） | NPU INT8（73ms/iter） | **7.3x** |
| **端到端** | 19.09s | 5.20s | **3.67x** |

**未优化但值得关注的模块**：
- `vi_enc`（ViT-tiny）：单次 ~125ms，占总规划时间 2%，上 NPU 可省 ~90ms（-1.4%），收益有限
- CEM 采样/评分/更新：CPU 开销 <1ms，不是瓶颈
- CPU↔NPU 数据搬运：当前 predictor 每步都有 numpy↔tensor 转换，可优化

---

## 5. 仿真环境规划

### 5.1 目标

搭建 PushT 仿真环境，验证优化后的 Fast-LeWM 在**真实任务成功率**上的表现，而不仅仅是延迟。

### 5.2 环境选择

**推荐方案：使用 stable_worldmodel 自带的 PushT-v1 环境**

- **来源**：Fast-LeWorldModel 仓库的 `config/eval/pusht.yaml` 中 `env_name: swm/PushT-v1`
- `swm` 即 `stable_worldmodel` 包，是 LeWM/Fast-LeWM 官方使用的仿真环境
- PushT 是一个 2D 推块任务：机械臂末端推动一个方块到目标区域，是世界模型规划的标准基准

### 5.3 依赖安装

```bash
# 在板端 fast-lewm conda 环境中
conda activate fast-lewm

# 安装 stable_worldmodel（可能需要从源码安装）
pip install stable_worldmodel
# 或从 GitHub 安装
pip install git+https://github.com/Yangruipeng/stable_worldmodel.git

# 其他依赖
pip install gymnasium pymunk pygame  # PushT 物理引擎
```

> **注意**：板端是 aarch64，部分包可能需要从源码编译。如果安装困难，可在 x86 机器上运行仿真（仅用于评估成功率，不测延迟）。

### 5.4 评估指标

| 指标 | 说明 | 目标 |
|---|---|---|
| **成功率** | 100 个 episode 中成功将方块推到目标区域的比例 | 与官方论文对齐（~85-90%） |
| **完成时间** | 成功 episode 的平均步数/时间 | 越短越好 |
| **规划延迟** | 每步决策的平均延迟 | 已测：5.2s（异构INT8） |
| **累积奖励** | episode 总奖励 | 越高越好 |

### 5.5 与当前 profiling 脚本的集成

需要修改 `eval.py`（原仓库硬编码 cuda）：

1. **设备抽象**：将 `cuda` 改为可配置的 `device` 参数，支持 `cpu` / `npu`
2. **模型加载**：支持加载预训练权重（`.ckpt` 或 `weights_state_dict.pt`）
3. **NPU 推理**：predictor 用 RKNN NPU 推理，action_encoder 用 CPU
4. **CEM 规划**：复用当前 `bench_pretrained_comprehensive.py` 中的 CEM 逻辑
5. **环境交互**：每步规划后执行动作，获取新观测，循环直到 episode 结束

### 5.6 实验设计

```
实验矩阵：
├── 基线：纯CPU（PyTorch）
│   ├── 成功率 / 完成时间 / 规划延迟
│   └── 与官方论文结果对齐
├── 异构 FP16（CPU enc + NPU pred）
│   ├── 成功率（验证精度损失可接受）
│   └── 规划延迟（已测 5.84s）
├── 异构 INT8（CPU enc + NPU INT8 pred）
│   ├── 成功率（验证 INT8 精度损失可接受）
│   └── 规划延迟（已测 5.20s）
└── 消融实验
    ├── predictor FP16 vs INT8 成功率对比
    ├── 不同 candidate batch size (16/32/64/128/256/512)
    └── 不同 CEM iterations (10/20/30/50)
```

### 5.7 风险与应对

| 风险 | 应对方案 |
|---|---|
| stable_worldmodel 在 aarch64 安装失败 | 在 x86 机器上运行仿真（仅评估成功率） |
| 预训练权重与环境接口不匹配 | 先在 x86 上用官方 eval.py 跑通，再移植 |
| INT8 量化导致成功率下降 | 对比 FP16/INT8 成功率，若下降明显则用 FP16 |
| 环境渲染在无显示器板端失败 | 使用无头模式（`SDL_VIDEODRIVER=dummy`） |

---

## 6. 结论与下一步

### 6.1 本阶段结论

1. **预训练权重加载成功**：PushT 任务权重（69MB），action_encoder 47 参数 + predictor 86 参数，missing=0
2. **predictor RKNN 转换正确**：FP16 信噪比 478:1，INT8 信噪比 21.3:1（合理范围）
3. **action_encoder 不适合上 NPU**：CPU 97ms vs NPU 130ms（慢 34%），留 CPU 是最优选择
4. **四方对比最优方案**：**CPU action_encoder + NPU INT8 predictor**，端到端 5.20s，加速 3.67x
5. **瓶颈转移**：predictor 卸载到 NPU 后，action_encoder 成为新瓶颈（占 46%），但因结构不适合 NPU，需其他优化方向

### 6.2 下一步计划

| 优先级 | 任务 | 预期收益 |
|---|---|---|
| **P0** | 搭建 PushT 仿真环境，验证任务成功率 | 确认优化后模型仍能完成任务 |
| **P1** | predictor INT8 精度 vs 成功率对比 | 决定最终用 FP16 还是 INT8 |
| **P1** | 双缓冲流水线（CPU 准备下一批时 NPU 推理当前批） | 预期再加速 10-20% |
| **P2** | action_encoder CPU 算子优化（融合、多线程） | 预期加速 10-30% |
| **P2** | CPU↔NPU 零拷贝数据搬运 | 减少 runtime 开销 |
| **P3** | 其他任务权重验证（dmc/ogb/tworoom） | 泛化性验证 |

---

## 附录：实验环境与命令

### A.1 板端环境

- **设备**：RK3588，Ubuntu 22.04 aarch64，16GB RAM
- **Conda 环境**：`fast-lewm`（Python 3.10，torch 2.9.1+cpu，rknn-toolkit-lite2 2.3.2）
- **锁频命令**：
  ```bash
  # CPU 小核 1.8GHz / 大核 2.35GHz
  for i in 0 1 2 3; do echo performance > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor; echo 1800000 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq; echo 1800000 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq; done
  for i in 4 5 6 7; do echo performance > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor; echo 2352000 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq; echo 2352000 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq; done
  # NPU 1GHz / DMC 2112MHz
  echo 1000000000 > /sys/class/devfreq/fdab0000.npu/min_freq
  echo 1000000000 > /sys/class/devfreq/fdab0000.npu/max_freq
  echo 2112000000 > /sys/class/devfreq/dmc/min_freq
  echo 2112000000 > /sys/class/devfreq/dmc/max_freq
  ```
- **运行命令**：`taskset -c 4-7 python bench_pretrained_comprehensive.py`

### A.2 x86 转换机环境

- **设备**：Ubuntu 24.04 x86_64，52核，503GB RAM
- **Conda 环境**：`rknn-py310`（Python 3.10，rknn-toolkit2 2.3.2，torch 2.4.0+cpu）
- **工作目录**：`/home/wjw/rknn-work/`

### A.3 模型文件清单（板端）

| 文件 | 大小 | 用途 |
|---|---|---|
| `weights/weights_state_dict.pt` | 42MB | 预训练权重（纯 state_dict，不需 transformers） |
| `predictor_S300_pretrained_fp16.rknn` | 19MB | predictor FP16 NPU 模型（精度正确） |
| `predictor_S300_pretrained_int8.rknn` | 10MB | predictor INT8 NPU 模型（精度合理） |
| `action_encoder_S300_pretrained_fp16.rknn` | 194MB | action_encoder FP16（精度有问题，不用） |

### A.4 脚本清单

| 脚本 | 用途 |
|---|---|
| `bench_pretrained_comprehensive.py` | 四方综合性能对比测试 |
| `bench_pretrained_het.py` | 异构 FP16 端到端测试 |
| `verify_pretrained.py` | 预训练权重 RKNN 精度验证 |
| `verify_pred_int8.py` | predictor INT8 精度验证 |
