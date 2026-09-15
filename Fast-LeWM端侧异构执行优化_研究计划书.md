# 面向 Fast-LeWM 的端侧异构执行优化

## 研究计划书（Proposal）

- 方向：端侧世界模型 / 模型预测控制 / 异构计算
- 平台基线：Rockchip RK3588（4×A76 + 4×A55 + 三核 NPU 6 TOPS INT8，16 GB）
- 版本：v0.1（初稿，2026-09-15）
- 一句话任务定义：*Profile and characterize Fast-LeWM on an edge heterogeneous platform, with emphasis on whether eliminating autoregressive latent rollout shifts the end-to-end planning bottleneck from world-model computation to CEM sampling, runtime synchronization, and CPU–accelerator data movement. Based on the profiling results, identify opportunities for heterogeneous execution, candidate-aware batching, and planning-compute co-optimization.*

---

## 0. 摘要

Fast-LeWM 用 action-prefix 直接多步预测替代了 LeWM 的自回归 latent rollout，把 dynamics 调用从 5 次降到 1 次、dynamics time 从 31.4 s 降到 8.0 s。但论文中完整 CEM solve time 仍高达 28.3 s，说明 dynamics 之外存在大量未被细分的端到端开销。本研究提出核心假设：**当 autoregressive rollout 不再是主要瓶颈后，端侧规划性能会转而受 CEM 采样、CPU/NPU 数据搬运、runtime 同步与 batch 组织等系统开销限制**。研究分两阶段：第一阶段只做 profiling 与可行性研究，在 RK3588 上对 LeWM / Fast-LeWM 做端到端时延分解、多后端（CPU/GPU/NPU）单模块性能、candidate batch 扫描、transfer/runtime/sync 微基准与能耗测量，验证瓶颈是否发生转移；第二阶段基于 profiling 结论探索 Fast-LeWM-aware 异构映射、candidate-aware batching、planning-compute 协同优化与 selective horizon 早筛。预期产出一份可复现的端侧 profiling 数据集与一篇面向系统/机器人交叉方向的论文。

---

## 1. 研究背景与问题

### 1.1 从 LeWM 到 Fast-LeWM 的算法演进

LeWM（Latent World Model）在规划时对每个候选动作序列做逐步自回归 latent rollout：

$$z_t \rightarrow \hat z_{t+1} \rightarrow \hat z_{t+2} \rightarrow \cdots$$

每一步都要调用一次 dynamics 网络，多次 sequential invocation 既慢又难以并行。Fast-LeWM 引入 action-prefix prediction，把当前状态与完整候选动作前缀一次性输入，直接预测多步 horizon：

$$(z_t,\, a_{t:t+k-1}) \rightarrow \hat z_{t+k}$$

从而把 dynamics 调用从 5 次降到 1 次。论文报告 dynamics time 从 31.4 s 降到 8.0 s，模型结构（action-prefix encoder 一次处理完整序列、parallel latent predictor 一次 forward 多 horizon）天然更偏好大 batch、静态计算图与高并行矩阵乘——这正是 NPU 类加速器擅长的负载形态。

### 1.2 端侧部署引出的新问题

算法层面的 FLOPs 下降并不等比例转化为端到端规划延迟。论文中 Fast-LeWM 的 dynamics time 仅 8.0 s，但完整 CEM solve time 仍为 28.3 s，剩余约 20 s 开销未被细分。在端侧异构平台（CPU + NPU，弱内存带宽、无独立大显存、host-device 拷贝昂贵）上，这一矛盾会被放大：

- CEM 每轮要生成数百候选、做 tensor 准备与 layout 转换、再批量送进加速器；
- NPU 偏好静态 shape，而 CEM 的 candidate 数、elite 选择、迭代收敛都是动态的；
- 多步预测输出需要回传 CPU 做 scoring 与 elite selection，频繁同步会抵消算力收益；
- 小 batch 吃不满 NPU，大 batch 又受内存/带宽限制，存在最佳运行区间。

### 1.3 已有预实验基础（RK3588 实测，作为 motivation）

本课题已在 RK3588 上完成 Fast-LeWM 的部署与初步效率评测（FP32，aarch64 CPU torch 2.9.1，CEM 配置 num_samples=300、n_steps=30、5 步 rollout），关键事实如下：

| 指标 | 实测值 |
|---|---|
| 世界模型参数量 | 10.83 M（动作编码器 1.80 M + 预测器 9.02 M；FP32 ≈ 41 MB） |
| 视觉编码器（ViT-tiny 近似） | 5.52 M |
| 5 步 rollout（300 候选）中位时延 | 4×A76 大核 732.7 ms / 全 8 核 1091.8 ms / 单 A76 2177.6 ms |
| 单次完整 CEM 决策估算 | 4 大核 22.08 s / 全 8 核 32.96 s / 单 A76 65.56 s |
| 相对 10/20 Hz 实时预算 | 慢约 221 / 442 倍（最优配置） |
| 单次决策内构成 | rollout 占 99.4%，ViT 仅 0.6%；rollout 内预测器 54.5% / 动作编码器 45.1% |
| NPU 软硬件 | 三核 6 TOPS INT8，RKNPU driver v0.9.8，RKNN Runtime 2.3.2，`/root/rknn-llm` 齐全 |

**反直觉发现**：仅绑 4 颗 A76 大核比全 8 核（混入慢 A55）还快 1.49×，说明异构调度本身已在影响端到端性能。

**对本课题的启示**：当前 CPU 实现下模型计算（rollout）仍占绝对主导，但这是因为尚未引入加速器。一旦把占比最大的预测器（无 self-attention、以大 GEMM 为主，最适合固定 shape 量化上 NPU）搬到 NPU，模型计算时延被压下来后，CEM 采样、数据搬运、runtime 同步的占比必然凸显——这正是本研究要量化验证的"瓶颈转移"。预实验同时确认：只把 ViT 搬上 NPU 几乎无效（仅占 0.6%），异构拆分必须针对预测器与 CEM 边界设计。

---

## 2. 研究目标

### 2.1 总目标

在端侧异构平台上系统刻画 Fast-LeWM 的执行特征，回答"消除自回归 rollout 后，端到端规划瓶颈是否从模型计算转移到采样规划与异构数据流"，并据此提出面向 Fast-LeWM workload 特性的异构执行优化方法。

### 2.2 具体目标

1. **G1（刻画）**：建立可复现的端到端 profiling 方法，把一次完整 planning 分解为 CEM 采样、prefix 编码、predictor 前向、scoring、数据搬运、runtime 同步等可独立测量的阶段，并在 LeWM 与 Fast-LeWM 之间做对照。
2. **G2（对比）**：在 CPU / GPU / NPU 三种后端上测量单模块 latency、throughput、利用率、能耗与 runtime 开销，判断 Fast-LeWM 是否比 LeWM 获得更高的 NPU 硬件利用率。
3. **G3（扫描）**：扫描 candidate batch size（16–512），寻找 latency/candidate、throughput、内存、能耗与 NPU 利用率的最佳运行区间。
4. **G4（定位）**：量化 CPU↔NPU 数据交互、layout 转换、RKNN runtime 调用、同步与输出回传的开销，判断异构 runtime 是否成为主要瓶颈。
5. **G5（优化，第二阶段）**：基于 profiling 结论，探索并原型验证至少一种 Fast-LeWM-aware 异构执行优化（异构映射 / candidate-aware batching / planning-compute 协同 / selective horizon 早筛）。

---

## 3. 核心假设与研究问题

### 3.1 核心假设

$$\boxed{\text{Fast-LeWM 消除 autoregressive rollout 后，端侧瓶颈可能从模型计算转移到采样规划与异构数据流。}}$$

### 3.2 研究问题

**H1：瓶颈是否发生转移？**
对一次完整 planning 做分解：

$$T_{\text{total}} = T_{\text{CEM}} + T_{\text{prefix}} + T_{\text{predictor}} + T_{\text{score}} + T_{\text{transfer}} + T_{\text{sync}}$$

比较 LeWM 与 Fast-LeWM。目标不是只证明 $T_{\text{Fast-LeWM}} < T_{\text{LeWM}}$，而是观察 Fast-LeWM 后各模块占总 latency 的比例是否发生明显变化。预期得到类似：

| | Dynamics（模型计算） | Planner + Runtime（采样/搬运/同步） |
|---|---:|---:|
| LeWM | ~65% | ~35% |
| Fast-LeWM | ~25% | ~75% |

若出现这种明显的 bottleneck shift，即支持核心假设。

**H2：Fast-LeWM 是否比 LeWM 更适合 NPU？**
比较 CPU / GPU / NPU 上的 latency、throughput、utilization、energy、runtime overhead，尤其关注：

$$\text{speedup}_{\text{NPU}}^{\text{LeWM}} \quad \text{vs.} \quad \text{speedup}_{\text{NPU}}^{\text{Fast-LeWM}}$$

若 Fast-LeWM 在 NPU 上获得显著更高的硬件利用率，则说明其优势不只是算法 FLOPs 更少，计算结构本身也更适合端侧并行加速器。

**H3：candidate batch 是否存在最佳运行区间？**
扫描 $N \in \{16, 32, 64, 128, 256, 512\}$，观察 latency/candidate、throughput、内存、energy/candidate、NPU utilization。验证"小 batch 吃不满 NPU，大 batch 受内存/带宽限制"的假设，定位甜点区间。

**H4：异构 runtime 是否成为主要瓶颈？**
重点测量 candidate tensor preparation、CPU↔NPU 数据交互、layout conversion、RKNN runtime invocation、synchronization、output retrieval、CPU scoring / elite selection。若

$$T_{\text{runtime+transfer}} \approx T_{\text{model}} \quad \text{甚至更高}$$

则说明进一步优化重点不应只是网络算子，而应转向 runtime 与数据流。

---

## 4. 技术路线与方法

### 4.1 统一 profiling 框架

自研轻量 profiling 包装层（在已有 `bench_fast_lewm.py` 基础上扩展），对 CEM 循环的每个阶段插入高精度计时（`time.perf_counter_ns`，NPU 侧用 RKNN runtime 的输入/输出回调与 `set_core_mask` 前后打点），统一输出 JSON。阶段定义：

| 阶段 | 含义 | 测量位置 |
|---|---|---|
| `cem_sample` | 候选动作采样、加噪、elite 复用 | CEM 迭代开头 |
| `tensor_prep` | candidate reshape / concat / layout 转换 / 归一化 | 送 NPU 前 |
| `transfer_h2d` | CPU→NPU 输入拷贝 | RKNN inputs set |
| `prefix_enc` | action-prefix encoder 前向 | 模型内 |
| `predictor` | parallel latent predictor 前向 | 模型内 |
| `transfer_d2h` | NPU→CPU 输出回传 | RKNN outputs get |
| `score` | latent / cost 计算与 elite selection | CPU |
| `sync` | 显式/隐式同步等待 | runtime 调用边界 |
| `vi_enc` | 视觉编码（当前帧+目标） | 每决策开头 |

所有计时重复 ≥ 15 次取中位数，并记录 RSS、NPU 频率（`/sys/class/devfreq/fdab0000.npu`）与整机功耗（如可用）。

### 4.2 三后端对照

- **CPU**：aarch64 torch 2.9.1+cpu，分别测全 8 核 / 4×A76 大核 / 单大核（已有基线）。
- **GPU**：Mali-G610 若有可用 ONNX Runtime / MNN OpenCL backend 则测；若工具链不成熟，用 x86 NVIDIA GPU（如 RTX 系列）作为"理想加速器"上界对照，用于分离"算法结构适配性"与"特定硬件工具链限制"。
- **NPU**：RKNN Toolkit2（x86 转换机）转 INT8/FP16，板端 RKNN Runtime 2.3.2 推理；预测器固定 shape（S=300 或分档），动作编码器 attention 序列固定 T=5，LayerNorm 不满足时改写或划 CPU 子图。

### 4.3 LeWM vs Fast-LeWM 对照

在同一任务（PushT，LeWM 标准基准）、同一视觉编码器、同一 CEM 超参（num_samples、n_steps、topk、horizon）下，分别跑 LeWM（自回归 rollout）与 Fast-LeWM（action-prefix 多步预测），保证唯一变量是 dynamics 调用范式。若 LeWM 官方代码复现成本高，可在 Fast-LeWM 仓库内用"逐步调用 predictor 模拟自回归"作为受控对照，明确标注为近似。

### 4.4 第二阶段优化方向（基于阶段一结论触发）

- **O1 Fast-LeWM-aware 异构映射**：CPU 跑 CEM/控制逻辑，NPU 跑 parallel latent predictor，action-prefix encoder 按 profiling 结果选 CPU/GPU/NPU；通过 profiling 自动寻优 partition。
- **O2 Candidate-aware batching**：batch fusion、dynamic batch sizing、chunked inference、double buffering——CPU prepare batch i+1 与 NPU evaluate batch i 流水并行，降低 host 与 accelerator 空等。
- **O3 Planning-compute 协同优化**：$N_{\text{candidate}} = f(\text{CEM convergence}, \text{uncertainty})$，搜索分布收敛时从 256→128→64 动态缩减，而非固定预算。
- **O4 Selective horizon / early filtering**：利用 Fast-LeWM 多 horizon 输出，300 候选先用短 horizon 粗筛到 100，再对 30 个做完整 horizon 预测，比普通 CEM pruning 更具 workload 特性。

---

## 5. 实验设计

### 阶段一：Profiling & Feasibility（不做复杂优化，约 4–6 周）

| 编号 | 实验 | 内容 | 产出 |
|---|---|---|---|
| E1 | 端到端 latency 分解 | LeWM / Fast-LeWM 各跑完整 CEM，按 4.1 阶段计时，输出占比饼图与瓶颈转移证据 | H1 结论 |
| E2 | 单模块多后端性能 | prefix encoder / predictor / ViT 分别在 CPU、GPU、NPU 上测 latency、throughput、utilization、energy | H2 结论 |
| E3 | candidate batch 扫描 | N=16,32,64,128,256,512，测 latency/candidate、throughput、内存、energy、NPU util | H3 结论 + 甜点区间 |
| E4 | transfer/runtime/sync 微基准 | 隔离测量 h2d/d2h 拷贝、layout 转换、RKNN invoke、同步、输出回传，与纯模型计算对比 | H4 结论 |
| E5 | 能耗与任务成功率保持 | 记录 NPU 频率/功耗/能耗/candidate；确认 profiling 改动不降低 PushT success rate | 约束验证 |

**阶段一交付门槛**：必须得到 LeWM 与 Fast-LeWM 的可对比 latency 分解表，并明确回答"瓶颈是否转移、转移到哪个阶段"。若数据不支持瓶颈转移（例如 NPU 上模型计算仍占 >80%），则如实记录并把研究重点收敛到"如何进一步压缩模型计算 + 降低 CEM 规模"，不强行进入优化阶段。

### 阶段二：异构执行优化（基于阶段一结论，约 6–8 周）

| 编号 | 优化 | 触发条件 | 评估 |
|---|---|---|---|
| O1 | 异构映射原型 | 阶段一显示 predictor 占模型计算主导且 NPU 友好 | 端到端 speedup、NPU util、与全 CPU/全 NPU 对比 |
| O2 | candidate-aware batching + double buffer | 阶段一显示 transfer/sync 或 host 空等显著 | pipeline 利用率、h2d 重叠率、端到端 speedup |
| O3 | 动态 candidate 预算 | 阶段一显示大 batch 边际收益递减 | 相同 success rate 下的 compute 节省、平均 latency |
| O4 | selective horizon 早筛 | 阶段一显示长 horizon 预测昂贵且短 horizon 已能区分候选 | 早筛精度、最终 success rate、compute 节省 |

---

## 6. 平台与工具

### 6.1 硬件

- **主平台**：RK3588（4×Cortex-A76 @2.35 GHz + 4×A55 @1.8 GHz，Mali-G610 MP4，三核 NPU 6 TOPS INT8，16 GB RAM，Ubuntu 22.04 / 内核 6.1.75）——已就绪，Fast-LeWM 已部署。
- **转换机**：x86 Ubuntu（用于 RKNN Toolkit2 模型转换与量化，需校准集）。
- **可选对照**：Jetson Orin Nano（GPU+DLA，另一类端侧异构）或 x86 NVIDIA GPU（理想加速器上界）。

### 6.2 软件

- 模型：Fast-LeWM（PyTorch，已在板端 conda 环境 `fast-lewm` 跑通）；LeWM 对照用同仓库或官方 `lucas-maes/le-wm`。
- 推理后端：PyTorch CPU、ONNX Runtime / MNN（GPU 若可用）、RKNN Toolkit2 + Runtime 2.3.2（NPU）。
- 任务环境：PushT（LeWM 标准，MuJoCo 无头渲染用 OSMesa）；可选 RoboSuite / LIBERO 扩展。
- 测量：自研 `bench_fast_lewm.py` 扩展、`taskset` 绑核、NPU devfreq 读取、`/usr/bin/time -v` 内存、功耗计（如硬件可用）或 RKNN runtime 能耗估算。
- 数据集与权重：HF `naiverer/fast-leworldmodel`、LeWM `.h5` 数据集、校准集从训练集采样。

### 6.3 已有代码资产

- 板端 `/root/Fast-LeWorldModel/`：源码 + `bench_fast_lewm.py` + `run_bench.sh` + 三组正式跑分 JSON。
- 本地：报告、4 图 ECharts HTML、覆写片段等。阶段一可直接在此基础上扩展，无需从零搭建。

---

## 7. 评估指标

| 维度 | 指标 |
|---|---|
| 时延 | 端到端 $T_{\text{total}}$、各阶段分解与占比、单次 CEM 决策时间、相对 10/20 Hz 预算的倍数 |
| 吞吐 | candidates/s、rollouts/s、decisions/s |
| 效率 | latency/candidate、energy/candidate、FLOPs 利用率 |
| 硬件 | NPU 利用率（runtime 占用率 / devfreq 时间）、CPU 各核占用、内存峰值、NPU↔CPU 带宽 |
| 精度 | INT8/FP16 量化前后多步 rollout 误差、CEM 代价排序一致性、PushT success rate |
| 可复现性 | 所有配置、权重、计时脚本、原始 JSON 入库；绑核、NPU 频率、torch/RKNN 版本固定 |

---

## 8. 预期产出与里程碑

### 8.1 里程碑

| 时间 | 里程碑 | 交付物 |
|---|---|---|
| W1–W2 | 环境与基线对齐 | LeWM/Fast-LeWM 在 PushT 上跑通完整 eval；profiling 框架 v1 |
| W3–W4 | E1+E2 端到端与单模块 profiling | latency 分解报告；CPU/GPU/NPU 对比表 |
| W5–W6 | E3+E4+E5 batch 扫描与 runtime 微基准 | 甜点区间结论；runtime/transfer 占比；能耗数据 |
| W6 末 | **阶段一评审**：瓶颈是否转移？是否进入优化？ | 阶段一技术报告 + go/no-go 决策 |
| W7–W10 | O1/O2 异构映射与 batching 原型 | 优化原型代码 + 端到端 speedup |
| W11–W12 | O3/O4 协同优化与早筛 | 消融实验；success rate 保持验证 |
| W13–W14 | 论文撰写与复现整理 | 论文初稿 + 开源代码 + 数据集 |

### 8.2 最终产出

1. 一份可复现的端侧异构 profiling 数据集（原始 JSON + 脚本 + 配置）。
2. 一篇论文（目标系统/机器人交叉方向会议或期刊，如 MLSys、MobiSys、ICRA、CoRL 或相应 workshop）。
3. 一个 Fast-LeWM 端侧异构执行优化原型（开源）。
4. 一份面向工程落地的 RK3588 部署与调优指南。

---

## 9. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| RKNN 算子不支持（LayerNorm 部分支持、self-attention 运行时矩阵乘） | predictor/encoder 无法整图上 NPU | 异构拆分：不支持算子划 CPU 子图或改写（SiLU→ReLU、LayerNorm 布局对齐）；attention 序列固定 T=5 易固化 |
| INT8 量化在多步 rollout 中误差累积，扰乱 CEM 代价排序 | NPU 结果不可用 | 用真实数据做量化感知校准；FP16/混合精度兜底；以 success rate 为硬约束 |
| LeWM 官方代码复现成本高、与 Fast-LeWM 环境不一致 | H1 对照不可比 | 优先在同一仓库内用受控方式模拟自回归；或固定同一视觉编码器+CEM 超参，明确标注近似口径 |
| NPU 利用率/能耗测量工具链不完善 | H2/H3 指标缺失 | 用 runtime 计时 + devfreq 频率 + 内存带宽计数器三角验证；必要时外接功耗计 |
| 阶段一数据不支持瓶颈转移 | 研究方向需调整 | 预设 go/no-go 门槛：如实报告，收敛到"模型计算压缩 + CEM 规模缩减"方向，不强行做优化 |
| 端侧实时性目标过远（当前慢 200×+） | 优化后仍不实时 | 不把"达到实时"作为硬指标；以"相对基线 speedup + 瓶颈定位 + 可扩展方法"为贡献 |
| MuJoCo/数据集/权重在 aarch64 上的依赖链复杂 | 完整 eval 跑不通 | 阶段一可先用"模型计算核 + 合成 CEM 循环"profiling（已有基线），完整任务闭环作为并行子任务 |

---

## 10. 创新点小结

1. **问题视角创新**：首次系统研究"世界模型算法范式从自回归转向 action-prefix 后，端侧异构系统瓶颈的转移现象"，把算法演进与系统执行特征关联起来。
2. **方法创新**：提出 Fast-LeWM-aware 异构执行（按模块特性而非整模型映射 backend）、candidate-aware batching（利用 CEM workload 特殊性做流水与双缓冲）、planning-compute 协同（按收敛度动态调 candidate 预算）、selective horizon 早筛（用多 horizon 输出做两阶段过滤）。
3. **数据与工具创新**：产出一套可复现的端侧世界模型 profiling 方法与数据集，填补"世界模型在真实端侧 NPU 上执行特征"的测量空白。
4. **工程价值**：为 RK3588 这类低成本边缘 SoC 上部署世界模型规划提供可落地的调优指南与异构原型。

---

## 附录 A：第一阶段最小可执行清单（给 Agent 的直接指令）

> 第一阶段**只做 profiling，不实现任何复杂优化**。按以下顺序执行，每步输出原始 JSON 与简短结论：

1. **环境对齐**：在 RK3588 上把 Fast-LeWM 完整 eval（PushT + MuJoCo OSMesa + HF 权重）跑通；若依赖链受阻，沿用已有的"模型计算核 + 合成 CEM"基线，并明确标注口径。
2. **E1 端到端分解**：扩展 `bench_fast_lewm.py`，按 4.1 阶段插入计时，跑 Fast-LeWM 完整 CEM ≥ 15 次；用受控方式模拟 LeWM 自回归 rollout 做对照。输出各阶段占比。
3. **E2 单模块多后端**：prefix encoder / predictor / ViT 分别在 CPU（4 大核）、NPU（RKNN INT8/FP16）上测 latency/throughput；GPU 若工具链可用则补，否则用 x86 GPU 上界。
4. **E3 batch 扫描**：N=16,32,64,128,256,512，测 predictor rollout 的 latency/candidate、throughput、RSS、NPU 频率。
5. **E4 runtime 微基准**：隔离测 h2d/d2h 拷贝（不同 tensor 大小）、layout 转换、RKNN invoke、同步、输出回传，与纯模型计算对比。
6. **E5 精度与能耗**：INT8 量化前后多步 rollout 误差、CEM 代价排序一致性；记录 NPU 频率与能耗估算。
7. **阶段一报告**：回答 H1–H4，给出瓶颈转移证据（或否定证据），输出 go/no-go 建议与第二阶段优化优先级。

**成功判据**：得到一张 LeWM vs Fast-LeWM 的 latency 分解对比表（类似第 3.2 节的 65/35 vs 25/75），并明确指出占比最大的 1–2 个非模型计算阶段。
