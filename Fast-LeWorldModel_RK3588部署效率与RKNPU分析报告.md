# Fast-LeWorldModel 在 RK3588 上的部署、效率评测与 RK-NPU 适配分析

- 目标板：`192.168.77.2`（hostname **YY3588**，瑞芯微 RK3588，Ubuntu 22.04.5 / aarch64 / 内核 6.1.75 / 16 GB RAM）
- 代码：<https://github.com/Yuntian-Gao/Fast-LeWorldModel>（Fast LeWorldModel，基于 LeWorldModel 的 JEPA 式世界模型 + CEM 模型预测控制）
- 日期：2026-09-14　｜　板端目录：`/root/Fast-LeWorldModel`　｜　conda 环境：`fast-lewm`（Python 3.10）
- 原始跑分：`bench_results_rk3588/bench_{all8,big4,big1}.json`（板端 `n_params` 与多次计时实测）

---

## 一、结论摘要（TL;DR）

1. **部署：模型本体已在板上跑通**。官方工程默认面向 NVIDIA CUDA，无法在无独显的 RK3588 直接安装运行：依赖锁 `torch==2.10.0+cu128`（无 aarch64 CUDA 包）、`eval.py` 硬编码 `.to("cuda")`、求解器 `device:"cuda"`、`MUJOCO_GL=egl`。已改用 **aarch64 CPU 版 torch 2.9.1+cpu** 建独立 conda 环境，成功构建真实网络并完成前向与跑分。
2. **效率：模型很小，但 CEM 推理范式决定 CPU 上不可实时。** 世界模型本体仅 **10.83 M 参数 / FP32 约 41 MB**（板上实测）；但每个动作要把 **300 个候选动作序列 × 30 次迭代**反复前向。实测最优核配置（仅 4×A76 大核）下，一次「300 候选×5 步推演」中位 **732.7 ms**，**单次动作决策约 22.1 s**；全 8 核约 **33.0 s**。相对 10–20 Hz 控制所需的 50–100 ms，仍慢约 **221–442 倍（最优配置）**。
3. **反直觉的工程结论：只用 4 颗 A76 大核比全 8 核（混入 4 颗慢 A55）还快约 33%**（rollout 732.7 ms vs 1091.8 ms）；4 大核相对单大核近似线性加速 2.97×。部署时应 `taskset -c 4-7` 绑大核，不要让小核参与。
4. **RK-NPU：硬件/驱动齐备，但无法整体一键上 NPU。** 板载三核 NPU（6 TOPS INT8）、RKNPU 驱动 v0.9.8、RKNN Runtime 2.3.2 均在。障碍在自注意力（运行时矩阵乘）、LayerNorm 仅部分支持、CEM 动态候选与控制流。**可行路线是异构拆分**：视觉 ViT 与以线性层为主的预测器在固定 shape 后量化上 NPU，CEM 采样/迭代留 CPU。
5. **关键判断**：单次决策耗时中 **99.4% 是世界模型 rollout，视觉编码仅 0.6%**。只把 ViT 放 NPU 几乎无效；必须把占大头的预测器/动作编码器固定 shape 后上 NPU，并**同步缩减 CEM 规模**——单靠 NPU 无法让该方案在 RK3588 实时。

---

## 二、目标环境（实测）

| 项目 | 实测值 |
|---|---|
| SoC | Rockchip RK3588：4×Cortex-A76 @2.35 GHz 大核（cpu4–7）+ 4×Cortex-A55 @1.8 GHz 小核（cpu0–3），L3 3 MB |
| 内存 / 磁盘 | 16 GB（可用约 14 GB）/ 根分区约 30 GB 可用 |
| GPU | Mali-G610 MP4（无 NVIDIA GPU） |
| NPU | 三核，300 MHz–1 GHz（`/sys/class/devfreq/fdab0000.npu`），标称 6 TOPS INT8，支持 INT8/INT16/FP16/BF16 |
| NPU 软件栈 | `/usr/lib/librknnrt.so`（**Runtime 2.3.2**，2025-04-09）、`rknn_server`、`rknn_api.h`、**RKNPU driver v0.9.8**；另有完整 `/root/rknn-llm`（含已转换的 Qwen 视觉 `.rknn`） |
| Python | 系统 3.10.12（无 pip）；Miniconda 26.1.1，已建 `fast-lewm`(py3.10) |
| 网络 | tuna 的 conda 通道、阿里云 PyPI 可用（tuna PyPI 节点与 pypi.org 在该网络下不通） |

---

## 三、部署过程与适配点

### 3.1 已完成步骤
1. 本地 `git clone`（GitHub HTTP/2 不稳，改 HTTP/1.1 成功）→ 打包上传到 `/root/Fast-LeWorldModel`。
2. conda 走 tuna 镜像创建 `fast-lewm`（Python 3.10，与 README 一致）。
3. pip 走阿里云镜像安装 **torch 2.9.1（aarch64 CPU，manylinux_2_28，104 MB，经 12327 成员 `testzip()` 严格校验）+ einops 0.8.2 + timm 1.0.29**（torchvision 0.24.1 作依赖）。
4. 编写 `bench_fast_lewm.py`：**直接 `import module`（仓库真实网络类）**，按 `config/train/Fast-lewm.yaml` 维度构建，前向与跑分全部成功。

### 3.2 官方工程在 RK3588 的不适配点（跑通完整 `eval.py` 闭环需改）
| 位置 | 问题 | 处理 |
|---|---|---|
| `requirements.txt` | 锁 `torch/torchvision/torchaudio==2.10/0.25/2.10 +cu128`，无 aarch64 wheel | 改 aarch64 **CPU** torch（本次 2.9.1） |
| `eval.py:305` | `model.to("cuda")` 硬编码 | 改 `.to("cpu")` 或按设备自适应 |
| `config/eval/solver/{cem,adam}.yaml` | `device:"cuda"` | 改 `cpu` |
| `eval.py:3` | `MUJOCO_GL="egl"`（偏 NVIDIA EGL） | 无头板改 `osmesa` 软渲染并装 OSMesa 库 |
| 运行前置 | 需 LeWM `.h5` 数据集、HF 权重 `naiverer/fast-leworldmodel`、`stable-worldmodel[train,env]`/`stable-pretraining`（带 mujoco/gymnasium/pygame/pymunk 重依赖链） | 需逐项在 aarch64 解决；本次效率评测**不依赖数据集/权重/MuJoCo**，直接测网络计算核，口径更干净、可复现 |

> 本次「跑通」层级是**模型计算核 + 推理路径**（动作编码器、预测器、视觉 ViT、rollout），正是效率与 NPU 评估关心的部分；MuJoCo 闭环属仿真环境集成，不影响下述计算结论。

---

## 四、效率评测（RK3588，FP32 CPU）

### 4.1 方法与口径
- 用仓库 `module.py` 的**真实类** `ActionPrefixEmbedder`、`ARPredictor`，严格按训练配置：`embed_dim=192`；动作编码器 3 层/6 头/dim_head32/MLP768；预测器 6 层/value_heads16/value_dim_head64/MLP2048/fusion768。
- 按评测配置 CEM：`num_samples=300`、`action_num_blocks=5`（每次推演 5 步）。视觉编码器用 timm `vit_tiny`(224) 作同量级近似（仓库为 patch14 DINOv2 ViT-tiny）。
- warmup 后重复 15 次（rollout/ViT 7 次）取中位数，`@torch.no_grad()`；**不含** CEM 采样/Python/MuJoCo。用 `taskset` 绑核、torch 线程数与核簇对齐。

### 4.2 模型规模（板上 `n_params` 实测）
| 模块 | 参数量 | 备注 |
|---|---:|---|
| ActionPrefixEmbedder 动作编码器 | 1,802,592 | 内含 3 层小 Transformer，自注意力序列长度仅 5 |
| ARPredictor 预测器（6 层） | 9,023,424 | Linear/GELU/SiLU/LayerNorm 为主，**无 self-attention**（Value-only） |
| **世界模型本体** | **10,826,016（≈10.83 M）** | FP32 权重 ≈ 41.3 MB |
| ViT-Tiny 视觉编码器（近似） | 5,524,416 | 每决策编码「当前帧+目标」各 1 次 |

### 4.3 全 8 核实测（正式，重复 15/7 次）
候选数扫描（中位 ms）：

| 单元 \ 候选 S | 1 | 32 | 100 | 300 |
|---|---:|---:|---:|---:|
| 动作编码器 | 8.4 | 34.0 | 51.2 | **114.6** |
| 预测器 | 53.0 | 45.4 | 74.9 | **123.9** |
| **5 步 rollout（一次 get_cost）** | 208.8 | 348.6 | 618.4 | **1091.8** |
| ViT-Tiny 编码 1×224 | — | — | — | 102.9 |

- S 从 1→300，rollout 仅增 5.23×（亚线性，多核把大批次并行摊薄）；S=1 仍需 208.8 ms，是 5 步串行前向的固定开销。
- 校验：5×(动作编码器 114.6 + 预测器 123.9)=1192.5 ms，与实测 rollout 1091.8 ms 同量级（大批次下融合/调度更优）。
- 进程峰值 RSS ≈ 483 MB。

### 4.4 核簇对比（S=300，中位 ms）—— 建议绑 4 大核
| 配置 | 动作编码器 | 预测器 | **5步 rollout** | ViT 编码 | **单次决策估算** |
|---|---:|---:|---:|---:|---:|
| big1：1×A76（1 线程） | 224.7 | 204.9 | 2177.6 | 114.5 | **65.56 s** |
| all8：4A76+4A55（8 线程） | 114.6 | 123.9 | 1091.8 | 102.9 | **32.96 s** |
| **big4：4×A76（4 线程）** | 84.0 | 74.1 | **732.7** | **50.1** | **22.08 s（最优）** |

- **big4 比 all8 快 1.49×**：大小核异构下，慢 A55 拖累负载均衡与同步，全核反而更慢。
- big4 相对 big1 加速 **2.97×**（4 大核接近线性）。
- 单次决策 = 30×rollout(S=300) + 2×ViT；即便最优 big4 仍 **22.08 s**，相对 10 Hz(100ms) 慢 **221×**、20 Hz(50ms) 慢 **442×**。

### 4.5 耗时构成与瓶颈
单次决策（all8 32.96 s）中 rollout 占 **99.4%**、ViT 仅 **0.6%**；rollout 内部预测器约占 54.5%、动作编码器约 45.1%。**瓶颈不是模型大小，而是 CEM 把同一小网络重复前向 30×300 次**。真实闭环还要叠加 CEM 采样/top-k、Python 循环与 MuJoCo 渲染，只会更慢。

---

## 五、RK-NPU（RKNN）适配评估

### 5.1 RKNN 能力边界（与本模型相关）
- 标准链路：**PyTorch → ONNX（固定 shape、去掉 topk/NMS 等动态后处理）→ x86 上 rknn-toolkit2 转换+量化（需校准集）→ 板端 librknnrt.so / rknn-toolkit-lite2 推理**；三核可独立/协同。
- NPU 面向**静态计算图**：shape 转换时确定（动态维度仅有限支持）；不支持的算子**回退 CPU**，NPU↔CPU 反复拷贝会显著拖慢。
- 与本模型相关的关键限制：
  - **标准 self-attention 难整图上 NPU**：QKᵀ、softmax(QK)·V 是「两个运行时才算出的矩阵相乘」，RKNN 的 MatMul 偏静态、且单次通常只用一个 NPU 核（社区移植 LLaMA/RWKV 的共识，需手工拆/改写）。
  - **LayerNorm 仅「部分支持」**（fp16、通道维范围与 NCHW 布局约束），不满足即回退 CPU。
  - GELU/SiLU/Softmax 等需对照 2.3.x 算子表，不满足要替换（如 SiLU→ReLU）或留 CPU。
  - 世界模型是**连续隐空间回归**，INT8 量化误差会在「30 迭代×5 步」rollout 中累积，可能扰乱 CEM 代价排序，需用真实数据做量化感知验证，必要时 FP16/混合精度（NPU FP16 吞吐低于 INT8）。

### 5.2 逐模块适配度
| 模块 | 主要算子 | 上 RKNN 适配度 | 说明 |
|---|---|---|---|
| 视觉 ViT 编码器 | Conv(patch)、Linear、self-attn(257 token)、LayerNorm | **中** | 卷积/线性友好，ViT 上 RKNN 有成熟实践；需固定 224、拆/改 attention、INT8 量化。每决策仅 2 次 |
| 动作编码器 | Linear + 3 层小 Transformer（**attn 序列仅 T=5**）+ 正弦 PE | **中** | attention 极小(5×5)且 T 固定，易固化；正弦 PE 预计算为常量 |
| **预测器（占 54.5% 时延、9.0 M 参数）** | 6×（Value-only 线性投影 + MLP 192→2048 + AdaLN 的 SiLU/Linear + LayerNorm + GELU） | **中偏高** | **无 self-attention，主体是 NPU 最擅长的大 GEMM**；主要工作是固定 S、处理 LayerNorm 布局/回退、确认 GELU/SiLU |
| CEM 求解器 | 随机采样、加噪、top-k、30 次迭代外层循环 | **不可上 NPU** | 纯控制流/动态算法，必须 CPU；rollout 逐步 `torch.cat` 的变长序列需改定长缓冲才便于导出 |
| 候选批量 S | 配置固定 300 | 需固化 | 按最大 300（或分档）静态导出，避免动态 shape |

### 5.3 三档落地路线（工作量从小到大）
- **路线 A（省事，收益很小）**：只把视觉 ViT 量化上 NPU。视觉仅占 0.6%，总时延几乎不变（22.08 s→约 22.0 s）。**不推荐作为主手段。**
- **路线 B（推荐主路线，异构）**：固定 S=300、rollout 变长 cat 改定长，将「动作编码器+预测器」（必要时含 ViT）导出 ONNX→RKNN，LayerNorm/不支持激活改写或划成 CPU 子图，CEM 外层留 CPU，三核并行子图。**这是唯一可能显著压缩那 99.4% rollout 的 NPU 路线**，需量化精度验证与一定工程改造。
- **路线 C（最重）**：仿 rknn-llm 手工拼/融合 attention、做 W8A8 级量化，追求 NPU 最大化占用；对这种小模型性价比低。

### 5.4 仅靠 NPU 不够，须叠加算法/工程优化
1. 降 `num_samples`（300→64/128）、`n_steps`（30→8/10）、top-k 热启动/低差异采样；
2. 扩大动作时间分块并复用（仓库 *fast buffered action path* 已是此思路，可更激进），编码特征跨迭代缓存；
3. 蒸馏更小预测器、FP16 推理、**进程绑 4×A76 大核（`taskset -c 4-7`，实测比全核快 1.49×）**；
4. 闭环实时需「降 CEM 规模 + 路线 B 固化子图上 NPU」同时取收益，再用真实 PushT 任务校验规划成功率。

---

## 六、交付物与复现
- 板端：`/root/Fast-LeWorldModel/`（源码 + `bench_fast_lewm.py`、`run_bench.sh`、`bench_*.json/log`）；wheel 缓存 `/root/wheels/`。
- 本地：本报告、`bench_results_rk3588/bench_{all8,big4,big1}.json`（原始跑分）、效率图表 HTML。
- 复现：
  ```bash
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate fast-lewm
  cd /root/Fast-LeWorldModel
  taskset -c 0-7 python bench_fast_lewm.py --threads 8 --tag all8
  taskset -c 4-7 python bench_fast_lewm.py --threads 4 --tag big4 --samples-scan 32,100,300
  taskset -c 7   python bench_fast_lewm.py --threads 1 --tag big1 --samples-scan 32,100,300
  ```

## 七、过程备注
- 评测途中板子曾从网络失联（后确认是**本机 Mac 的 VPN 接口 utun7 下发 `192.168.77.0/30` 路由劫持了到 .2 的流量**，加主机路由后恢复；同时板子 `uptime` 显示期间发生过一次重启）。三组基准在失联前已全部完成并落盘，未受影响，结果已取回。
- 完整 `eval.py` 闭环（MuJoCo+数据集+HF 权重）尚未在板上跑，需按 3.2 适配；不影响本报告计算效率与 NPU 判断。
- NPU 路线 B 的量化精度（尤其 predictor 多步 rollout 误差累积）需真实校准集与任务成功率实测确认。

---
### 参考资料（RKNN 能力边界）
- RKNN Compiler Support Operator List（LayerNorm 部分支持与算子约束），Rockchip 官方算子表
- Porting RWKV/LLaMA to RK3588 NPU（RKNN 无法直接做 self-attention、MatMul 单核）：<https://clehaxze.tw/blobs/porting-llm-to-rk3588.pdf>
- Rockchip NPU CPU fallback / 静态对齐约束：<https://clehaxze.tw/gemlog/2023/07-13-rockchip-npus-and-deploying-scikit-learn-models-on-them.gmi>
- RK3588 硬件规格（NPU 三核 6 TOPS INT8、CPU/GPU/内存）与 ONNX→RKNN 转换实践：CSDN RK3588S 系列、<https://blog.csdn.net/W25679/article/details/162368456>
