# RK3588 端侧机器人推理实验

本仓库按探索过程整理三大块实验：**Fast-LeWM 世界模型规划 → vla.cpp 推理框架对照 → SmolVLA 部署与系统优化**。vla.cpp 是推理框架，其测试模型也是 SmolVLA，不是第三种模型。所有结果均来自实际记录，未完成的机制与负结果也保留。

## 实验总览

| 实验块 | 要回答的问题 | 关键结论 | 证据范围 |
|---|---|---|---|
| Fast-LeWM / PushT | 世界模型规划应怎样分给 CPU 与 NPU？ | 固定搜索预算下，重规划均值约 4504 → 2652 ms；减少候选后进一步至 1566 ms，但改变了搜索预算。 | 同一批 50 个 PushT 样本；另有独立调参验证。 |
| vla.cpp / RK3588 | 现成 C++ 推理框架能否替代当前部署？ | 测试版本完整 CPU 路径慢、视觉输出差异较大；Prefill 单独更快，不代表整体更快。 | 官方 SmolVLA base、合成输入与精确输入诊断；没有任务成功率结果。 |
| SmolVLA / RKNN | 能否跑通 VLA，并从真实热点继续优化？ | base 合成输入完整推理约 1.503 s；LIBERO 闭环已跑通；近期实际观测离线重放约 2.08 s，剩余视觉约占 53%。 | 三种不同协议，不能拼成一条累计加速链。 |

这里的“强基线”指**当前已测、需要被新机制击败的优化对照**，不是理论性能上限，也不代表已达到 Roofline 算存平衡。

## 1. Fast-LeWM：世界模型与 CEM 异构规划

### 做了什么

使用 Fast-LeWM 的 PushT 预训练世界模型，以 CEM（交叉熵方法）搜索动作。实现对齐原论文的终端潜变量代价、权重的注意力分组、最终精英均值选择，以及**一次规划后执行完整 25 个动作**的节奏。

已验证的执行分工是：图像 ViT 与投影走 NPU，动作前缀编码走四个 Cortex-A76 大核，终端 Predictor 与投影走 NPU，代价计算与 CEM 更新留在 CPU。

### 主要结果

以下是同一批 50 个 PushT 样本的结果；时延为**每次重规划的平均等待时间**，不是每个动作的反馈时延。

| RK3588 配置 | 成功样本 | 平均每次重规划 |
|---|---:|---:|
| 纯 CPU，固定 300 候选、30 轮 CEM | 43/50 | 4504 ms |
| CPU/NPU，固定 300 候选、30 轮 CEM | 42/50 | 2652 ms |
| CPU/NPU，候选数 `300 → 150 → 64`，仍为 30 轮 | 44/50 | 1675 ms |
| 上一项加候选级 CPU/NPU 并行编码 | 44/50 | **1566 ms** |

![Fast-LeWM 的执行分工与完整 CEM 重规划耗时](breakdown.png)

上图来自独立固定负载计时：完整规划约 4.47 → 2.63 s，与 50 样本表不是同一组调用。除以 25 得到的摊销 ms/action 不等于逐动作闭环时延。

![Fast-LeWM 固定观测下的候选预算与分项耗时](hardware_icem_pilot.png)

上图是固定观测、三次运行均值的专项实验，不是 50 样本任务评测。分级方案对齐固定形状 RKNN 图，将候选评估数从 9000 降到 5140，因此后半段加速不能归为“等工作量系统优化”。

### 关键结论与负结果

- **异构分工比全部交给 NPU 更合理。** 完整动作编码器虽已转换、校验，但 batch-300 时 NPU 约 130 ms，慢于四核 CPU 的约 67 ms；当前路径选择 CPU 有实测依据。
- **先对齐语义，再比较性能。** 早期图预测五个未来潜变量却只消费最后一个，且注意力分组未对齐；修正后才比较同一个终端预测问题。八线程挤在四个大核上造成的退化，也不能解释成 NPU 干扰。
- **减少搜索预算有质量风险。** 50 个样本的成功数不足以证明跨任务质量不下降；输出余弦相似度也不能替代任务成功率。
- **动态停止未形成稳定优势。** 独立 600 样本合并验证中，动态 20/30 轮为 512/600，固定 25 轮为 514/600；平均重规划 838 vs 874 ms，但 P95 为 1045 vs 937 ms。该调参方向已停止，不作为论文贡献。
- 已测试系统中的 Mali GPU 没有可用的 OpenCL/Vulkan 计算设备，不属于已验证计算路径。

原始记录：[固定负载分解](results/latest_benchmark.json)、[候选级协同 50 样本](results/pusht_dataset_board_npu_hybrid_tiered_50.json)、[研究审查与后续门槛](RESEARCH_PLAN.md)。此方向保留为完成的实验积累，当前不继续以它立题。

## 2. vla.cpp：现成 C++ 推理框架的部署对照

### 做了什么

测试 `VinRobotics/vla.cpp` 的 `0644fc62` 版本，将官方 `lerobot/smolvla_base` 转为约 777.3 MiB 的 BF16 GGUF，在 RK3588 上使用四线程 CPU。**该测试版本没有 RKNPU 后端**，所以它不是 NPU 实现与 RKNN 的同设备后端对比。

### 主要结果

下表使用 base 的单相机、十步去噪合成 fixture。PyTorch/RKNN 是暖机后 2/5 次中位数，vla.cpp 是一次精确输入匹配诊断，单位 ms。各阶段中位数不能相加冒充总中位数。

| RK3588 路径 | 视觉 | Prefill | 十步去噪 | 完整推理 |
|---|---:|---:|---:|---:|
| PyTorch 四核 CPU | 3968 | 6469 | 29309 | 39759 |
| RKNN 视觉/去噪 + CPU Prefill | 872 | 6457 | 515 | 7856 |
| vla.cpp BF16 CPU，匹配输入单次 | 44785 | 5216 | 24542 | 74588 |

![vla.cpp 部署耗时与视觉路径输出差异诊断](figures/vla_cpp_pilot_summary.png)

### 关键结论与负结果

- **Prefill 单独更快，整体却更慢。** vla.cpp 的 CPU Prefill 约 5.22 s，低于 PyTorch 的约 6.47 s，但视觉约 44.79 s，不能直接替代已验证的 RKNN 路径。另一次默认暖机后 3 次调用的总中位数为 78.06 s。
- **视觉路径存在明显输出差异。** 完整动作相对 i5 PyTorch CPU 的余弦为 0.948093、MAE 为 0.20120；送入同一 PyTorch 视觉特征后，余弦升至 0.999991、MAE 降至 0.00249。这支持将主要差异定位到该版本的视觉路径，不代表任务质量已验证。
- **特征旁路不是完整推理加速。** 旁路后约 29.81 s 不包含视觉特征生成，不能与完整推理直接算加速比。
- **Q8 未跑通。** 测试版本拒绝部分视觉、连接器及语言层的量化权重；没有 Q8 时延或质量结论。公开接口也未暴露当前 RKNN 动作图所需的 Prefill K/V，不能直接拼接两套运行时。

结论：保留为 C++ 实现与数值诊断参考，不作为当前加速路线。详见 [vla.cpp 实验协议](docs/vla_cpp_pilot.md) 与[原始记录](results/smolvla_vla_cpp_rk3588_pilot.json)。

## 3. SmolVLA：从部署验证到系统机制探索

此块分成三种证据：**base 合成输入剖析、LIBERO 在线仿真闭环、实际观测离线重放与局部优化**。模型形状、输入、测量环境不同，不能直接跨表相减。当前研究保持模型、观测、全部 heads/tokens、动作数量与十步去噪，不靠跳帧、剪枝、近似复用或降低目标精度配置加速。

### 3.1 base：跨设备剖析与完整 NPU 主网络部署

使用 `lerobot/smolvla_base`，单相机、16 层动作专家、同一合成图像/指令/状态/初始噪声。RKNN 承担视觉、Prefill（多模态前缀处理）和动作去噪主网络；CPU 仍负责预处理、输入嵌入与去噪循环辅助，**不是纯 NPU 推理**。

| 路径 | 完整动作块推理中位数 |
|---|---:|
| RK3588 四核 CPU | 39.759 s |
| RK3588 NPU 视觉 + CPU Prefill + NPU 去噪 | 7.856 s |
| RK3588 NPU 视觉 + NPU Prefill + NPU 去噪 | **1.503 s** |
| i5-13490F CPU | 2.309 s |
| Mac M5 Pro MPS | 0.233 s |
| RTX 5060 CUDA | 0.122 s |

![SmolVLA base 各设备的分项耗时](figures/smolvla_stage_latency_stacked.png)

图的不同面板使用各自标明的线性轴；“其他”是使阶段合计对应总中位数的残差，不全是 CPU 算子。

**关键结论：** Prefill 搬到 NPU 后，该阶段约 6457 → 66 ms，完整路径由 7.856 → 1.503 s，视觉成为约 58% 的剩余热点。这是合成输入性能与数值验证，不是任务成功率；约 907 MB 权重可装入板端内存，也不代表纯 CPU 可用于交互控制。

补充图：[全部设备同轴对比](figures/smolvla_stage_latency_all_devices.png)、[各设备阶段占比](figures/smolvla_stage_share_donuts.png)。详细协议见 [base 阶段剖析](docs/smolvla_rknn.md)。

### 3.2 LIBERO：相同动作噪声下的仿真闭环

使用 `HuggingFaceVLA/smolvla_libero`，双相机、32 层动作专家。四种部署每一步使用完全相同的动作噪声，均完成 `libero_spatial` 任务 0、初始状态 0、种子 1000 的一个完整回合。

| 部署 | 成功所需动作数 | 平均推理耗时/动作 |
|---|---:|---:|
| RTX 主机 CPU | 70 | 4.849 s |
| RTX 5060 CUDA | 69 | **0.238 s** |
| Mac M5 Pro MPS | 70 | 0.509 s |
| RK3588 NPU 主网络 + CPU 辅助 | 68 | 3.236 s |

![相同噪声下的 LIBERO 闭环延迟与 RK3588 分项](figures/smolvla_libero_matched_breakdown.png)

**关键结论：** RK3588 已跑通闭环，但双相机视觉约占推理时延 54%，3.236 s/动作仍不足以视为响应式实体机器人控制。每种部署只有一个回合，68–70 的动作数差异不能用来排名成功率或精度。

推理计时不含输入预处理、动作后处理和模拟器步进；远端部署包含请求序列化与网络传输。模拟器始终在 RTX 主机，RK3588 只执行策略。base 与 LIBERO 的视觉/连接器权重经核验相同，但 Prefill/动作图不同，不能混用。详见 [闭环协议与数值边界](docs/smolvla_libero_closed_loop.md)。

### 3.3 完整策略优化：常驻原生缓冲区与条件准备外提

实际观测来自固定版本 LIBERO-Spatial 演示数据的一个 episode、三个帧；保留双相机、32 层动作专家和十步依赖 CPU FP32 Euler。每帧各方案固定同一初始噪声，两次独立会话，每帧每方案 8 次。

| 完整策略路径 | 六组“会话 × 帧”的总时延中位数范围 |
|---|---:|
| 原 Lite 动作接口，视觉已做整相机并发 | 2553–2563 ms |
| 原图常驻原生输入/输出绑定，减少重复接口工作 | 2135–2138 ms |
| 再将固定条件准备外提为每观测一次，缓存为消费就绪布局 | **2078–2084 ms** |

![完整策略各轮优化与每个部分省时](docs/figures/smolvla_optimization_progress/01_policy_progress.png)

**关键结论：** 常驻接口吸收了主要工程开销；条件准备外提相对常驻强对照再降低约 2.43%–2.78% 的完整策略中位数。图的代表帧中，十步 Action 分项均值分别省约 426 ms、52 ms，视觉与 Prefill 基本不变。最终视觉约占 53%，Action 约占 34%。

所有测试步骤的 velocity、FP32 latent 与最终输出逐位一致，比较对象是共享视觉/Prefill 后的原 RKNN 路径，不是 CPU FP32 全模型。**这是离线观测重放，没有新闭环成功率**；计时不含外部预处理、网络、模拟器、动作后处理与图初始化，不能写成此前 3.236 s 闭环直接降到 2.08 s。

独立合成条件的完整 action-flow 专项也测得 747.77/749.43 → 692.65/693.32 ms，约降低 7.4%；一次准备已计入，但不是完整 VLA 的 7.4%。详见 [条件缓存验证](docs/SMOLVLA_CONDITIONING_VALIDATION.md) 和[完整策略重放](docs/SMOLVLA_POLICY_REPLAY_VALIDATION.md)。

### 3.4 视觉优化：原生布局续接与 Head 并行

直接拆 Attention 图会增加布局转换与交接。实验将 head 分区与前后图的 NPU 原生缓冲区布局对齐，让不同核读写同一缓冲区的互不重叠区域，避免 host 上拆分、拼接 Q/K/V。

| 完整视觉路径 | 单视角中位数 | 双视角驻留计算中位数 |
|---|---:|---:|
| Patch / 恒零 Mask 修复后的融合控制 | 775.49 ms | 串行 1532.02 ms |
| 整相机分核并发的融合强对照 | 不适用 | **1103.42 ms** |
| 原生分图，4+4+4 heads 并行 | **531.62 ms** | 相机串行 1056.27 ms |
| 上一项再让两相机 pipeline 并发 | 不适用 | 1056.44 ms |

单视角与双视角来自各自专项，不跨列累计。驻留计算包含图提交/同步，不含输入准备和最终 host 读回。

![视觉优化历程、分项收益与失败尝试](docs/figures/smolvla_optimization_progress/02_vision_progress.png)

**关键结论：** 单视角相对融合强基线省约 244 ms（31.4%），但双视角整相机并发已吸收多数收益，head 方案只再省约 47 ms（4.27%）。Attention 专项阶段约 500 → 220 ms，而输出投影/MLP 仍约 231 ms；继续叠加相机并发或跨阶段交错没有扎实净收益，部分方案更慢。

保留的另一负结果是：双相机合批图的恒零 Mask 修复大幅减少 SDK 分配字段，却只将时延 1739.01 → 1722.39 ms，约 0.96%。分配字节不是实测 DDR 流量，忙核百分比也不是 MAC 利用率。

数学等价改写与“不降低精度配置”不保证融合后浮点舍入一致。head 分图相对原融合 FP16 的特征输出非逐位相同，**任务质量门槛未完成，未计入上一节完整策略**。详见 [视觉专项调查](docs/SMOLVLA_VISION_INVESTIGATION.md)。

### 3.5 动作 Cross-Attention：CPU/NPU 原生位桥接

动作专家的此处有 15 个 Q heads 共享 5 组 K/V。紧凑执行保留共享 K/V，将同组查询打包以改善 NPU Attention 几何，但这种排列不匹配原来的前后算子；编译器自动恢复会引入额外 reshape/恢复矩阵，改投影权重又会改变浮点累加顺序。

最终机制是：

```text
原 Q/RoPE → CPU 原生存储位打包 → NPU 紧凑 Attention 并行
         → CPU 原生存储位恢复 → 原输出投影 / 残差
```

CPU 仅复制已有 FP16 存储位，不做重新量化或浮点计算；全部 heads、tokens 与输出投影顺序保留。它不是 CPU softmax，也不是已经实现的 CPU/NPU overlap。

![原生位桥接的各轮方案与阶段耗时](docs/figures/smolvla_optimization_progress/03_c17_progress.png)

**关键结论：** 同轮十次消费，head 并行强对照 11.46 ms，CPU 仅恢复输出 10.87 ms，双向位桥接 10.28 ms；三轮相对 head 控制降低 10.3%–10.7%。新增双向位复制/同步约 0.397 ms，换来更便宜的 Attention 与前后交接，而不是减少模型工作。

512 个合成扰动、30 个观测驱动步骤逐位通过。观测驱动重放仍省约 **1.2 ms/观测/该块十次消费**，但只覆盖一个 Q/RoPE/Attention/输出投影边界，不含 MLP，也未让该实现反馈进完整 Euler 轨迹。**局部 10% 不等于完整 VLA 10%**；尾延迟也非始终改善，暂不机械铺开全部层。

完整数据、阶段拆分与失败消融见 [优化进展图表](docs/SMOLVLA_OPTIMIZATION_PROGRESS.md) 和[原生位桥接续验](docs/SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md)。

### 3.6 无损端云传输：先否定“特征一定更小”

同一批六张实际记录图像，每视角无损 PNG 为 **57–66 KiB**；NPU connector 的既有 FP16 存储经无损 zlib 压缩仍约 **111 KiB**。因此当前输入上“先提特征再上传可省字节”的默认动机不成立。

FP16 表示在此可逐位恢复，是因为这些 NPU 输出本来就用 FP16 存储，不是新降低目标精度；不能推广到任意云端 FP32 特征。目前只完成字节与编解码测量，**没有端云时延胜区或协作加速结论**。详见 [无损 payload 实测](docs/SMOLVLA_POLICY_REPLAY_VALIDATION.md)。

## 当前研究重点

当前主模型固定 SmolVLA，不继续 Fast-LeWM 调参，也不通过训练、后训练、减观测/动作/去噪步骤获得收益。上述工程收益与局部机制尚未构成完整 CCF-A 论文证据。

| 接下来验证什么 | 已有依据 | 必须补的证据 |
|---|---|---|
| 视觉关键路径与重要布局边界 | 实际观测下视觉约占 53%；原生位桥接的局部增量成立。 | 相同原因是否在重要热点重复出现；超过最强整图控制的净收益与任务质量。 |
| Attention 内部的异构交接 | CPU/NPU 的阶段能力可能不同，现有探针不足以证明赢家反转。 | 真实阶段时延、精确归一化与交接成本；没有净收益前不建设复杂 runtime。 |
| 端云分工是否改变本地最优并行维度 | 单/双相机最优并行可能不同；特征不是天然更小。 | 实际分支服务矩阵、网络与云排队；同时击败优化全端、最强静态分工及可流式重叠的全云。 |

这三项是独立假设，不是一个失败后自动接班的方案。详细进展见 [后续机制计划](docs/NEXT_MECHANISM_PLAN_2026-09-28.md)、[候选想法与判废条件](docs/IDEA_CANDIDATES.md)。历史文档中的候选编号以各文档说明为准，README 使用机制名称。

## 设备、制品与复现

- [机器与环境交接](docs/test_hosts.md)：RK3588、RTX 主机、Mac，以及完整评测顺序。
- [模型制品备份](docs/artifact_backups.md)：区分 Fast-LeWM、SmolVLA base、SmolVLA LIBERO；大模型/编译图不入 Git，校验清单见 [制品 SHA-256](artifacts/2026-09-24.sha256)。
- [LIBERO 闭环脚本](scripts/eval_smolvla_libero.py)：`--matched-noise` 对齐噪声；[闭环原始数据](results/smolvla_libero/)中的 `matched_*_full.json` 对应任务结果。
- [完整策略与局部原始数据](results/smolvla_libero/policy_replay_probes/)：仅离线重放，不能替代闭环。

在已配置的 Mac 环境重画图，不会重跑性能实验：

```sh
PY="$HOME/.cache/fast-lewm-smolvla-venv/bin/python"
MPLBACKEND=Agg "$PY" scripts/plot_breakdown.py
MPLBACKEND=Agg "$PY" scripts/plot_hardware_icem.py
MPLBACKEND=Agg "$PY" scripts/plot_vla_cpp_summary.py
MPLBACKEND=Agg "$PY" scripts/plot_smolvla_stage_comparison.py
MPLBACKEND=Agg "$PY" scripts/plot_smolvla_libero_summary.py
MPLBACKEND=Agg "$PY" scripts/plot_smolvla_optimization_progress.py
```

已部署 Fast-LeWM 权重与固定形状图的板端，可用以下命令启动规划服务；重跑前核对图文件、Runtime 版本和数据集：

```sh
ssh -F ~/.ssh/config_rknn rk3588 \
  'cd /root/Fast-LeWorldModel && taskset -c 4-7 \
   /root/miniconda3/envs/fast-lewm/bin/python -u rk3588_planner_server.py \
   --mode npu --cem-steps 30 --num-samples 300 \
   --candidate-schedule 300x10,150x10,64x10'
```

所有成功数仅适用于写明的数据集与回合数；模拟器成功不能替代实体机器人时延、质量与安全性验证。
