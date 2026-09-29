# SmolVLA 优化进展图与原生位桥接

2026-09-29；只重画已有测量，没有新增性能实验。

历史候选编号对照：C17 指 CPU/NPU 原生位桥接；C06 指固定条件准备外提与消费就绪缓存。编号不是模型名或系统名。

## 原生位桥接是什么

原生位桥接优化的是 action expert 的一个 cross-Attention 边界，不是视觉编码，也不是完整 transformer block。
当前几何是 15 个 Q heads、5 个 KV heads、50 个查询、149 个条件 token；包含 Q/RoPE、Attention、输出投影与残差，不含 MLP。

原来为了便于执行，会把每组共享的 K/V 展开成三份，供三个 Q heads 消费。紧凑方案保留 5 组 K/V，将共享同一 K/V 的三个 head 的查询打包，数学工作不变。
真正的障碍是：这个紧凑 Attention 的输入/输出排列与原 Q/RoPE、输出投影的原生存储不匹配；让编译器恢复会出现额外 reshape/恢复 Conv，让矩阵权重顺着改又改变浮点累加顺序。

最终路径：原 Q/RoPE → CPU 原生位打包 → NPU 紧凑 GQA 并行（2+2+1 KV groups）→ CPU 原生位恢复 → 原输出投影/残差。
CPU 只搬 FP16 存储位，不重新计算数值、不降低精度；每方向复制 96 KB，双向复制和同步十次约 0.397 ms（S13）。条件缓存 583680 → 194560 bytes/层，不是峰值 RSS 或实测 DDR 流量。

已测单块的 512 个合成扰动及 30 个观测驱动步骤逐位通过，但不是全输入数学证明。局部比 head 强对照快约 10%，每观测/该块十次省约 1.2 ms；还没集成完整策略，更不能称为 VLA 整体快 10%。

## 图与统计口径

所谓‘强极限’应改称‘当前已测强基线’，不是理论性能上限。没有充分证据证明 NPU 已达到 Roofline。
完整策略堆叠的是同会话、同帧、同一组调用的分项均值，均值之和核验等于总均值；总中位数另标。
局部阶段使用分组柱画各自中位数，不能相加成总中位数。不混池不同会话、输入、I/O 口径，也不将尚未集成的方案串加。

![完整策略](figures/smolvla_optimization_progress/01_policy_progress.png)

### 完整策略代表帧的每步分项变化

strong2 / frame 0 / 每方案 8 次，单位 ms；正值是省时。Action 包含一次条件准备及十步依赖 FP32 Euler。

| 步骤 | 视觉 | Prefill | Action | 其他 | 总均值省时 |
|---|---:|---:|---:|---:|---:|
| 常驻绑定 vs Lite | +0.11 | +1.72 | +425.82 | +0.59 | +428.24 |
| 条件准备外提 vs 常驻绑定 | -0.07 | -1.42 | +51.52 | +0.79 | +50.81 |

视觉/Prefill 约 0-2 ms 的非目标变化属于本轮观测波动，不归因于优化机制。

### 两次会话、全部三帧的完整策略总中位数

| 会话 / 帧 | Lite | 常驻绑定 | 条件准备外提 | 常驻省时 vs Lite | 条件外提省时 vs 常驻 |
|---|---:|---:|---:|---:|---:|---:|
| strong1.json / frame_0.npz | 2562.65 | 2136.10 | 2077.69 | 426.55 | 58.41 |
| strong1.json / frame_42.npz | 2552.89 | 2138.49 | 2078.93 | 414.40 | 59.56 |
| strong1.json / frame_83.npz | 2553.71 | 2135.74 | 2083.84 | 417.97 | 51.89 |
| strong2.json / frame_0.npz | 2553.61 | 2134.73 | 2078.58 | 418.88 | 56.15 |
| strong2.json / frame_42.npz | 2553.52 | 2136.47 | 2078.19 | 417.05 | 58.28 |
| strong2.json / frame_83.npz | 2552.84 | 2136.25 | 2078.41 | 416.59 | 57.84 |

以上是总中位数之差，不等于逐轮配对差值的中位数。原 Lite 列已经共享优化后的整相机并发视觉，不是最早 3.236 s 闭环部署。

![视觉优化](figures/smolvla_optimization_progress/02_vision_progress.png)

### 视觉各部分省时

额外三轮分项诊断，完整单视角 12 层；全头分图减去 head 并行分图的中位数，不能并入正式总时延中位数。

| 部分 | 省时 / ms |
|---|---:|
| Patch | +0.002 |
| Norm + QKV | -0.032 |
| Attention + 等待 | +279.856 |
| 输出投影 + MLP | -0.112 |
| Connector | +0.003 |

单视角 head 方案相对融合强基线 775.49 → 531.62 ms，省 243.88 ms；双视角最强融合并发 1103.42 → head 方案 1056.27 ms，仅再省 47.15 ms（4.27%）。
这两组不是同一实验，不能相乘。图中 Patch、head 分图相对各自原融合 FP16 均有舍入差异；没有减精度配置，但完整任务质量还没过门槛。
B2 mask 修复大幅减少分配字段，却只省 16.62 ms。跨阶段 overlap 的几个尝试持平或更慢；图中保留失败结果。

![原生位桥接优化](figures/smolvla_optimization_progress/03_c17_progress.png)

### 原生位桥接每个部分的变化

S13，同轮十次消费的分项中位数；表内省时为两分项中位数之差。CPU 位复制包含同步，输入 hidden 打包与最终输出读回在总时延中另计。

| 部分 | Head 强对照 | Compact / 编译恢复 | CPU 单向恢复 | CPU 双向桥接 | 双向 vs Head 省时 |
|---|---:|---:|---:|---:|---:|
| 一次条件准备 | 0.675 | 0.542 | 0.540 | 0.540 | +0.135 |
| Q / RoPE Producer | 3.613 | 4.430 | 4.440 | 3.630 | -0.017 |
| Attention + 并行等待 | 5.046 | 3.560 | 3.561 | 3.569 | +1.477 |
| 输出投影 / Consumer | 1.771 | 3.906 | 1.786 | 1.791 | -0.020 |
| CPU 位复制 + 同步 | 0.000 | 0.000 | 0.187 | 0.397 | -0.397 |

最终图中保留全部 head 和条件 token。Compact 本身让 Attention 快约 1.48 ms，却会让上下游更贵；双向 CPU 位桥接保住原 Q/O 算术路径，绕开恢复矩阵。它不是 CPU softmax，也不是已实现的 CPU/NPU overlap。

S13 总中位数：head 11.463 → 单向位恢复 10.865 → 双向位桥接 10.281 ms。三轮 S13-S15 比 head 低 10.3%-10.7%；旧传播布局/权重重索引方案未通过扩大数值门槛，不计入合法最优。
真实条件重放中每块只省约 1.2 ms，约为 2.08 s 完整策略的 0.058%。这只是重要性对照，不是已集成策略收益，不能机械外推全部层。尾延迟并非始终改善。

### S13 全部 19 个消融（没有只挑赢家）

| 实现标识 | 十次消费总中位数 / ms |
|---|---:|
| `ready_joint` | 15.418 |
| `propagated_ready` | 13.567 |
| `restore_o` | 14.304 |
| `original_q` | 26.363 |
| `original_q_restore_o` | 27.344 |
| `split_full_heads` | 16.828 |
| `split_serial_heads` | 17.958 |
| `split_parallel_heads` | 11.463 |
| `original_recompute` | 18.952 |
| `compact_split_full` | 15.696 |
| `compact_split_serial` | 16.814 |
| `compact_split_parallel` | 12.790 |
| `original_order_compact_full` | 20.510 |
| `original_order_compact_serial` | 20.389 |
| `original_order_compact_parallel` | 13.573 |
| `compact_parallel_cpu_restore` | 10.865 |
| `compact_full_cpu_restore` | 13.718 |
| `compact_parallel_cpu_both` | 10.281 |
| `compact_full_cpu_both` | 13.118 |

此表是消融，不是全部方案具有同一数值资格或穷尽全部编译配置。

## 证据与边界

- [完整策略验证](SMOLVLA_POLICY_REPLAY_VALIDATION.md)；只含离线模型内推理，不含外部预处理、网络、仿真、动作后处理或图初始化。
- [视觉专项](SMOLVLA_VISION_INVESTIGATION.md)；完整融合相机并发已在实际帧逐位通过，packed/head 分图未作为完整策略质量已通过的实现。
- [原生位桥接续验](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md)；真实条件重放的 hidden 是 CPU FP32 重建，不是捕获的 NPU 内部 hidden。
- [绘图数据](figures/smolvla_optimization_progress/chart_data.json)；图与表直接由原始 JSON 生成，SVG/PDF 同目录。
- 不把旧闭环、合成输入、不同版本模型的数字拼成一条 3.236 → 2.08 s 加速链。Prefill 暂未在这一轮建立独立优化收益，端云也未取得实测胜区。
