# SmolVLA 视觉机制调查

2026-09-30 补充：[实际服务矩阵与阶段几何反事实](SMOLVLA_VISION_SERVICE_VALIDATION_2026-09-30.md)
已测六张实际图、两会话；单相机 head 收益约 32.4%–32.8%，双相机相对融合
整相机并发约 6.2%–6.3%。归一化后拆边界可保住更快 MLP 几何，但原数值
gate 尚未通过；简单单图 reshape 也失败。最新顺序以 [执行计划](PLAN_VISION_BOUNDARIES_2026-09-30.md)
和 README 为准；下面保留历史专项，不代表当前还要继续扫描 FC1 分区。

2026-09-29 补充：[实际观测完整策略重放](SMOLVLA_POLICY_REPLAY_VALIDATION.md)
已接入融合整相机并发强控制，真实演示三帧中视觉约 1.108 s、占优化策略约
53%；相机并发/交换/恢复逐位通过。下文 packed/head 结果仍是此前专项口径，
未据此宣称完整策略质量已过关。端云 payload 也已补实测，特征并不比 PNG 小。

更新：2026-09-28。定位：实验决策记录，不是已成立的论文贡献。
本页保留视觉专项过程。最新总体执行顺序见 [五条复审](IDEA_REASSESSMENT_2026-09-28.md)：
FC1 分区已降为可选工程项，不再作为优先主线。
当前只研究视觉；保持输入、模型参数、所有 token/head、动作数量和去噪步骤。
不采用剪枝、近似 Attention、跳帧或有损 temporal reuse。

## 1. 当前判断

**最值得继续的问题：数学上独立的视觉计算，为什么在实际 NPU 图里不能低成本地独立执行？**

新证据不是“多核比单核快”，而是：同样的 head 分区，若边界仍采用
token-major 的语义张量，就需要额外布局处理；将前后图接口一起改成兼容的
原生布局后，head 分区可以成为共享缓冲区的视图，而不是 CPU 上新生成的张量。

我们已经实现并测量了真实 vision layer-0 的完整 block，不只是一个 toy
Attention。单层在修复后的原生布局融合控制下为 62.98 ms，三路 head 并行为
42.24 ms。现在完整单视角的 patch embedding、12 层和 connector 也已接通：
775.49 -> 531.62 ms，降低约 31.4%。但同口径双视角强对照已将收益压缩到
约 4%-6%：整视角分核并发 1103.42 ms，head 并行串行处理两视角 1056.27 ms。
双视角同时跑 head 并行没有明显进一步缩短驻留计算，尾延迟反而更大。
当前是有效的单视角机制、偏弱的双视角系统收益，不足以确认 CCF-A 论文立题。
闭环质量仍未验证。

## 2. 痛点、动机与方案

直白地说：**最耗时的 Attention 有可以同时算的 heads，却仍主要由一个
NPU 核执行。直接把它拆成几张图，又会把中间张量取出来、换布局、重新提交，
让并行省下的时间被交接成本吃掉。**

动机来自三项测量，而不是先发明一个调度矛盾：

- LIBERO 的 RK 推理平均 3.236 s，双视角编码 1.749 s，占约 54%。
- 原视觉 SDK profile 中 Attention 494.06 / 848.40 ms，占约 58.2%；
  core mask 1/7 的 Attention 时间基本相同，FC2 等算子则能利用多核。
- 早期外部 head 并行约 45.6 ms，仍慢于原融合 Attention 的约 41.2 ms。
  只证明有并行性，并不证明拆图值得做。

局部方案：

```text
packed hidden buffer ----------------------------------------+
       |                                                     |
       v                                                     |
Norm1 + Q/K/V projections (original weight computation)       |
       |                                                     |
       +-- Q/K/V native buffers: three disjoint channel views |
                |               |               |            |
             heads 0-3       heads 4-7       heads 8-11       |
              core 0          core 1          core 2         |
                +---------------+---------------+            |
                | disjoint writes to one native output buffer|
                v                                            |
output projection + residual <-------------------------------+
Norm2 + MLP + residual
```

这里不改变模型工作量。CPU 负责提交与同步，没有把 CPU 计算加速包装成已验证的
异构优势；投影/MLP 保留原编译方式，也不假定所有线性算子都能高效三核执行。

## 3. 关键细节：绑定原生内存还不够

真实 Attention 原接口是 `[1,1024,768]`，SDK native attr 仍是 rank-3
`UNDEFINED`，即按 token 排列通道。一个 head 组跨所有 token，不能简单地
用一个连续 FD-offset 区域代表。

我们将等价图接口改为 channel-major `[1,768,1,1024]`，让 SDK 暴露
`NC1HWC2 [1,96,1,1024,8]`。全 Q/K/V 各 1572864 bytes；4-head 分区为
`[1,32,1,1024,8]`，各 524288 bytes。分区沿 channel-block 轴连续，
可用 `rknn_create_mem_from_fd` 的 0/524288/1048576 offsets 直接绑定。

输出分区直接写入同一个全头输出缓冲区的不同区域，后续图直接读取该缓冲区。
没有 host Q/K/V 分拆、host concat 或每次 `outputs_get`。默认 SDK cache
同步仍开启，并未将一致性成本假设为零。残差支路共享 producer 的输入缓冲区。

注意：这是 B1、固定几何下经过 attr/stride 检查的实现。不能假定 B2、任意
padding 或任意张量维度都能这样取连续视图。最初单层 consumer 最终输出仍是
rank-3；完整 12 层版本已将 consumer 输出也改为 packed layout，逐层直接续用。

## 4. 完整 Block 结果

RK3588，RKNN Toolkit 2.3.2，FP16，真实 layer-0 权重，1024 token、12 heads、
head dim 64。每种计划 warmup 后轮换顺序执行 20 次，报告中位数。
计时包含 producer、Attention、consumer、提交和同步，不包含输入准备与最终
host 输出取回。所有对照采用相同的 resident native-I/O 计时口径。

| 计划 | 两路 head 实验 | 三路 head 实验 |
|---|---:|---:|
| 原 rank-3 输入的融合 block | 66.78 ms | 66.76 ms |
| 零 mask 修复 + packed 输入的融合 block | 63.07 ms | 62.98 ms |
| packed 分图，但 Attention 仍一次计算 12 heads | 64.95 ms | 65.23 ms |
| packed 分图 + 独立 head 并行 | 47.85 ms（6+6） | 42.24 ms（4+4+4） |

分图本身略慢，head 并行才带来净收益。这是比“原来没绑 native I/O”更强的控制。
两路/三路数据在不同轮次获得，不能将其小数差当成严格跨计划配对。
最终可配置版的两路复测为原融合 66.91 ms、native 融合 63.09 ms、native
分图全头 66.00 ms、两路 head 48.63 ms，主结论一致；保留各轮原始记录。

数值边界必须保留：

- ONNX ReferenceEvaluator 的随机 FP32 对照：零 mask 修复、head 分区及整个
  native-interface split block 的最大差均为 0；同时检查了输出有限性。
- RK 三组随机 hidden 输入：并行方案与 native 分图全头控制输出位一致。
- native 融合/分图相对原融合 FP16 结果并非位一致。分图最大差 0.015625，
  平均绝对差约 0.000148。数学等价改写不保证编译融合后的浮点舍入一致。
- 这些单层测试不覆盖真实图像分布、动作以及闭环成功率；12 层和 connector
  的后续测试见下节。不减少精度配置、不减少工作量，也不能宣称质量完全无变化。

原始结果：[两路](../results/smolvla_libero/vision_probes/native_block_2way.json)、
[三路](../results/smolvla_libero/vision_probes/native_block_3way.json)、
[最终版两路复测](../results/smolvla_libero/vision_probes/native_block_2way_final.json)。

### 完整 12 层与 Connector：已接通

完整单视角，使用已有 exact patch space-to-depth 表示。每种计划轮换顺序
测量 20 次。所有计划预先准备 resident native FP16 输入；计时包含整个视觉
及 connector 的运行、层间调用和默认 SDK cache 同步，不含输入准备、初始化、
最终 host 输出取回。不是双视角、VLA 整体或闭环指标。

| 计划 | 完整单视角中位数 |
|---|---:|
| patch 修复后的原融合图 | 788.69 ms |
| patch + 零 mask 修复后的融合强基线 | 775.49 ms |
| 原生布局完整分图，但 Attention 不分 head 并发 | 810.25 ms |
| 原生布局完整分图，4+4+4 heads 并发 | **531.62 ms** |

相对融合强基线为 1.46x，时延降低 31.4%。分图全头比融合强基线慢约 4.5%，
因此收益不能归因于分图或 native binding 本身。该轮正式测量尚未加载额外的
多输出 audit graph；后续诊断复测单独记录，不混入这 20 轮。

持久数据路径使用两个 hidden 缓冲区交替承接 12 层输出、三个共享 Q/K/V
缓冲区和一个共享 Attention 输出缓冲区，共 9 MiB 的这些外部张量缓冲区。
它不包括图内临时区、图权重、输入和 connector 输出，不是整个运行时峰值内存。
24 个带权 producer/consumer 图各自保留真实层权重；参数为空且经结构指纹
检查相同的 Attention 图/contexts 跨全部 12 层复用。

额外三轮分阶段诊断给出了瓶颈迁移（阶段中位数之和不是正式总时延中位数）：

| 阶段（全 12 层合计） | 全头分图 | 并行 head 分图 |
|---|---:|---:|
| patch embedding | 11.66 ms | 11.66 ms |
| Norm1 + QKV producer | 55.41 ms | 55.44 ms |
| Attention（含提交、等待） | 499.99 ms | 220.14 ms |
| 输出投影 + residual + Norm2 + MLP consumer | 231.10 ms | 231.22 ms |
| postnorm + connector | 12.04 ms | 12.04 ms |

**Attention 的主导地位已经改变：优化后 consumer 约 231 ms，略高于 Attention
约 220 ms。** 下一步不能继续仅扩大 Attention 并行度；需要检查两视角下这两种
阶段的资源占用与相互干扰。此结果不证明它们已经逼近 Roofline，也没有隔离
consumer 内具体哪一个算子或 DDR/vector/矩阵单元是瓶颈。

数值与状态验证：

- 完整 packed/head-partitioned FP32 参考在一个合成图像上，12 个层输出以及
  connector 输出相对原 ONNX 最大差均为 0。
- 同一图像的 NPU 全头/并行路径，逐层输出和 connector 输出位一致。
  换为水平/垂直翻转图像后最终输出仍位一致；切回原图、连续轮换运行后输出不变。
- 相对原融合 FP16，connector 最大差 0.1875，MAE 0.01158；两张翻转图像
  分别为最大差 0.171875/0.15625，MAE 约 0.01169/0.01164。
- connector 对 FP32 的 MAE：原融合图 0.05392，分图路径 0.05431。
  不能把相近的特征误差当成相同任务质量，也不能将差异全部归因为舍入。
- 单独构造多输出 fused audit graph 获取各层：该合成图像的最终 connector
  与原融合图位一致。最后一层 hidden 对 FP32：audit 最大差 16.7841、MAE
  0.02616；pipeline 最大差 15.4091、MAE 0.02662。该层 FP32 最大幅值约
  389.49；不可仅凭绝对最大差忽略或夸大误差。pipeline 对 audit 的最后一层
  最大差 3.875、MAE 0.00622。额外层输出会改变编译的可见边界，audit 仅作
  诊断对照，最终输出一致不是所有内部执行完全相同的证明。
- 尚未接入 SmolVLA policy 的闭环调用，也未测真实机器人观测分布与成功率。

数据：[完整正式测量](../results/smolvla_libero/vision_probes/vision12_paired.json)、
[FP32 逐层 manifest](../results/smolvla_libero/vision_probes/vision12_export_manifest.json)、
[数值 audit 复测](../results/smolvla_libero/vision_probes/vision12_audit.json)。

### 双视角强对照：单视角收益不能直接相乘

完整 12 层与 connector；相同两张合成图（fixture 与水平翻转），12 轮轮换
计划和计时模式顺序。融合控制均已修复 patch 与零 mask。每套图常驻，第二
相机通过 `rknn_dup_context` 初始化，但单独分配 I/O；没有物理权重共享的测量。
每轮既测驻留计算，也测 CPU FP32 图像转 native FP16、同步、计算和 connector
取回 FP32；后一口径不含图像 resize/normalize、真实摄像头或后续 VLA。

| 计划 | 驻留计算中位数 | 含上述 I/O 中位数 | 驻留计算 p95（nearest rank） |
|---|---:|---:|---:|
| 融合图两视角串行，各 mask7 | 1532.02 ms | 1544.80 ms | 1541.92 ms |
| **融合图整视角并发，mask1/mask2** | **1103.42 ms** | **1115.65 ms** | **1106.36 ms** |
| 融合图整视角并发，mask4/mask3 | 1137.23 ms | 1150.39 ms | 1139.39 ms |
| 融合图整视角并发，mask3/mask4 | 1135.04 ms | 1147.70 ms | 1135.86 ms |
| 融合图整视角并发，各 mask7 | 1529.08 ms | 1541.90 ms | 1533.64 ms |
| head 并行，串行处理两视角 | 1056.27 ms | 1068.84 ms | 1073.06 ms |
| head 并行，两套完整 pipeline 并发 | 1056.44 ms | 1050.58 ms | 1101.05 ms |

对最强整视角基线，串行 head 方案降低约 4.27%/4.19%；两套并发降低
约 4.25%/5.84%。各方案与该基线逐轮比较均为 12/12 次更快，但只测 12 轮，
p95 仅作尾部诊断。并发方案含 I/O 中位数低于驻留中位数，不应解释为 I/O
带来加速：它们是独立时段测量，并发执行的调度波动更大。

所有计划相对各自融合/分图 anchor 输出位一致；交换两张图后输出对应交换，
恢复输入和切换计划后结果不变。分图相对融合的两个 connector MAE 为
0.0115756/0.011693，max 为 0.1875/0.171875，不能宣称融合/分图位等价。
测量期 4108 个采样点的 NPU/DRAM 频率始终为 1.0 GHz/2.112 GHz。

并发 packed 的驻留阶段三核 driver busy 均值约 95.75/66.98/67.14%，串行
packed 约 96.63/45.77/46.82%，但 makespan 基本相同。busy 的统计窗口未知，
跨计划可能有边界污染；它不是 MAC occupancy，也不是 DRAM 带宽证据。
**更多核显得忙，不意味着更多有效进展。** 这只能触发竞争/阶段互补实验，
不能直接断言 bandwidth-bound，更不能重新使用已被反证的图级独占解释。

SDK 不支持 mask6；有效枚举有单核、core0+1 和三核。两核+单核控制只使用
mask3/mask4，不把位组合当成任意合法核心集合。

证据：[双视角轮换测量](../results/smolvla_libero/vision_probes/dual_view_paired.json)、
[负载与频率采样](../results/smolvla_libero/vision_probes/dual_view_load.json)。
共享 `PackedVision` 构造重构后另跑两轮单视角回归，完整层输出和图像变体
检查通过，parallel/full Attention 最大差 0；该两轮只作正确性回归，不混入
原 20 轮正式单视角结果。

### 第 12 层阶段竞争与互补：简单交错未找到收益

使用实际完整推理留下的第 12 层 hidden、Q/K/V、Attention 输出。两视角输入
仍为合成 fixture/翻转；每种计划轮换执行 20 次，包含调用、线程提交、同步，
不含输入/输出 host 交接。重复 final-layer producer/consumer 不破坏其输入
hidden 缓冲区。各计划经过 consumer 输出位一致检查，不是闭环或完整调度结果。

| 相同工作量的阶段组合 | 串行中位数 | 并发中位数 |
|---|---:|---:|
| 两视角 QKV，各 mask7 | 8.698 ms | 8.890 ms |
| 两视角 QKV，并发 mask1/mask2 | 8.698 ms | 7.622 ms |
| 两视角 Attention，各 4+4+4 heads | 36.783 ms | 36.795 ms |
| 两视角 consumer，各 mask7 | 38.102 ms | 38.074 ms |
| 两视角 consumer，并发 mask1/mask2 | 38.102 ms | 36.139 ms |
| 两视角 consumer，并发 mask3/mask4 | 38.102 ms | 37.946 ms |
| 右 Attention + 左 consumer（mask7） | 37.379 ms | 37.314 ms |
| 右 Attention + 左 consumer（并发 mask1） | 37.379 ms | 50.801 ms |
| 右 Attention + 左 consumer（并发 mask3） | 37.379 ms | 39.899 ms |

这里 Attention 始终保留三组各自绑定单核的 h4 图；consumer mask 指的是另一
视角整个 projection/residual/Norm2/MLP/residual 图。最后两行改变了 consumer
可用核数，不是同一个核分配方案下的纯串行/并发比较。两视角 consumer 的
mask1/mask2 有约 1.96 ms 的小收益，但限制单个 consumer 核数去与 Attention
重叠明显更慢。因此目前不能照搬“Attention 与 MLP 资源互补”的说法。

这不穷尽所有调度，也不证明硬件禁止 overlap；它否定了当前最直接的阶段
互补假设。未知的底层排队、共享内存流量、不同 kernel 的资源使用需单独测量。
在没有新的机制证据前，暂停建设双视角协调器。

下一处候选是 consumer 内部的并行性边界：已有原始 profile 的 FC1+activation
约 87.23 ms/12 层，单核且未随 mask7 加速；FC2 则从单核约 266.06 ms 降到
三核约 108.76 ms。先独立测准确的 FC1/激活边界与原生接口，尝试按输出通道
分区 FC1+逐元素激活、直接写入完整 FC2 所读的共享 activation buffer，保留
所有通道、权重和求和项。这只是实验候选；分图成本、布局、编译 kernel 改变
及新的带宽竞争可能完全抵消收益。不能预先相加理论节省，也不能仅凭 FC1
映射单核断言 compiler 缺陷。

证据：[初轮阶段控制](../results/smolvla_libero/vision_probes/stage_pairs.json)、
[补充跨阶段控制](../results/smolvla_libero/vision_probes/stage_pairs_mixed.json)、
[补充轮频率采样](../results/smolvla_libero/vision_probes/stage_pairs_mixed_load.json)。
补充轮测量期 165 点 NPU/DRAM 为固定 1.0 GHz/2.112 GHz，细粒度 busy 不作
定量解释。

## 5. 已排除或降级的方向

### B2 常量膨胀：真实，但不是大速度机会

真实视觉 mask 在当前固定全有效几何下恒为零，形状 `[2,1,1024,1024]`。
提取实际图并移除 12 个 score-mask Add 后：

| SDK allocation 字段 | 原 B2 | 等价零 mask 修复 |
|---|---:|---:|
| weight region | 802625856 bytes | 198646080 bytes |
| internal | 51154944 bytes | 51154944 bytes |
| DMA allocated | 1714794496 bytes | 506728448 bytes |

weight-region 差值为 `12 * 50331648`；每层差值恰等于
`2 * 12 * 1024 * 1024 * 2 bytes`，即完整 FP16 score-shaped 大小。
实际 attention 子图也独立复现这一差值；标准 optimization level 3 仍保留它。
这定位了额外分配的 mask-lowering 路径，但 allocation query 不是物理访存轨迹。
不能将这些字节解释成学习权重复制、峰值 RSS 或每次推理 DDR 流量。

完整 B2 的交替测量只有 1739.01 -> 1722.39 ms，约 0.96% 改善；测试图像输出
位一致。因此作为内存修复和强基线保留，不能拿“省 604 MB”推导大幅加速。
独立 toy SDPA 的零 mask 不复现这个异常，图模式是必要条件。
证据：[allocation 控制](../results/smolvla_libero/vision_probes/b2_memory_causal_control.json)、
[完整图配对](../results/smolvla_libero/vision_probes/b2_full_unmasked_pair.json)。

### CPU Attention：尚无 backend 反转

四 A76 核、四线程 CPU standalone FP32 SDPA 为 40.98 ms，Torch FP16 为
413.13 ms；native NPU 全头约 41.55 ms，6+6 heads 约 24.24 ms。
CPU FP32 已没有明显整包优势，还未计入原生布局交接。不同精度不能作为等价
加速对比，Torch FP16 的慢也不能当成 ARM 硬件上限。

CPU softmax 分项约 16-21 ms，但没有 NPU 对应分项与完整交接的证据，不能
据此宣称矩阵/NPU、归约/CPU 是最佳分工。暂缓 C03/C08。

## 6. 论文逻辑审查

若后续成立，定位为 **Technique**，不制造“首次发现 head 可并行”的新问题。

| 环节 | 当前能够支撑的内容 |
|---|---|
| 背景 | 精确端侧 VLA/VLM 视觉推理；SmolVLA 双视角编码占 RK 延迟约 54%。 |
| 局限 1 | batch/core mask 的模型级选择没有让当前融合 Attention 有效跨核。 |
| 局限 2 | 朴素 head 拆分的边界处理使独立 probe 慢于原融合 kernel。 |
| 关键想法 | 联合设计执行分区和原生布局，使独立 head 执行不必变成 host 张量交接。 |
| 挑战 1 / 对应机制 | head 分区在语义布局中不连续；用兼容 packed 接口与经检查的 FD-offset views。 |
| 挑战 2 / 对应机制 | 双视角强基线将收益压缩至约 4%-6%；同阶段和 Attention/consumer 交错未显著改善。不能把资源竞争推断当成已定位的根因，暂停复杂协调。 |
| 候选贡献 | 结构性测量、可验证的布局/执行接口构造、完整视觉资源分配策略及端到端评测；后两项仍缺证据。 |

一致性审查：挑战 1 的机制已有完整 12 层原型；挑战 2 的简单方案已遇到负
结果，还没有系统方案，所以完整论文骨架尚不闭合。不得提前把它写成已实现
的运行时或端云协同系统；跨阶段互补没有成立，不能借负结果制造新动机。

Orca 的 selective batching、Rammer 的 inter/intra-operator 规划、CoDL 与
HeteroInfer 的设备分区、llada.cpp 的 NPU layout 都是必须对齐的近邻，见
[文献与反证记录](IDEA_TASTE_EXPANSION.md)。新意不能只剩 head split、zero-copy
或几个 core mask；必须证明接口/布局规律、端到端资源协调和跨实例可复用性。

## 7. 下一轮优先级与判废条件

1. **P0：consumer 内部边界，不先造调度器。** 双视角收益约 4%-6%，简单阶段
   并发与 Attention/consumer 交错没有明显收益，暂停双视角协调器。测 FC1+
   激活通道分区与完整 FC2 的原生数据通路；若净收益不成立，保留单视角机制，
   不扩写为多视角系统主贡献。只有规律跨形状/模型成立才讨论论文范围。
2. **P0：实际输入与 VLA 集成。** 接入常驻 policy 推理，不计一次性初始化
   为每帧工作；计入输入准备与最终特征交接，测试真实观测与层间数值边界。
3. **P1：阶段粒度为何变化。** consumer 已成为略大的阶段。测 heads/view
   分配下各阶段时延、重叠区间、
   各核 busy、CPU 提交成本及共享 DRAM 竞争。必须用真实 bandwidth/算量
   标定 Roofline；SDK RW 和 busy 不能替代带宽或 MAC occupancy。
4. **P1：可迁移规律。** 测 token/head 几何、至少另一视觉编码器，以及热稳态、
   持续请求的 p50/p95、内存驻留成本。否则只是固定 checkpoint 的配置修补。
5. **P1：VLA 闭环。** 固定输入协议、任务与随机种子，比较 connector、动作
   和成功率。视觉收益必须最终进入相同推理协议下的端到端结果。

端云、投机执行与 CPU/NPU tile 协作暂缓。当前先回答一个问题：**能否在不
改变模型工作量的前提下，让可用并行性穿过 NPU 图边界，并在真实双视角 VLA
关键路径上产生超过现成强基线的收益？**

## 8. 复现入口

- `scripts/extract_smolvla_vision_probes.py --native-interfaces --partition-heads 4`
  从实际 ONNX 提取子图，检查 mask，验证 FP32 等价性；heads=6 为两路控制。
- `scripts/convert_smolvla_denoise_rknn.py` 默认 optimization level 0，支持
  `--optimization-level 3` 编译器控制；模型文件保留在 cache/board，不入 Git。
- `scripts/probe_smolvla_native_block.cpp` 在 RK 链接 librknnrt，运行
  `native_block MODEL_DIRECTORY 20 OUTPUT_JSON 3`；最后参数 2 为两路。
- 原始实验数据在 `results/smolvla_libero/vision_probes/`，含参考 manifest、
  SDK profile、CPU 分项、native SDPA 和完整 block 的轮换测量。
- `scripts/export_smolvla_native_vision.py --onnx PATCHED_ONNX --input INPUT_NPY
  --output-dir OUTPUT_DIR` 导出完整 12 层 producer/consumer、共享 Attention、
  stem、tail、融合控制、audit 图及 FP32 fixtures。
- `scripts/convert_smolvla_native_vision.py --directory OUTPUT_DIR` 在 Toolkit
  环境逐图编译并保留 build logs，包含同源 patch 修复融合控制
  `vision_original.rknn`。SDK 某些图会输出 REGTASK 范围警告；不能只看 export
  返回码，必须执行上述输出与状态验证。
- `scripts/probe_smolvla_native_vision.cpp` 与 `rknn_native_graph.hpp` 在 RK 编译，
  运行 `native_vision MODEL_DIRECTORY 20 OUTPUT_JSON`。生成模型和大型二进制
  fixtures 保留在 cache/board，不入 Git。
- `scripts/smolvla_packed_vision.hpp` 共用完整 12 层的构造与运行路径；
  `probe_smolvla_native_dual_view.cpp` 运行 `native_dual_view MODEL_DIRECTORY
  12 OUTPUT_JSON`，`probe_smolvla_native_stage_pairs.cpp` 运行
  `native_stage_pairs MODEL_DIRECTORY 20 OUTPUT_JSON`。用
  `sample_rknpu_load.py --output LOAD_JSON -- COMMAND...` 记录标记、频率和 driver busy。
