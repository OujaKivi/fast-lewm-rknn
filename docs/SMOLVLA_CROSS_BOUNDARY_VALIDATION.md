# C17：完整 cross-Attention 边界验证

2026-09-29 续验已完成，最新判断见 [数值、强并行与位复制续验](SMOLVLA_CROSS_FOLLOWUP_VALIDATION.md)。
原生 head 强对照击败本页旧候选；最终 CPU 双向原生位桥接较匹配 level-3
head 控制中位数改善 10.3%-10.7%，独立 512 个扰动逐位一致，但尾延迟不稳定。下文保留初轮口径，
不能再把数值定位/head 强对照理解为尚未执行。

2026-09-29。结论：**动态布局传播与紧凑消费就绪条件一起，得到局部性能正结果；
但测试存在少量末位差异，尚未通过严格逐位数值门槛。** 不替换现有 C06 部署，
不扩展到全部 32 层，不宣称完整 VLA 加速或已形成论文主线。

上一轮 [C14](SMOLVLA_CONDITIONING_VALIDATION.md) 从已经生成的 Q 开始。
本轮包含真实 Q 生产和输出消费，回答的是不同的边界问题。

## 1. 实际测试了什么

- 同一 `HuggingFaceVLA/smolvla_libero` checkpoint 的 layer 1。
- 输入是 layer 0 之后的 hidden，包含 layer 1 的 RMSNorm、Q 投影、完整 RoPE、
  cross-Attention、原输出投影及 residual add；输出接下一个 post-attention norm。
  **不包含 MLP，故不是整个 Transformer layer，也不是完整 flow。**
- 输入/输出都是 `[1,50,480]`；15 Q heads，5 K/V heads，每 head 64 维，149 prefix tokens。
- activations 由原 checkpoint ONNX 前缀计算产生；prefix 仍为导出 fixture 的合成
  条件，不是实际 LIBERO 图像。使用另一 suffix、改变 prefix、恢复 prefix，以及
  12 个固定种子的 hidden/prefix 扰动，避免只检查一个固定输出。
- 原 RKNN FP16 精度不降低；全部 head/token 保留。表示传播版本只等价重索引
  Q/O 投影的权重存储和相邻操作，不重新训练，不改变学习参数的数学作用。
- “十次消费”是同观测下十个交替的独立 hidden 输入，**不是十步依赖 Euler 轨迹**。
  本轮没有 CPU latent/Euler 更新，没有闭环成功率测试。

所有模式的 hidden native 打包、显式同步、run 和一次最终读回均计入；缓存
准备每个序列做一次并计入。共享的 raw prefix native 打包另报。图间采用经
format/dims/stride/bytes 核验的 native FD view，保留 SDK 默认同步。
分拆版本的三次图提交也全部计入，没有通过 CPU 取回 Q/K/V 再拼接。

## 2. 自动融合不够，确实需要改变表示

本轮实现九种对照：原重算、普通 expanded ready、compact repeat、grouped
整块、grouped 三图分拆、另一种 query 行排列、动态表示传播，以及两种 group-ready 消融。

表示传播版本让 Q 的通道按 `(KV group, head dimension, shared query head)`
排列，而不是先固定为 token-major Q 后再重排。Q 投影的对应输出列重索引，
RoPE 在这个表示上保留原 sin/cos 和半维旋转，输出投影的输入行相应重索引。
这样避免为 Attention 的输出恢复默认通道排列，再立刻执行输出投影。

它不是“任意 token/head 转置都能靠权重解决”。这里只在构造出的通道表示上
传播静态 channel permutation；实际 reshape 和 native padding 整理仍保留、计费。
FP32 数学等价与浮点逐位等价分别核验，后者本轮没有全部成立。

静态条件另外比较：

1. 原 compact token-major K/V：每次消费仍需要转成 Attention 的布局。
2. compact group-ready K/V：一次准备成 5-head head-major 格式，物理 width 152，
   消费前 Slice 到原 149 tokens，padding 不参与 Attention。

加入“只做 compact group-ready、保持原 grouped Q/O”的消融，防止把普通
静态布局外提的收益错误归给动态表示机制。它仍然慢，不能单独解释最后的收益。

## 3. 完整边界的配对时延

RK3588，Toolkit/Runtime 2.3.2，CPU affinity 4-7。正式完整对照为三个独立
进程启动的 session，每个方案/复用数各 40 次轮换测量。S5 使用 optimization
level 0；S6/S7 将普通 ready 和 grouped-ready 消融提升到 level 3，其余图保持
level 0，检查合理的更强编译控制。level 3 没有吸收本轮的机制收益。

下表为十次消费总中位数，单位 ms；每行都含相应一次准备成本。

| 方案 | S5 / level-0 控制 | S6 / level-3 控制 | S7 / level-3 控制 |
|---|---:|---:|---:|
| 原 block，每次重算 K/V | 18.88 | 18.90 | 18.94 |
| **普通 expanded consumer-ready 强基线** | **15.27** | **15.44** | **15.42** |
| compact cache + 原 repeat | 21.30 | 21.37 | 21.23 |
| grouped 整块一起编译 | 29.00 | 29.38 | 29.10 |
| grouped producer/attention/consumer 三图 | 35.32 | 34.89 | 35.19 |
| grouped，改为 token/head 交错行排列 | 20.29 | 20.32 | 20.28 |
| 动态表示传播，但 static K/V 尚非消费布局 | 15.49 | 15.57 | 15.56 |
| 仅 compact group-ready，保留原 grouped 动态路径 | 27.23 | 27.46 | 27.24 |
| **动态表示传播 + compact group-ready** | **13.59** | **13.68** | **13.57** |

相对普通 ready，最后一行的总中位数降低 **11.01% / 11.37% / 12.02%**。
逐轮更快 40/40、38/40、39/40；paired median gain 为 1.686/1.751/1.849 ms。
20,000 次 session 内配对 bootstrap 的描述性区间分别为
[1.681,1.693] / [1.748,1.759] / [1.843,1.858] ms，不是跨任务/设备/热状态的结论。
这些区间不能代表实验之外的系统变异，独立 session 也没有混池。

S5 分项：普通 ready 准备约 0.681 ms，十次 run 合计约 14.238 ms；最终候选
准备 0.548 ms，run 合计 12.692 ms。以原始 JSON 的分项为准；分项中位数不
要求加和恰好等于总中位数。一次消费 S5 为普通 ready 2.233 ms、候选 1.952 ms，
原重算 1.998 ms；不要只挑比原方案弱的缓存控制。

测量采样的 NPU 均为 1 GHz，DRAM 均为 2.112 GHz；未连续记录 CPU 频率
和温度，没有长时间热稳态、能耗或多板验证。S1-S4 是逐步加入控制的历史数据，
保留但不混入正式三轮统计。

## 4. 瓶颈具体移动到了哪里

另外的 SDK profile 使用固定零输入/普通 API，含 profiling 开销，只用于解释
编译机制，不作为正式 native 时延。记录中：

- 普通 ready 的 Q projection 与 Attention 之间 reshape 为约 0 us，Attention
  为 868 us；说明原完整块并没有支付孤立 probe 的 1206 us Q 重排。
- 原 grouped 整块中，Q 重排仍为 940 us，另有 258 us reshape 和 198 us transpose。
  “一起编译就自动消除转换”被反证，而不只是一个隔离输入 ABI 的假象。
- 原 grouped 的输出排列恢复被降低为额外 `(960,960,1,1)` Conv，名称包含
  `gather_2conv`，报告 88 us。grouped weight region 为 3696320 bytes，普通
  ready 为 1853568 bytes。原 ONNX 没有这个学习层；这是编译新增的恢复工作。
  未读取二进制权重内容、未测真实 MAC/DDR，不能进一步声称实际多做了多少硬件计算。
- 最终表示传播图不再有上述三个昂贵 Q adapter，也没有该额外 960->960 Conv。
  仍有 121 us 的 Q reshape、约 539 us Attention，以及 **171 us、Target=CPU 的
  输出 reshape**。它不是零拷贝到底，也不是纯 NPU block。
- compact ready 的两次 Slice 合计 52 us，普通 expanded ready 为 112 us。
  候选没有偷偷跳过 padding 整理。

因此需要**动态表示与静态准备同时匹配后续消费**。只合并图、换行排列或只
整理 K/V 均不能得到最后的结果。原 Attention 内核的几何收益仍在，但剩余
代价已转移到 native reshape/边界，不能把更细 overlap 当作必然收益。

## 5. 数值门槛：尚未闭合

独立 ONNX Runtime CPU、关闭图优化，验证原输入、另一 suffix、改变 prefix：

- 原/普通 ready/原 grouped/交错 grouped/静态 group-ready 消融均逐位一致。
- 两种表示传播图相对原图的 FP32 最大差为 2.38e-7 至 4.77e-7，native adapter
  对各自原始逻辑图逐位一致。该小差异与算术重排相容，但本轮未单独定位每个差值来源。

NPU 正式三轮：

- 普通 ready 与各未做权重表示传播的控制，在固定 fixture、切换/恢复和 12 个
  扰动样本上均与 original block 逐位一致。
- 最终候选：固定原输入有一个输出元素不同，最大差 **0.00048828125**；另一
  suffix 逐位一致；改变 prefix 最大差 **0.00006103515625**，也是一个元素不同。
- 12 个合成扰动中 9 个逐位一致，3 个存在一个元素不同，最大差 0.00048828125。
  一个输出共有 24,000 个元素，但“只有一个元素”也不能证明多层/多步累积无影响。
- 改变 prefix 后输出确有变化，恢复原 prefix 后 original anchor 逐位恢复；
  每个模式的所有计时重复都相对自己的 anchor 逐位确定。

这不是有意少算或降低精度，但**不是原后端的逐位执行复现**。不将其标为已验证
无损，不直接用于完整 policy；原 C06 的严格十步 baseline 保持不变。

## 6. 存储收益和科研判断

本层条件的实际 native 输出：expanded ready 两支共 **583680 bytes**，compact
group-ready 两支共 **194560 bytes**，少 2/3。对应 prep weight region 为
1231104 / 411904 bytes。最终候选 consumer weight region 为 1853632 bytes，
并未为新的动态表示增加另一个学习矩阵。

十四个 context 同时加载，以上是 SDK 单图/单状态数据，不是隔离部署 peak RSS。
不能把存储变小直接换算成 DDR 流量、能耗或在 16 GiB 板上的容量瓶颈。

**判定：C17 的局部性能机制得到支持，数值/完整策略/泛化门槛仍未通过。**
不同于 C14 的旧消费子图，不能再说所有 compact 路径都已被强基线击败；也
不能用这 11%-12% 单块收益宣称论文已成立。沿用
[规划中的近邻边界](NEXT_MECHANISM_PLAN_2026-09-28.md#4-本轮文献自审)，
GQA、静态 cache、layout propagation 和权重重索引本身都不直接主张新颖。

接下来只保留两个窄问题，不马上接全部 32 层：

1. 定位表示传播的末位差异，区分 RoPE 路径与输出 dot 的累加次序；验证是否
   存在保留原数值协议且仍有收益的实现。不通过容忍阈值把严格数值失败改写为成功。
2. 解释已测的 CPU reshape：核验其 native format/padding 与真实交接。50 行
   合并为 150 行时的物理整理是候选原因，尚未证明。可用只增加 dummy query、
   保留并最终取回全部原 50 行的 cross-only padding 控制作诊断；不能把它推广
   到会改变动态 K/V/causal mask 的 self 层，也不能先假定 reshape 可以免费 alias。

局部结果足以值得继续核验，不是靠新名字挽救旧失败；但收益重要性需要优化后
完整 VLA trace，近邻机制差异还需独立检索，当前不建立大调度框架。
此前规划中的原生 head 并行完整 block 对照尚未实现，必须作为下一关的合理
替代方案，不能据本轮结果宣称已胜过所有强控制。数值定位与该对照是先决条件，
不是将目前的单块时延比例直接搬到完整策略。

## 7. 证据与复现

- [正式 S5](../results/smolvla_libero/cross_boundary_probes/session5.json)、
  [level-3 控制 S6](../results/smolvla_libero/cross_boundary_probes/session6.json)、
  [独立重复 S7](../results/smolvla_libero/cross_boundary_probes/session7.json)、
  [配对审计与制品 hashes](../results/smolvla_libero/cross_boundary_probes/paired_audit.json)。
- [FP32 parity](../results/smolvla_libero/cross_boundary_probes/fp32_parity.json)、
  [边界 manifest](../results/smolvla_libero/cross_boundary_probes/manifest.json)。
- [原 grouped profile](../results/smolvla_libero/cross_boundary_probes/block_grouped_joint_profile.txt)、
  [最终 profile](../results/smolvla_libero/cross_boundary_probes/block_grouped_propagated_ready_profile.txt)。
- 源码：`extract_smolvla_cross_boundary.py`、`smolvla_native_cross_boundary.cpp`、
  `probe_smolvla_cross_boundary.py`、`analyze_smolvla_cross_boundary.py`，位于 `scripts/`。
  既有 native adapter 增加可选 hidden-width/head-count 参数，默认旧行为不变。
- Mac 制品 `/Users/wangjiwei/.cache/smolvla-cross-boundary/`；板端
  `/root/smolvla-cross-boundary/`。没有替换原部署图，没有留下常驻新服务。

提取、ORT 验证后，用现有 converter 对 `models/` 编译；两个 `_opt3.rknn` 分别
从 `block_ready.onnx` 与 `block_grouped_joint_ready.onnx` 以 optimization level 3
生成。编译 native library；原生 probe 参数和完整负载采样命令保存在各 `_load.json`。
设置 `SMOLVLA_CROSS_OPT3=1` 仅切换这两个对照图，不改变候选图精度或算法。
