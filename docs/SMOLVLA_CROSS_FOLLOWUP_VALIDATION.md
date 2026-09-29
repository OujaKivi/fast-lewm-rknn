# Cross 边界续验：数值定位、强并行对照与原生位复制

最新观测驱动重放、完整策略重要性与三轮 500 次尾部诊断见
[策略重放与尾部验证](SMOLVLA_POLICY_REPLAY_VALIDATION.md)。局部增量仍成立，
但每个 block 十次只省约 1.2 ms；不能扩写为完整策略的 10%。本页保留原会话口径。

2026-09-29。最新结论取代 [初轮 C17](SMOLVLA_CROSS_BOUNDARY_VALIDATION.md)
中的“下一关尚未完成”：原生 head 并行先击败旧方案；再测单向恢复和双向
CPU native tile 桥接。**最终保持原 Q/RoPE/输出权重，十次消费约 10.28 ms，
较 level-3 head 强对照再低 10.3%-10.7%**，见第 8 节。
最终独立 512-case 检查逐位通过，但 p95 不稳定，不能把中位数改善当尾延迟保证。
没有替换 C06 部署，没有接全部 32 层，也没有完整 VLA/闭环收益声明。

## 1. 口径不变

同一真实 SmolVLA LIBERO layer 1，包含 RMSNorm、Q 投影、RoPE、全部
15 Q heads / 5 KV heads 的 cross-Attention、输出投影与 residual。
Q50 / K149 / head64，输入输出 `[1,50,480]`；不含 MLP。
输入仍来自原 checkpoint 对合成 prefix 的计算，不是实际机器人图像。

十次消费为交替的两个独立 hidden 输入，**不是十步依赖 Euler**。
每序列一次准备、每次 hidden 打包/同步、所有图提交、并行派发/等待，以及
一次最终读回均计入；公共 raw-prefix native 打包仍等额排除。
不减 head/token/步数，不降低原 RKNN FP16 精度，不训练。

首次完整位恢复对照为 S7/S8/S9：每个方案、复用数各 40 次轮换，独立进程启动；
不混池。普通 ready 和旧 grouped-ready 控制保留 level-3 编译；新增图为
level 0。17 种模式、33 个常驻 context 同场比较，不是隔离 peak RSS。
S1-S6 保留为逐步加入消融的历史。S10-S12 加强编译控制；最后 S13-S15
含 19 种模式，仍为 33 个 context，第 8 节记录最终胜者，不混池或挑 session。

## 2. 数值差异定位到输出投影的表示

独立 CPU ONNX Runtime，关闭图优化，在原输入、另一 suffix、改变 prefix
三个输入上，捕获 Q/RoPE、Attention 输出与最终残差输出：

| 检查 | 结果 |
|---|---|
| 传播布局的 Q/RoPE，恢复原 head 顺序后 | 与原图逐位一致 |
| Attention 输出，恢复原通道顺序后 | 与原图逐位一致 |
| 重索引输出权重的最终结果 | 有 2.38e-7 至 4.77e-7 差异 |
| 保留原 Q 路径、仍重索引输出权重 | 差异保留 |
| 仅恢复原输出通道和原输出权重 | 逐位一致 |

因此 CPU 上差异首次出现在输出投影的浮点累加顺序，而不是 RoPE。
NPU 上的独立编译消融也支持这个定位：恢复原输出路径通过逐位检查，
只恢复原 Q 路径则没有修好。它不是直接捕获融合 NPU 内核的中间累加器，
不进一步声称已证明内部每一条指令的差异来源。

扩大到 512 个固定种子 929 的 hidden/key/value 扰动，标准差 0.05：

- 旧权重重索引方案仅 260/512 个样本逐位一致，最多 4 个输出元素不同，
  最大差 `0.001953125`。初轮 12 个扰动不足以描述这个差异范围。
- 恢复原输出顺序、原生 head 并行、两个紧凑并行控制，以及最终 CPU 位复制
  路径均为 512/512 逐位一致。
- 固定 fixture、另一 suffix、prefix 改变/恢复亦通过；每轮计时输出对各自
  anchor 逐位确定。prefix 改变确实影响输出，恢复后 original anchor 复原。

512 个扰动包含先前 64 个，不把重复 session 当作更多独立输入。
这是已测单块的数值证据，不是全部输入的证明、完整 flow 或闭环质量测试。

## 3. 原生 head 并行强控制确实更好

保留原 Q/RoPE 和输出投影：producer 输出 head-major Q；15 heads 分成
5+5+5，分别在 NPU core mask 1/2/4 上执行。使用三个常驻 CPU 工作线程
派发，不在每个 token 上创建新线程，派发与 barrier 均计入。

Q、expanded ready K/V 和 Attention 输出使用严格核验的 FD-offset view。
输出写入同一个原生消费缓冲区的互不重叠区域，不经 CPU 取回/拼装 Q/K/V。

开始时不能直接绑定：producer 的逻辑 Q50 在 native 输出中物理 width52，
而原 consumer 的输入 width50。核验发现后，消费图显式接 width52 并 Slice
回 50。K/V 同样物理 width152、Slice149；padding 不参与 Attention。
没有放松布局检查，也没有把 padding 当成额外有效 token。

十次消费总中位数，ms：

| 完整边界方案 | S7 | S8 | S9 |
|---|---:|---:|---:|
| 原 block，重算条件 | 19.086 | 18.914 | 19.033 |
| 普通 expanded ready 整块 | 15.351 | 15.600 | 15.344 |
| 旧 C17：传播布局 + compact ready，未恢复原输出累加顺序 | 13.633 | 13.623 | 13.622 |
| 传播布局 + 恢复原输出顺序 | 14.298 | 14.277 | 14.326 |
| 恢复原 Q、仍重索引输出权重 | 26.520 | 26.488 | 26.377 |
| 原 Q 和原输出顺序均恢复 | 27.383 | 27.338 | 27.321 |
| 原 Q/O，分图完整 15-head Attention | 16.886 | 16.802 | 16.945 |
| 原 Q/O，三个 5-head 图串行 | 17.989 | 18.014 | 18.036 |
| **原 Q/O，三个 5-head 图并行** | **11.518** | **11.530** | **11.524** |
| 传播布局 compact，完整 5-group Attention，编译器恢复输出 | 15.684 | 15.694 | 15.662 |
| 同上，2+2+1 group 串行 | 16.829 | 16.810 | 16.844 |
| 同上，2+2+1 group 并行 | 12.818 | 12.787 | 12.824 |
| 保留原 head 排列 compact，完整 5-group Attention | 20.344 | 20.336 | 20.393 |
| 同上，2+2+1 group 串行 | 20.423 | 20.353 | 20.527 |
| 同上，2+2+1 group 并行 | 13.647 | 13.590 | 13.626 |
| compact 完整 5-group Attention + CPU 原生位恢复 | 13.736 | 13.755 | 13.695 |
| **compact 2+2+1 group 并行 + CPU 原生位恢复** | **10.930** | **10.907** | **10.917** |

普通 head 并行比普通 ready 低 25%-26%，也比旧 C17 低约 15%。
所以旧 C17 不能再按“最优完整边界”推进，且其权重重索引没有通过数值门槛。
一次消费也是比较相同的完整边界：S7/S8/S9 的 head 并行为
1.841/1.846/1.847 ms，最终位恢复为 1.680/1.682/1.677 ms。

## 4. 为什么两个紧凑并行组合仍输

S7 的十次消费分项中位数，ms。分项中位数不保证恰好加和成总中位数。

| 方案 | 准备 | Q producer | Attention 与并行等待 | 输出消费 | CPU 位恢复及同步 | 总计 |
|---|---:|---:|---:|---:|---:|---:|
| expanded head 并行 | 0.667 | 3.612 | 5.076 | 1.789 | 0 | 11.518 |
| compact 传播布局、编译器恢复输出 | 0.549 | 4.423 | 3.581 | 3.898 | 0 | 12.818 |
| compact 传播布局、CPU 位恢复 | 0.547 | 4.425 | 3.584 | 1.820 | 0.183 | 10.930 |

传播布局确实让 Attention 更快，但 producer 更贵；编译器输出恢复还吞掉了
更多收益。保留原 head 顺序的另一控制减轻输出端，却把重排放回 Attention
内部，S9 十次 Attention/等待为 7.300 ms，仍慢于 expanded 的 5.073 ms。

另外的 fixed-zero / 普通 API / SDK profile 只解释编译机制，不充作 native
生产时延：传播布局的 group consumer 有 CPU reshape，报告 403 us，
并有 `(960,960,1,1)` 恢复 Conv。普通 head consumer 的 weight region
923904 bytes，group consumer 为 2767168 bytes，增加约 1.84 MB。
这是编译恢复工作，不是原模型新增学习层；未测真实 DDR/MAC。

原 head 顺序的 compact Attention 也引入恢复 Conv；2-group 图中为
`384×384`，还有 190 us reshape、132 us transpose，不能只拿
262 us Attention 内核宣传加速。

## 5. 最后补测的微机制：不做浮点运算的恢复

完整 group 输出与原 head consumer 的 native `NC1HWC2` 布局中，C8 lanes
已经一致。布局差别落在 group/shared-head 与 query 行的组合方式上：

- 源为 `[1,40,1,152,8]`，有效宽度 `3×50`。
- 目标为 `[1,120,1,52,8]`，每 head 有效宽度 50。
- 5 groups × 8 channel blocks × 3 shared heads = **120 个连续 tile**。
- 每 tile 复制 `50×8×2 = 800 bytes`，共 96000 bytes/次。

CPU 直接复制 native FP16 存储位，再调用**原顺序的输出权重**。
没有 FP16->FP32->FP16 转换，没有重新计算数值，也没有改 dot 的累加顺序。
FROM_DEVICE、所有复制、TO_DEVICE 全部计入；没有取消 SDK 默认同步。
S7/S8/S9 十次复制与同步分别约 0.183/0.183/0.187 ms。

本路径绕过了原 group consumer 的 CPU reshape/恢复 Conv，不再因消除
该 Conv 而重索引输出权重。**NPU 仍做 Attention 和输出矩阵计算，CPU 仅做
准确的存储排列恢复。** 它不是 CPU softmax，也不是 CPU/NPU 并行流水。

相对最好的 expanded head 并行，最终总中位数降低
**5.11% / 5.41% / 5.27%**，paired median gain
0.590/0.620/0.606 ms，逐轮更快 39/40、34/40、38/40。
相对同一 compact 并行但编译器恢复输出，约降低 14.7%-14.9%。
只用 CPU 位恢复而不做 group 并行为约 13.7 ms，因此不能把并行收益
全归给这个恢复机制，或只比较更差的编译器恢复控制。

紧凑条件仍为本层 194560 native bytes，expanded 为 583680 bytes。
这里是状态/SDK 单图记录，不是 peak RSS、实际 DDR 流量或能耗。

## 6. 不能遗漏的反面：尾延迟

最终十次消费 p95，ms：

| 方案 | S7 | S8 | S9 |
|---|---:|---:|---:|
| expanded head 并行 | 12.423 | 11.669 | 15.939 |
| compact 并行 + CPU 位恢复 | 11.106 | **16.097** | 15.699 |

S8 明显恶化，不能声称尾延迟也改善。其慢样本中位恢复本身约 0.182 ms
没有明显变大，producer/Attention/consumer 都有增加；不能未经诊断归因于
复制、工作线程、热降频或外部负载。没有删除这些样本或换 session。

采样 NPU 固定 1 GHz，DRAM 2.112 GHz；未连续记录 CPU 频率/温度，
没有长时热稳态、多设备或能耗验证。驱动 busy 也不是算力利用率。

## 7. 阶段性判断：单向恢复

保留的新证据是：**数值顺序、算术几何和物理交接不能独立优化**。
重索引权重省恢复工作，但改变浮点累加；保留原顺序让恢复变成编译矩阵；
在本层的 native 布局下，纯位复制让 compact 内核与原输出顺序共存。

这是比“头拆开并行”更具体的机制增量，但目前只有一个形状、约 5% 超强对照
中位数收益，不能单独宣称 CCF-A 主线。CPU 转置、GQA、head 并行、native
interop 本身也不是新颖性声明。此时不建设通用 scheduler、不强塞端云模块。

下一关是同一真实观测下的优化后完整策略 trace 与数值；先查该边界在完整
VLA 中是否重要，以及尾延迟来源。再选择真实层/形状验证这个机制的适用域。
32 层集成仍需完整自注意力 mask、动态 K/V 和 FP32 Euler 契约，不能直接
复制这个 cross-only 图。C08 与端云 C04 继续独立保留，不是自动替代品。

## 8. 最后的反证：上游也不必强扭成新表示

先把两个可比方案的 producer、consumer、Attention 等八个图提升到
optimization level 3：`head_producer/head_consumer/head_attention_5/15`，
`group_producer/group_attention_1/2/5`。S10-S12 各 40 次：head 强对照
11.498/11.483/11.500 ms，单向恢复 10.898/10.893/10.867 ms。
编译控制没有吸收其 5.1%-5.5% 增量；64 个扰动在这三轮均逐位一致。
其余旧消融图没有全部升级，不能说穷尽了所有编译可能。

但单向恢复路径的 Q producer 仍比原 head producer 慢约 0.08 ms/次。
原 head-major Q 已具有相同的 C8 lanes：**同一 120-tile 映射反过来，就能
把原 Q 精确打包成 group Attention 输入**，无需重索引 Q 权重，也无需
把新表示贯通 RMSNorm/RoPE。输入和输出各复制一次，所有同步均计入。

最后的执行路径因此为：

`原 Q/RoPE → CPU native 位打包 → NPU compact group 并行 → CPU native 位恢复 → 原输出投影/residual`

这里 Q 和输出投影都保留原权重、原算术路径。两个复制仅重排存储位，
Attention 仍处理全部原查询/条件；150 行只是 3 个共享 head 各 50 行的打包，
不是新增或删去动作。条件继续一次准备、十次消费。

最终 level-3 配对结果，S13/S14/S15 每方案各 40 次，ms：

| 方案 | S13 | S14 | S15 |
|---|---:|---:|---:|
| 普通 expanded ready 整块 | 15.418 | 15.357 | 15.342 |
| **level-3 expanded head 并行强控制** | **11.463** | **11.502** | **11.500** |
| compact 并行，CPU 仅恢复输出 | 10.865 | 10.889 | 10.913 |
| compact 完整 Attention，CPU 双向桥接，无 head 并行 | 13.118 | 13.116 | 13.088 |
| **compact 并行，CPU 双向桥接** | **10.281** | **10.269** | **10.290** |

相对同轮 head 强控制，中位数降低 **10.31% / 10.73% / 10.52%**，
paired median gain 1.179/1.230/1.197 ms，逐轮更快 37/40、39/40、38/40。
相对单向恢复再降低 5.38%-5.71%。不将它与串行完整 Attention 的更大
差距当作新复制机制的全部贡献。

S13 十次分项：原 Q producer 3.630 ms，compact Attention/等待 3.569 ms，
双向复制/同步合计 0.397 ms，原输出消费 1.791 ms，准备 0.540 ms。
对应单向路径 producer 4.440 ms、复制 0.187 ms；即多付一次小复制，
反而避免更昂贵的 producer 表示处理。分项中位数不直接相加作总时延。

一次消费最终路径为 1.612/1.603/1.611 ms，同轮 head 控制为
1.844/1.833/1.841 ms。十次消费 p95 仍不稳定：

| 方案 | S13 | S14 | S15 |
|---|---:|---:|---:|
| expanded head 并行 | 13.303 | 14.525 | 16.351 |
| compact 并行 + 双向位桥接 | **14.029** | 13.823 | 15.486 |

S13 更差，不宣传尾延迟胜利。另起进程、启用相同 level-3 控制的最终独立
512-case 审计：双向并行、双向串行及 head 强对照均 512/512 逐位一致；
固定输入、另一 suffix、prefix 改变/恢复也一致。不用旧单向路径的测试代替它，
也不把这 512 个样本当完整策略或所有输入的证明。

**对初始 insight 的修正：** 不必强迫矩阵生产者直接生成所有后续布局；
在本层，保留原算术路径，并让 CPU 用原生存储位接通 NPU 的合适计算几何，
比全链表示传播更好。它是已测的选择，不是“CPU 转置必然胜过 NPU”的通则。
是否构成论文仍取决于真实策略重要性、适用域、尾延迟与近邻机制增量。

## 9. 证据与复现

- [S7](../results/smolvla_libero/cross_followup_probes/session7.json)、
  [S8](../results/smolvla_libero/cross_followup_probes/session8.json)、
  [S9](../results/smolvla_libero/cross_followup_probes/session9.json)、
  [配对审计](../results/smolvla_libero/cross_followup_probes/paired_audit.json)。
- [独立 CPU 阶段审计](../results/smolvla_libero/cross_followup_probes/fp32_audit.json)、
  [512 个扰动](../results/smolvla_libero/cross_followup_probes/numerical512.json)、
  [图边界](../results/smolvla_libero/cross_followup_probes/manifest.json)、
  [源图 hashes](../results/smolvla_libero/cross_followup_probes/sources.json)。
- [最终 S13](../results/smolvla_libero/cross_followup_probes/session13.json)、
  [S14](../results/smolvla_libero/cross_followup_probes/session14.json)、
  [S15](../results/smolvla_libero/cross_followup_probes/session15.json)、
  [最终配对审计](../results/smolvla_libero/cross_followup_probes/final_paired_audit.json)、
  [八图 level-3 审计](../results/smolvla_libero/cross_followup_probes/opt3_paired_audit.json)、
  [最终双向 512-case 审计](../results/smolvla_libero/cross_followup_probes/numerical512_opt3_both.json)。
- [group consumer profile](../results/smolvla_libero/cross_followup_probes/group_consumer_profile.txt)、
  [2-group 原 head 排列 profile](../results/smolvla_libero/cross_followup_probes/original_order_attention_2_profile.txt)。
- 脚本：`extract_smolvla_cross_followup.py`、`smolvla_native_cross_followup.cpp`、
  `probe_smolvla_cross_followup.py`、`analyze_smolvla_cross_followup.py`。
  followup library 复用原边界代码，编译命令需要同时提供该文件与 native helper。
- 本机制的双向 byte shuffle 写在 `copy_native_tiles(bool)`；支持形状严格检查，
  不将本层固定布局错误推广为任意 native 张量格式。
- 所有正式负载命令、clock 样本见对应 `_load.json`；模型与 library hashes
  见 session。CPU 验证关闭 ORT 图优化。编译沿用 Toolkit2.3.2，无量化。
  `SMOLVLA_FOLLOWUP_OPT3=1` 只选择上列八张 `_opt3.rknn`，已在 session metadata
  中标明，不改变旧消融的编译等级；最终三轮均启用。
- Mac 制品在 `/Users/wangjiwei/.cache/smolvla-cross-followup/`，板端在
  `/root/smolvla-cross-followup/`；原模型与 C06 部署未替换，无新常驻服务。
