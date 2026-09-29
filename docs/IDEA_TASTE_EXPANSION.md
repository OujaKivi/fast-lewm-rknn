# SmolVLA 系统路线：候选扩展与反证

更新：2026-09-28。本文是立题前的工作记录，不是已成立的创新声明。
动态排序与判废条件见 [IDEA_CANDIDATES.md](IDEA_CANDIDATES.md)。
最新五条复审及执行顺序见 [2026-09-28 复审](IDEA_REASSESSMENT_2026-09-28.md)；
下方各阶段记录保留当时判断，不以旧优先级覆盖最新复审。
本日最新执行结论：C09 native 常驻相对普通 native pass-through 无稳定大收益，
降为强基线；C13 实际 mask/GQA 后只留较小的 cross 线索。见
[Flow 调查](SMOLVLA_FLOW_INVESTIGATION.md) 与
[动作 Attention 调查](SMOLVLA_ACTION_ATTENTION_INVESTIGATION.md)，不以旧排序
推定两条已能立题。
后续 conditioning 实测现已完成：C14 紧凑/grouped 路径落败，普通消费就绪
缓存接完整十步约快 7.4%，作为 C06 强基线保留，不升格新主线。C04 的端云
payload 仍是独立审计；C15 只查首步 ceiling，C16 只留严格投机兼容性备选，
不换 SmolVLA。详见 [实际验证](SMOLVLA_CONDITIONING_VALIDATION.md)。

## 研究问题与边界

1. 相同模型计算为何在 NPU 上对应不同的实际工作量、内存占用和可用并行性？
2. 不预测动作、不减少去噪步数，哪些执行边界可以改变，哪些真依赖不能改变？
3. 在强全云基线存在时，端侧 NPU 能否承担有价值的独立工作，而不是人为增加端云往返？

主模型固定 SmolVLA，观测、动作数、十步 flow matching 保持不变。不以更低精度、近似复用或跳步获得收益。
等价图改写保证算法语义不变，不自动保证不同后端的浮点结果逐位相同。需要分别报告图内数值一致性、动作误差和闭环质量。
本轮按执行抽象、exact 投机、端云切口三种视角检索原始论文；前两种由独立研究子任务核验，最后结合本地依赖与数据审查。

## 新证据，以及没有成立的解释

| 对照 | 已测结果 | 能说明什么 / 不能说明什么 |
|---|---|---|
| 双视角串行 B1 / 普通 B2 / 双图分核 | 1746 / 1731 / 1364 ms | B2 没解决单观测双视角关键路径；不能仅由此声称新调度范式。 |
| B1 / B2 compiled weight region | 223.8 / 802.6 MB；ONNX initializer 的 shape/value hash 多重集合一致 | 编译后的常量区膨胀已确认；尚未定位具体张量，不能说复制了三份模型权重。 |
| 独立 Attention：12 heads / 6+6 heads 并发 | 74.5 / 45.6 ms；测试输出差异 0 | 独立 head 并行可兑现。但原完整视觉图中的融合 Attention 平均约 41.2 ms，直接插入这个外部探针反而可能更慢。 |
| Full Attention mask7 / mask1，同时运行 core1 的独立 companion | makespan 79.0 / 79.5 ms；companion 42.6 / 43.5 ms | **不支持“三核图一定锁住空闲核”。** 在此探针中，其他核上的工作能前进。 |
| Patch Conv / exact space-to-depth + 1x1 Conv | 同输入交替运行 874.4 / 814.4 ms | 编译工作膨胀可规避，但单项约 60 ms，不足以立题。FP32 fixture 输出差异 0；两个 FP16 图之间并非 bitwise 相同。 |
| Standalone B2 Attention，无 mask / zero mask / shape-derived zero mask | compiled weight region 均为 384 bytes | **batch 或零 mask 本身不是 full-vision 常量膨胀的充分解释。** 必须提取真实子图，不继续凭直觉归因。 |

原始数据：[双视角](../results/smolvla_libero/smolvla_camera_batch2.json)、[compiled memory](../results/smolvla_libero/smolvla_compiled_memory_probe.json)、[head 并发与 reservation 反证](../results/smolvla_libero/smolvla_attention_heads_reservation.json)、[patch 交替对照](../results/smolvla_libero/smolvla_vision_patch_pair.json)。
这些不是 LIBERO 闭环或多设备泛化实验。Patch 与双图并发组合达到约 1.172 s 视觉时延，也不能直接宣称完整 VLA 达到相同比例加速。

## 从好系统论文借什么，不借什么

| 论文 | 真正值得借鉴的推导 | 对我们的约束 |
|---|---|---|
| [Lever](https://arxiv.org/html/2605.16786v1), 2026 | 动态、小 frontier 留在 CPU；形成可批量工作后转 NPU；只为最终接收路径物化必要结果。 | 借“执行形状改变后换机制”，不能捏造 SmolVLA 连续 latent 的 exact acceptance tree。 |
| [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf), OSDI 2022 | request 不是正确调度单元；共享参数算子与不共享参数 Attention 有不同 batching 价值。 | “线性层合、Attention 分”已有直接先例，必须进入基线。 |
| [Rammer](https://www.usenix.org/system/files/osdi20-ma.pdf), OSDI 2020 | 单个算子的局部最优可能妨碍整体并行，打通算子内外执行边界。 | “overlap 粒度太粗”本身是老问题，须证明 NPU 上具体新障碍与可执行解法。 |
| [LithOS](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/lithos_sosp25.pdf), SOSP 2025 | 已提交工作难以重分配，控制 outstanding work、atomization 与 per-kernel right-sizing。 | CUDA thread-block 控制不是 RKNN 默认能力；本轮也未证实 attention 的 graph-wide reservation。 |
| [MTS](https://zhouzimu.github.io/paper/secon22-wang.pdf), SECON 2022 | 语义上共享权重，不等于融合图保留共享收益；stitching/grouping 权衡常量与临时内存。 | compiled constants 膨胀必须定位成可重复机制，不能只展示一个大模型文件。 |
| [SpotServe](https://arxiv.org/pdf/2311.15566), ASPLOS 2024 | 保存已提交的推理进度，结合迁移时机继续执行，而非整个请求重跑。 | “每步 checkpoint”不是新点；需要明确 VLA 大 conditioning 与小 latent 的不对称，并计入前者成本。 |
| [llada.cpp](https://arxiv.org/html/2606.13740), 2026 preprint | tile 阶梯、native layout 和有限映射空间共同决定执行与驻留。 | 动态离散 token 与固定连续 flow 不同；借 layout/lifetime 分析，不直接借其算法变化。 |

共同的 taste：先找到一个被现有抽象隐藏的、能被强对照测出的损失，再改变执行单元、状态表示或提交边界。不是先命名 runtime，再去找三个模块。

## 五条值得保留的探索

### A. 融合保住了数据，却隐藏了可用并行性（C01）

**痛点直述：** 三核 NPU 上，视觉 Attention 仍主要在一个核执行；把 heads 拆开确实能同时跑，但从融合图取出张量、换布局、重新提交，会把节省的时间吃掉。

**问题本质：** 不是“要不要拆”，而是能否只暴露独立执行边界，不暴露昂贵的数据物化边界。当前 45.6 ms 外部并行探针仍比原融合算子的约 41.2 ms 慢，已经给出这个约束。

**候选机制：** 大投影保留编译器高效多核形态；独立 head/camera 归约使用可并发形态；中间值通过持久 native buffer、兼容布局和权重共享 context 继续执行，避免 CPU 往返。先研究一个完整 vision block，而不是立刻做通用 runtime。

**升格条件：** 需要发现可重复的“融合与并行边界错位”，且 native continuation 在多种 shape/模型上同时保住两者。Orca、Rammer、HeteroInfer 已覆盖基本思想，不宜声称新的 selective batching。

**下一关：** SDK shared-weight context + native I/O 强基线；完整 block 与整视角并发公平对照。若 native 边界仍比节省量贵，停止这条主线。

**2026-09-28 实验更新：** 完整实际 layer-0 block 已越过局部门槛。原生布局融合控制 62.98 ms，原生分图但全头 Attention 65.23 ms，原生分图并行 4+4+4 heads 为 42.24 ms。实现不是仅绑定原来的 rank-3 张量，而是将接口改写为 channel-major 4D，使 producer/consumer 与 head 分区具有兼容 NC1HWC2 布局，再用 FD-offset 视图交接。三组随机输入上，并行与原生全头分图输出一致；相对原 FP16 融合图存在舍入差异。整网、双视角、强基线和闭环验证尚未完成，不能把这个局部结果直接当成论文贡献。

**同日 12 层更新：** 完整单视角视觉与 connector 已接通，20 轮同口径测量为修复融合基线 775.49 ms、分图全头 810.25 ms、分图 head 并行 531.62 ms。层间使用两个 hidden ping-pong 缓冲区，全网共用 Q/K/V 和 Attention 输出缓冲区。FP32 逐层参考一致；NPU 全头/并行逐层及换图像后最终输出一致，但相对原融合 FP16 的 connector MAE 为 0.01158。阶段诊断显示 Attention 约 500 -> 220 ms，consumer 约 231 ms，瓶颈已迁移到更接近平衡的两阶段；不等于已达到 Roofline。下一关是双视角资源协调和真实闭环，不能直接用两倍单视角时延宣称胜过整视角并发。

**同日双视角反证更新：** 公平对照后，最强融合整视角并发 1103.42 ms，packed 两视角串行 1056.27 ms，两套 packed pipeline 并发 1056.44 ms；含输入转换/同步与 connector 取回分别为 1115.65/1068.84/1050.58 ms。收益只剩约 4%-6%，并发 packed 的更多核 busy 没有换来更短驻留 makespan，尾延迟更差。这个结果削弱了“大规模多视角调度”的动机，不能反过来制造一个竞争矛盾来保住选题。先测哪些阶段的并发真有净收益；只有证明可达关键路径与现成调度间存在实质差距，才值得写对应机制。详见 [视觉调查](SMOLVLA_VISION_INVESTIGATION.md)。

**同日阶段反证更新：** 真实第 12 层、20 轮轮换测试，跨视角 Attention+consumer 串行 37.379 ms、并发 37.314 ms；限制 consumer 为 mask1/mask3 后为 50.801/39.899 ms。同阶段双视角 Attention 并发也没有净收益。不能直接假定它们在 RK 上资源互补，暂停阶段调度器。下一项 C12 只验证 FC1+激活单核与 FC2 多核之间的具体执行/布局边界；它是延续同一机制调查，不是为了维持选题而再命名一个新问题。

### B. 模型没变大，编译后的执行状态却变大了（C02）

**痛点直述：** 相机从一路变两路，本来仍是同一套权重；实际编译常量区却由约 224 MB 变成 803 MB，而且双视角时延没怎么降。

**问题本质：** 当前优化经常按模型 FLOPs、参数量和 batch 推理，遗漏编译器生成的广播、padding、选择算子与辅助常量。这里至少有两个不同证据：patch 的多余空间工作、B2 的常量膨胀；它们尚未被证明属于同一根因。

**候选机制：** 在保留相机隔离语义的前提下，比较等价 rank/布局表示；建立从 semantic tensor 到 compiled allocation/work 的审计，选实际有效工作更少的形式。还可审查共享权重的 recurrent unrolling：减少十步调用边界是否反而复制常量和权重区域？

**升格条件：** 形成一种跨算子、跨模型可重复的 lowering 规律及自动规避方法，而不是一个 reshape 配方。语义参数身份与编译常量身份要分别建模。

**2026-09-28 实验更新：** 真实 Attention 子图定位了零 mask lowering 路径：每层额外 50331648 bytes，12 层总差 603979776 bytes。去掉已验证恒零的 score-mask Add 后，完整 B2 的 weight region 为 198646080 bytes；实际 Attention 的标准 optimization level 3 控制仍保留膨胀。独立 toy SDPA 不复现，说明是实际图模式相关，而非“零 mask 必然膨胀”。但是完整视觉仅 1739.01 -> 1722.39 ms，输出一致。因此这条降级为内存修复/强基线，不再拿内存大小推导显著加速。

### C. 不把整个 Attention 交给一种设备（C08）

**痛点直述：** 一个融合算子虽然叫 Attention，内部却同时有大矩阵乘、归约和指数归一化；NPU 擅长其中一部分，不代表整包都是最佳执行单位。

**候选机制：** 以 exact stable/online softmax 为算法基线，将 QK/PV tiles 留在 NPU，CPU 承担可验证更快的归一化；按 head/tile 形成生产-消费流水，避免把整个 score matrix 放回主存后再读取。CPU 与 NPU 共享内存也不等于同步、cache coherence 和布局转换免费。

**未知与硬伤：** 目前没有证明 CPU softmax 更快，更没有证明 NPU Attention 瓶颈在 vector 部分。FlashAttention、CoDL 与 HeteroInfer 都是直接反驳；online softmax 公式不是创新。

**2026-09-28 实验更新：** 四 A76 线程的 CPU FP32 SDPA 40.98 ms，与 NPU 全头约 41 ms 相当，但慢于原生 head 并发且还未计转换。Torch FP16 SDPA 413.13 ms；这只说明当前软件实现不适合，不能推出 ARM FP16 的硬件上限。暂不推进整包 CPU Attention，CPU/NPU 分项协作保持未证实。

**下一关：** 单独测 QK / normalization / PV 和 native 交接；先算包含同步与共享 DRAM 竞争的上限。只有形成相反的阶段赢家且完整块胜出，才继续实现。这是独立于 A 的 backend 分工假设，不是 A 失败后的自动替补。

### D. 同一观测的相机分支协作，而不是大 KV 往返（C04）

**痛点直述：** 把模型按 backbone/expert 切开，可能为了卸载前半段，额外下传大量 conditioning，再把后半段搬到慢端侧。多相机则提供真正独立的支路，但整帧必须等最慢支路完成。

**候选机制：** 以 observation/camera/block 的提交前沿描述进度；联合选择输入或完成特征的传输、独立相机本地/远程分工，以及何时启用本地备用路径。只有经过测量值得导出的 checkpoint 才参与续跑，不能默认任意 vision block 都是便宜切口。

**有味道的地方：** 优化目标是 join 的剩余关键路径，不是分别让每个支路最快；投机的是路径和到达时间，不是动作值。最终只接受同一观测、同一模型版本的完整支路结果。

**致命反驳：** RTX CUDA 对照的整次推理约 0.238 s（含该评测协议开销，不是孤立 GPU 核时，也不是公网全云基准），本地 vision 容易成为自己制造的 straggler；中间特征可能比源图更大。网络受限不是默认利好切分。Neurosurgeon、DeepThings、普通 delayed hedge 都必须进入对照。

**下一关：** 从真实 LIBERO 观测统计 source/connector/checkpoint bytes，计入已有或无损 codec 时间、GPU queue 和真实带宽/RTT，再推导何时能胜过全云。先证明非空的合理胜区，再谈迁移实现。绝不以未压缩 float32 图像为弱上传基线。

### E. 把十步 flow 看成常驻条件上的状态机（C09/C10）

**痛点直述：** 十步中真正改变的是 latent，但部署接口可能每次都重新整理同一 conditioning，并跨 CPU/NPU 边界来回提交。将十步直接展开成大图，又可能膨胀编译常量。

**候选机制：** observation epoch 中 conditioning/native binding 常驻；step commit 只推进 latent、时间步和 solver 元数据。比较小型 unroll、重复 bound-input 调用和 resident conditioning，寻找计算循环与编译图的合理边界。

**定位约束：** 若收益只是消除约 8 ms/步的 host repacking，它就是值得实现的强基线，不是论文核心。小状态端云迁移必须计入双端 conditioning 的建立成本，不能假设 6.1 MB KV 免费常驻。

## 近期执行顺序

1. **完整 native 十步 flow。** 先比较原图持久绑定、首步输出静态 conditioning、prefill 一次准备与当前接口；8 ms host repack 不等于全部跨边界成本。保持原 FP32 solver 协议，普通 hoisting 只记强基线。
2. **短 action Attention 边界。** Q50/K149/199 的 CPU FP32 单算子已到亚毫秒，但 Torch FP16 更慢；先测真实 mask/GQA 的完整 block 与交接，不由视觉负结果外推，也不宣称阶段赢家已反转。
3. **端云低成本可行域。** 同观测无损 source/connector/conditioning 字节、codec、准备、queue/service、join 全计费；先证明合理胜区，不启动通用层切分或迁移系统。
4. **保留视觉机制与真实集成。** 当前 B1 native 机制有效、双视角调度收益弱；真实输入、数值与闭环仍须完成。selective linear batching 尚未被普通 B2 对照否定，可作独立小控制；Orca 是直接近邻。FC1 分区降为可选工程项。
5. **零 mask 与 patch 作为强基线。** 不继续以小速度收益或 compiled allocation 大小包装主贡献；不把这些候选合成多模块系统。

暂时不立题。A 保留完整单视角实证，B 降级为基线；E 是首先要补完的强对照，
C 仅在短 action 几何下重开小门槛，D 只做可行域测量。最终只保留经强反证
仍成立的一条机制，详见 [复审](IDEA_REASSESSMENT_2026-09-28.md)。
