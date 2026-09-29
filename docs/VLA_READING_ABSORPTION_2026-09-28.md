# VLA 阅读讨论的辨析与后续优化清单

更新：下文 A/C14 的实验已完成，当前论文版本降级。普通 consumer-ready 缓存
通过完整 flow，紧凑/grouped 子图没有优势；最新决定与数据见
[实际 conditioning 验证](SMOLVLA_CONDITIONING_VALIDATION.md)。下文保留实验前的
阅读判断与门槛，不再把 A 当作下一主实验。

2026-09-28。输入为用户粘贴的 speculative VLA 讨论，以及本仓库实测。
主模型仍为 SmolVLA；不因讨论提到 OpenVLA 就换流派。这里是候选筛选与证据
门槛，不是已成立的 CCF-A 论文选题。推理图重排/拆分允许，但不改训练得到的
网络、权重或采样协议，不训练新 drafter/verifier。

## 1. 这份讨论值得吸收什么

有价值的不是把 temporal reuse、adaptive K、validity verifier 拼在一起，
而是三个判断维度：conditioning 是否占关键路径；提案与目标的职责能否分离；
收益是否需要完整 runtime 与目标模型语义才能成立。先证明 opportunity 再设计
policy 的顺序应保留，不能先画一个大框架再找动机。

原文没有给出综述的可定位链接，`chatgpt-content-reference` 无法在本任务解析。
检索得到同主题的 [Speculative Decoding for Multimodal Models: A Survey](https://www.preprints.org/manuscript/202603.2344)，
但不认定这就是原文指代的版本。以下关键技术边界依据原论文/代码和本地数据。
附件中“30% 条件、acceptance 少 5%、latency 少 25%”是设想，不是实测证据。

### 必须区分三种保证

1. **training-free**：无需新训练。不自动意味着动作、输出分布或推理次数不变。
2. **算法/目标语义保持**：不删目标条件、token/head/动作/步骤，不放宽接受规则。
   浮点后端/融合顺序仍可能造成数值差异，需单独报告并验证闭环。
3. **逐位保持**：同输入/噪声时所有输出 storage bits 一样。当前同 RKNN 图
   六输入模式已通过这一检查，但 CPU fused Attention 没通过。

不能用“成功率暂时没下降”替代后两种保证，也不能将分布保持理解成固定随机数
下的同一采样路径。

## 2. 对附件建议逐项处理

| 建议 | 值得吸收的部分 | 当前处理 |
|---|---|---|
| Selective Conditioning | 目标完整、提案简化可以分离两种成本 | 只在具有严格 verifier 的提案端保留。直接删 SmolVLA 目标视觉 token 不做。 |
| Temporal Condition Reuse | 不同信息有不同变化/依赖周期 | 同观测十步条件不变量可 exact 复用。跨观测“相似就复用”不能用于目标端；若将来有严格投机，则可用于提案端。 |
| Adaptive Speculation Depth | 应优化接受的有效工作/总成本，而非只优化 acceptance | 可作 AR 严格投机的系统控制，但不能改成 SmolVLA 的自适应去噪步数/动作数；也不把规则调 K 当新颖主点。 |
| Validity-aware Verification | 控制等价性是独立研究问题 | 当前不放宽接受。动作距离小不保证相同闭环结果，也不是原目标策略的严格验证。 |
| Closed-loop Validity Check | 生成与执行是不同的反馈边界 | 它改变推理触发与控制协议，是另一类研究。不能为了 runtime 故事重新引入 freshness/动态物体假设。 |
| Visual KV/ROI 局部刷新 | 依赖审计比像素相似度阈值更重要 | global attention 后旧 KV 一般会依赖已变化区域。仅逐字节相同的独立视觉分支能 exact 复用；命中率和检测成本未测，不立题。 |
| Draft/Verify Overlap | 提案端可占用目标端等待窗口 | 保留原理，但属于已有系统轴，需验证后端兼容、回滚和完整收益；SmolVLA 尚无兼容 verifier。 |
| Phase-aware Speculation | 任务阶段可能影响收益 | 不先建 free-motion/contact 分类器或阈值；任务依赖较强，且已有近邻，不作为当前基础痛点。 |

一个关键修正：**提案的近似不必意味着最终推理有损**。例如草稿使用旧帧，
目标仍使用当前完整帧，在正确严格验证/回退下可保持目标语义；提案质量影响的是
接受长度和浪费成本，而非自动改变最终目标输出。但这种保证有前提，不能直接
用于连续 flow 的候选 latent。“目标算完后比两个动作是否接近”也没有自动省掉
目标计算。

## 3. 文献校正：哪些不能当空白

- Songsheng Wang 等的 [Spec-VLA, 2025](https://arxiv.org/html/2507.22424)
  是离散 AR 动作模型，论文明确区分严格匹配和距离放宽；它不要求重新训练
  target，不等于所有模型都无需训练。[作者代码](https://github.com/PineTreeWss/SpecVLA)
  包含 drafter 训练脚本和训练过的 draft checkpoints。不能把“training-free
  relaxed verification”当作我们已满足无损条件的起点。
- Quan Kong 等的 [ParallelVLM, 2026](https://arxiv.org/html/2603.19610)
  已采用 draft/target 并行 prefill 与 decode，并只裁剪 draft 条件，target
  保留全条件进行验证。因此“少看图的 draft + overlap”不是空白。
- Zihua Wang 等的 [SV-VLA, 2026 preprint](https://arxiv.org/html/2604.02965)
  已使用宏规划与在线 verifier 触发重规划；这是控制协议层的研究，不能将
  附件组合直接称为新 runtime。这里只核验总体机制，不主张其无需训练。
- Valentin De Bortoli 等的 [Accelerated Diffusion Models via Speculative Sampling,
  2025](https://arxiv.org/html/2501.05370) 已有连续随机链的 exact 投机与免训练
  提案。不能声称连续动作不可能 exact speculation；但论文的随机转移核不
  自动等于 SmolVLA 当前固定噪声下的十步确定性 FP32 Euler。
- Joshua Ainslie 等的 [GQA, 2023](https://arxiv.org/abs/2305.13245) 和
  [Diffusers 静态 text K/V cache](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/text_kv_cache.py)
  约束下一条候选的新颖性：保留 KV heads 共享、不重复投影静态条件本身都是
  已有基本机制。我们要验证的是部署/编译是否破坏了这个结构，而不是发明 GQA。

这些是近邻机制核验，不是穷尽检索或“无人做过”的证明。

## 4. 新补的本地形状/成本审计

此前 CPU 静态投影报告中，32 个 cross K/V projection 的输入宽度 320，权重
为 `[320,320]`。实际 action attention 为 15 query heads / 5 KV heads / 64
head dim。旧 NPU operator profile 的同名奇数层 Conv 却报告：

```text
输入：[1,320,1,149]
权重形状：[960,320,1,1]
输出：[1,960,1,149] -> [15,64,1,149]
```

**已观察到的是编译报告宽度三倍，不是独立证实了三倍物理 MAC、参数复制或
DDR 流量。** “GQA 展开折入投影”是可核验的原因假设；需查看真实子图、分组
输出与编译形态的对照。不得重演 B2 常量膨胀未定位时的过度归因。

按已有 SDK 表分类，而不是新跑 benchmark：

| 一步中可疑的固定条件工作 | 算子数 | SDK 累计时间 |
|---|---:|---:|
| cross prefix reshape | 32 | 2.769 ms |
| cross prefix transpose | 32 | 3.292 ms |
| cross projection，报告输出 960 | 32 | 2.415 ms |
| self 的固定 prefix transpose | 32 | 3.268 ms |
| 合计 | 128 | 11.744 ms |

原 profile 算子总时间 73.149 ms；这不是当前 native 十步总时间，也不能将
11.744 直接乘十宣称节省。图边界、SDK 计时、融合、缓存和 allocation 都会变化。

若 materialize 所有 16 个 cross 层的 projected K/V，紧凑 5-head FP16 数据
为 3051520 bytes，展开到 15 heads 为 9154560 bytes。它们是按形状计算的
状态 payload，不是已测整个 runtime peak memory。当前 CPU expanded FP32
cache 的 18.3 MB 是另一精度口径，不能混用。

它提出的是一个值得验证的问题：**固定条件准备得更“接近消费”，是否也可能
提前放大存储与读入成本？最合适的常驻形式未必是最终全展开形式。**
“缓存越靠后越好”不能先当结论；普通紧凑 GQA cache 是必需强基线。

证据：[条件 profile 审计](../results/smolvla_libero/vision_probes/denoise_conditioning_profile_audit.json)、
[逻辑投影检查](../results/smolvla_libero/static_projections_overlap_probe.json)、
[原 SDK 表](../results/smolvla_libero/denoise_operator_profile.txt)。分类脚本
`scripts/analyze_smolvla_conditioning_profile.py` 对当前几何/算子数做严格检查，
没有进行新的硬件计时。

## 5. 后面还能做的优化：按对象分，而不是拼模块

### A / C14：视觉条件的紧凑常驻与消费表示

这是近期最有本地依据的候选，不等于最有论文新颖性。

- **痛点对象**：VLM prefix 的固定条件在动作专家十步中反复转换/投影；输出
  还可能提前展开成三倍 head 状态。原生绑定只解决图外提交，没移除图内工作。
- **动机**：CPU cross Attention 搬迁最多只留小线索；先保持 NPU 消费，检查
  条件的生命周期和表示边界是否破坏原模型已有的共享结构。
- **候选机制**：同观测一次 NPU preparation；warm 图只处理动态 latent。
  对照 compact 5-head projected K/V 与 expanded 15-head native K/V；在消费者
  处按需共享/展开，不降精度、不删 heads、不减少十步。
- **挑战**：compact 状态可能导致慢 broadcast/Gather；expanded 状态可能让
  工作集更大。必须包含准备、graph/context 内存、layout/sync 与十步总耗时。
- **最强控制**：原图普通 native pass-through/持久输入；普通 loop-invariant
  hoisting；紧凑 GQA cache；expanded consumer-ready cache。CPU preparation
  overlap 不能只打串行 CPU 准备，NPU preparation 是必需控制。
- **停止条件**：若普通 cache 就解决，归入优化基线；若紧凑表示被 kernel
  代价抵消，停止，不用更大的调度器包装。只有可重复的编译/表示规律、完整
  收益和跨形状/模型效果成立，才讨论论文贡献。

先提取一个真实奇数层的 K/V projection + GQA 变换，对照每次准备/一次准备、
紧凑/全展开；然后接同图 warm 十步。保持原 NPU 数值协议，CPU FP32 准备结果
不能直接当原 FP16 图的逐位等价替代。这里的“展开”不是之前 B2 零 mask 常量区。

### B / C04：同观测相机分支的计算—传输协作

这是端云可行域审计，不直接写一个新层切分算法。

- **痛点对象**：送原图、视觉 connector、完整 prefix KV 是不同传输对象；
  一个更靠后的模型切点并不保证字节更小或观测到动作更快。
- **动机**：多相机在视觉编码时独立，直到多模态 prefill 才 join。可以让一支
  在端 NPU 编码、另一支上传远端编码，但两支都必须完整保留，且不能改变帧。
- **候选机制**：计算与上传重叠，按真实 bytes/本地剩余工作/远端 service 的
  联合成本决定哪支在何处处理。先测试静态分工，不先建在线 scheduler。
- **最强控制**：相同源 uint8 图像的无损编码全云、优化全端、最好静态分工。
  不拿未压缩 float32 上传作主基线。已有 0.238 s CUDA 推理结果包含旧协议，
  不能当隔离云算时延；优化后的本地完整策略也尚未测。
- **停止条件**：若原图无损编码总比 connector 更小，或本地成为 join
  straggler，单请求方向停止。容量/云预算是不同目标，不能偷换目标救方案。

只序列化当前部署已经产生的 native FP16 值可以保持那些值的 bits；将 FP32
target 特征转成 FP16 则是另一个协议，不能称为无损传输。预处理搬到另一端也
必须检查同一图像的模型输入一致。当前没有实际 bytes/codec/network 收益结果。

### C / C15：prefix 就绪与首步动作的依赖交错

这是廉价依赖/上限审计，优先级低于 A/B。

- **问题**：当前完整 prefill 输出 64 个 K/V 后才启动动作专家；某层 K/V
  就绪是否已足以释放第一步对应 expert 层，而不必等待整个 prefix？
- **机制候选**：保留模型所有计算，按层/chunk 生产并消费条件；NPU/CPU 或
  端云可用不同资源重叠，不跨未知未来观测作预测。
- **硬限制**：当前每一步 flow 必须得到前一步 FP32 latent。简单流水只可
  重叠 prefill 与**第一步**，不能把每个 flow 步都算成可隐藏。
- **收益审计**：相对“prefill 后做十步”的串行控制，在没有独立算子加速时，
  理想收益不超过 `min(prefill, first_step)`。本地现有不同口径数据仅提示
  首步为约 70-110 ms 量级，不是新测统一上界。共享 NPU/DRAM 还可能更慢。
- **停止条件**：若依赖/上限已不足以影响完整 VLA，停止 layer-level
  streaming，不先生成几十张子图。V→A 叙事更贴切不代表收益一定更大。

### D / C16：近似只在提案端，严格验证的系统协作

保留为 AR VLA/VLM 的备用机制，不作为当前 SmolVLA 的直接实现路线。

允许草稿使用旧视觉条件/较少条件或历史动作检索，但 target 必须对当前完整
观测严格验证；拒绝时精确回退。优化对象是新观测的 conditioning startup 与
验证/回滚开销，而不是降低 target quality。不是“免训练的新 drafter”，也
不是 generic adaptive K；ParallelVLM 与检索投机是强近邻。

当前 SmolVLA 为 `x_next = x + dt * velocity(x, condition, time)`，未来输入
依赖本步结果。平行计算预测 latent 上的 velocity 不等于原轨迹上的 velocity；
放宽 latent 距离或新增 stochasticity 会改变当前协议。必须先找到兼容该
确定性 ten-step 目标的 verifier/proof，不能直接套用 AR acceptance 或随机
diffusion coupling。模型/输出类型改变必须由用户另行决定，不悄悄换主模型。

## 6. 工程成果仍然要保留，但不能假装都是研究候选

| 项目 | 当前可做什么 | 当前不能说什么 |
|---|---|---|
| 视觉 native head 并行 | 真实图像、policy 集成、数值/闭环验证；多形状与热状态重复 | 单视角 31% 不等于双视角/完整 VLA 31%；双视角强控制只余约 4%-6%。 |
| patch/零 mask 改写 | 作为强视觉基线与内存修复 | B2 常量区下降不等于巨大时延收益，实测仅约 1%。 |
| 十步 native 输入 | 同图严格 parity 已通过；移除现有 wrapper 的多余整理 | 持久绑定 793 ms 对普通 native 提交 795 ms 不支持新 residency 加速贡献。 |
| 图内静态 projection/布局 | 先完成一次准备与 warm 图公平控制 | CPU 26.47 ms 不是 NPU 节省，也不能把 SDK 11.744 ms 当测得净收益。 |
| CPU cross Attention | 只作需要时的后端诊断，不排在主线前面 | 0.546 ms 不包含 NPU 往返；最终 velocity 非位一致，不能声称无损。 |
| selective camera batching | 低成本测 linear/MLP 合批、Attention 分开 | Orca/ActionFlow 已提供直接机制先例；不能仅重新命名为 VLA runtime。 |

## 7. 执行决定与论文门槛

执行顺序：**A 的真实条件子图因果审计**与 **B 的 payload 可行域审计**。
C 先画依赖/算 ceiling，不先实现；D 只做兼容性/文献门槛，不切换模型。
取消此前 C13 自动作为下一主实验的排序。A/B 是不同问题，不预定组成同一篇
论文，也不是其中一个不行便自动用另一个替换。

如果 A 成立，最多先定位 Technique 候选；不宣称“首次提出新问题”。论文链条
应由实测现象决定：固定条件的逻辑共享 -> 编译/常驻表示是否扩大实际工作 ->
紧凑/消费就绪表示为何不能简单兼得 -> 一个可验证机制 -> 完整收益与适用边界。
目前关键 idea 与方法尚未完成，创新性、数值和完整 VLA 效果均是缺口。

逻辑自审：背景与 A 的对象对应，ABI/依赖/表示挑战有依据；方法到贡献暂不能
闭合。B 缺真实链路数据，C 缺统一时序且上限偏小，D 缺当前模型的兼容 verifier。
**因此目前能确认的是优化机会与验证清单，不是已确认一条可出 CCF-A 的主线。**
