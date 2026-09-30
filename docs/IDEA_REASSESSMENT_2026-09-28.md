# 五条旧候选的复审与下一轮证据门槛

本页为旧阶段规划。最新顺序见
[主线计划与自审](MAINLINE_EXECUTION_PLAN_2026-09-30.md)：完整服务、协作可行域、
条件式关键路径机制、完整系统验证。较早的
[机制计划](NEXT_MECHANISM_PLAN_2026-09-28.md)保留历史，不再定义优先级。
下文“尚未执行”的 cold/warm 条件图已在后续完成，勿按历史段落重复立题。

2026-09-28。这是实验规划，不是已成立的论文故事。以 SmolVLA 为主，保持
观测、权重、token/head、动作数、十个 flow 步骤与数值协议；不再优先追逐 FC1。

本日最新验证已完成：C14 紧凑/grouped 子图未胜过普通消费就绪缓存，当前论文
版本降级；普通缓存接完整 action flow 得到约 7.4% 配对收益，保留为 C06 强
基线。C13 cross-block 迁移仍是辅助探针。C04 是独立的实际 payload 可行域
审计；prefix/首步交错只先审计上限。下面保留之前阶段规划，最新决定见
[conditioning 实际验证](SMOLVLA_CONDITIONING_VALIDATION.md)。

## 1. 旧结论哪些需要修正

| 旧方向 | 当前证据 | 当前决定 |
|---|---|---|
| 编译表示膨胀规避 C02 | 实际 B2 零 mask lowering 已定位，weight region 802.6 -> 198.6 MB；完整视觉只快约 1%。 | 保留为内存修复与强基线。除非跨图模式复现更一般规律，不作为速度主线。 |
| 不落 CPU 的并行边界 C01 | 完整 B1 775.49 -> 531.62 ms；双视角最强整图并发后只剩约 4%-6%；Attention/consumer 交错无显著收益。 | 完整单视角机制有效，复杂双视角协调暂停。布局检查、FD views、持久缓冲区作为后续工具保留。 |
| Attention 快慢路径 C08 | CPU 长视觉 SDPA 无优势；此前未测 action expert 的短 query。 | 不将视觉上的负结果外推到动作阶段；先做短 query 的后端与交接控制，不直接建设 tiled runtime。 |
| 同观测相机端云 C04 | 无实际链路、payload 或云排队收益证据；本地分支可能成为 join straggler。 | 做低成本可行域审计，不能直接启动调度器。 |
| 常驻 conditioning 上的 flow C09/C10 | 同图十步已测：FP32 普通提交 1029.21 ms、native 持久绑定 793.24 ms，但预整理 native 普通 pass-through 794.65 ms；完整逐位验证通过。未改图内静态投影。 | 持久绑定降为强基线，不能声称约 23% 的创新收益。图内 cold/warm 与后端协作仍独立待测。 |

FC1/FC2 通道分区 C12 降为可选工程优化，不因存在剩余热点而升格为主线。
上述五条不是五个系统模块，也不是一个失败后另一个自动顶上。

## 2. 新做的短 query 探针

旧视觉几何为 Q/K 长度 1024/1024；已有动作 RKNN profile 的 SDPA 为
15 heads、head dim 64、query 50、key 149 或 199。它们不是同一工作负载。

RK CPU 四 A76 线程，30 次测量，合成随机 Q/K/V；各 dtype 单独测试。
CPU thread 1 对照分别为 FP32 1.668/2.215 ms、FP16 14.534/19.314 ms。

| 几何 | CPU 四线程 FP32 SDPA | CPU 四线程 Torch FP16 SDPA | 已有 NPU operator profile |
|---|---:|---:|---:|
| Q50/K149，15 heads x 64 | 0.478 ms | 3.938 ms | 约 0.849 ms |
| Q50/K199，15 heads x 64 | 0.624 ms | 5.214 ms | 约 1.080 ms |

这些列不是配对公平基准：NPU 是此前完整图的 operator profile，CPU 是本轮
standalone，不含 NPU/CPU 布局、同步、QKV 准备、输出交接。CPU 输入已经是
展开到 15 heads 的 contiguous 张量，没有 GQA 展开成本。CPU 未加 mask，
所以 K199 不能当成实际带 suffix causal mask 的完整 self-attention 性能。
K149 也须核验实际 mask/布局。FP32/FP16 不可混成同精度加速结论，没有动作
精度或闭环验证。四个新探针没有进行热稳态测试或连续 CPU 频率采样；运行后
读取 A76 governor 为 performance、频率 2.352 GHz，不能证明全程固定频率。

因此只留下一个线索：短 query 的 CPU FP32 fused SDPA 已到亚毫秒尺度，
值得测边界；现有 Torch FP16 路径仍无优势。单算子性能尚未证明阶段赢家反转。
分项 QK+softmax+PV 也不能相加替代 fused SDPA；此次分项与 fused 明显不同。

证据在 `results/smolvla_libero/vision_probes/cpu_action_q50_k{149,199}_t{1,4}.json`。
脚本 `scripts/probe_smolvla_cpu_attention.py` 增加了几何参数，保留原默认视觉形状。

**执行更新：真实 mask/GQA 前置审计已完成。** 32 层交替 16 个 self 和 16 个
cross。真实 CPU 激活上 fused SDPA 包括 GQA/layout 为 self 1.162 ms、cross
0.853 ms；仅 cross 可以完整预准备 K/V，降至 0.546 ms。相对旧 NPU profile
只剩约 0.3 ms/layer 的 cross 线索，仍未包含设备边界。32 层 CPU 单步 velocity
最大差 1.91e-6，非位一致；C13 只保留一次有停止条件的 cross-block 测试，
不能作为自动接任的论文方向。详见 [动作 Attention 调查](SMOLVLA_ACTION_ATTENTION_INVESTIGATION.md)。

## 3. 实验优先级

### P0：先建立真正强的常驻十步 flow 基线

**执行更新：原图 native 十步与普通 pass-through 强对照已完成。** 两者中位数
793.24/794.65 ms，主要收益被简单布局准备吸收，持久绑定作为论文主机制判废。
原图固定/动态输入生命周期已有可复用实现；不再追加 dirty-flag 调参。
以下 cold/warm 图是尚未执行的独立控制，而非被这个结果证明有效。
详见 [Flow 调查](SMOLVLA_FLOW_INVESTIGATION.md)。

直白的问题：一个观测的条件不变，动作 latent 连续更新十次；当前接口在每次
调用前又提交整套 prefix 张量。之前测出的 8 ms 只是 host repack，不是全部
设备输入准备/转换/同步成本，不能用它直接给 native path 的收益封顶。

依次比较相同十步、相同输入/noise 的四种完整路径：

1. 当前 RKNNLite 重复输入。
2. 原 denoise 图，prefix native buffer 一次准备和绑定，重复运行十次。
3. 原生 cold graph 首步输出已计算的静态 conditioning，后九步 warm graph。
4. NPU prefill/preparation 一次生成 warm graph 的 consumer-ready conditioning。

CPU 26.47 ms 准备并与首步 overlap 作为额外控制，不能只对串行 CPU 准备
宣称胜出。核验 native format/stride；共享 DRAM 不等于片上驻留，不假定零流量。
同时记 CPU 准备、输入转换/同步、run、输出交接、solver、一次性准备与实际内存。

FP32 latent/Euler 更新与原协议保持一致；把更新塞进 FP16 图可能改变数值
协议，不能当作无损 unroll。先用 CPU FP32 solver 驱动持久 NPU context；只有
编译与执行后端能保持原协议，才测 2/4 步 unroll 的调用成本和 allocation。

判定：若只是普通 hoisting/native binding 就解决了问题，记为强基线，不立题。
但没有这条完整基线，后续 CPU/NPU 或端云收益都会被重复搬运误导。

### P1：短 query + 固定 conditioning 的设备/布局分工

与 FC1 区别：研究对象不是通用矩阵，而是同一 VLA 内从长视觉编码转到短
action query、且后者对同一条件重复执行的具体组合。假设是：静态条件可以
预先转换成消费后端布局，反复跨设备边界时只交换动态 Q/结果，而不是整套 K/V。

这不是“首次缓存 cross-attention”，也不是预设 CPU 赢。最强对照必须是 P0
的 warm/native 全 NPU 路径，包括 NPU 端静态布局整理。

最小完整测试为一个真实 action block：NPU producer -> Attention backend ->
NPU consumer。比较全 NPU、native head 分区、CPU attention，并包含每次转换、
cache sync、启动和等待；测试 prefix 70/149 以及实际 mask/GQA。若短 Attention
仍没有净收益，停止搬迁，而不是用更大的调度器掩盖边界成本。

若成立，再接完整 32 层、十步和真实观测。研究贡献只能来自可复用的边界规律
与完整效果，而不是“按算子选择后端”。没有同协议数值与闭环结果前，不宣称无损。

### P2：端云先审计传输对象和进度，而不是搜索层切点

从同一真实观测统计三种完整传输对象：原始 uint8 图像的无损编码、connector、
prefix conditioning；分别测实际字节、编解码、端侧准备、服务器 service/queue。
最终保持同一模型输入，不用 JPEG/有损特征压缩制造胜区。

比较全云、优化全端、按相机分支的静态分工；把已经发送的字节、并发竞争、
等待最后支路的 join 成本全部计入。同观测的本地工作能否减少尚未完成的远端
工作，是候选问题；已有本地进度不等于应该继续，重启/取消的简单方案必须比较。

若完整全云在合理带宽/排队范围内始终占优，停止单请求加速方向。服务器预算/
多机器人容量是另一个研究目标，只能在明确改目标、测 queue/service 后开展。
不假设免费的双端 conditioning，也不把小 latent 迁移单独当成新颖机制。

## 4. 哪些小实验没有被现有负结果否定

- 完整 B2 不快不等于 selective batching 不快：尚未测同观测两相机仅合批
  linear/MLP、分别 attention，并由兼容 native layout 连接。Orca 是直接近邻，
  即使测赢也不是自动新颖；作为视觉方向的低成本控制保留，不恢复复杂交错。
- CPU 长视觉 Attention 不快不等于 CPU 短 action Attention 不快：本轮已有
  shape 线索，但同 dtype 与完整交接尚不成立。
- B2 零 mask 内存修复只有约 1% 时延收益，不等于所有等价表示都没有价值：
  只有出现另一个可重复图模式时才继续编译器规律调查。
- 本地视觉分支可能慢，不等于所有端云目标均不成立：单请求时延与固定云
  预算下容量必须分开，不能偷换目标来救一个被全云支配的方案。

## 5. 论文逻辑自审

若后续成立，定位 Technique，不宣称已提出新问题。

| 环节 | 候选逻辑与当前缺口 |
|---|---|
| 背景 | 端侧 flow VLA 对固定观测条件执行多次动作细化，视觉/动作有不同几何。 |
| 局限 1 | 当前 wrapper 重复整理/提交条件，但 native 十步的主要收益已被普通 pass-through 吸收；不能作为新机制痛点。图内投影仍待 cold/warm 图验证。 |
| 局限 2 | 真实 mask/GQA 后 self 无孤立优势、cross 仅余小线索；统一后端是否错过完整边界收益仍未证明，不能写成现有事实。 |
| 目标 | 验证按条件生命周期准备布局、按完整边界成本选择消费后端，能否优于最强全 NPU 路径。 |
| 挑战 A -> 机制 A | 静态/动态依赖、native ABI 不同 -> 真实依赖审计、一次准备、持久 buffer 和 attr 检查。 |
| 挑战 B -> 机制 B | 单算子收益可能被交接抹去 -> 一块完整 producer/attention/consumer 的边界计费。 |
| 挑战 C -> 机制 C | 浮点后端/solver 改变可能影响动作 -> 保持数值协议，逐层、十步、动作和闭环验证。 |
| 候选贡献 | 规律、执行机制、端到端评价均尚未闭合；端云不是当前机制的必需模块。 |

自审：目标与机制在逻辑上对应，挑战有真实实现依据；但局限 2、完整收益和
跨模型规律缺证据，不能通过“已可立题”的审查。不起系统名，不写胜利式 Introduction。

## 6. 文献边界

- [HeteroInfer, SOSP 2025](https://arxiv.org/html/2501.14794v2)：已研究 mobile
  GPU/NPU shape/order sensitivity、tensor partition 和同步；后端 shape 选择不是新点。
- [Diffusers text K/V cache](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/text_kv_cache.py)：
  exact 静态 text conditioning cache 是已有实现；不能把不变量 hoisting 重新包装。
- [SmolVLA](https://arxiv.org/html/2506.01844v1)：已有异步 action chunk 执行；
  感知/动作执行解耦或大小脑命名不能单独构成贡献。
- [Hybrid SD](https://arxiv.org/abs/2408.06646) 与
  [EC-Diff](https://arxiv.org/abs/2507.11980)：已有 diffusion 端云阶段协作，摘要
  显示涉及大小模型切换/近似；与当前同模型、保留全部步骤的约束不同，但
  “分阶段迁移 denoising”本身不能主张新颖。此处仅为元数据/摘要级筛查。
- Orca、Rammer、CoDL、SpotServe 仍为强近邻，见 [路线扩展](IDEA_TASTE_EXPANSION.md)。

上述检索只校正机制边界，不构成“没有相关工作”的新颖性证明。
