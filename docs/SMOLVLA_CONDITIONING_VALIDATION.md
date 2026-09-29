# SmolVLA conditioning 的实际验证与判定

最新 [完整策略重放](SMOLVLA_POLICY_REPLAY_VALIDATION.md) 已将 C06 接入实际
LIBERO 演示图像与原 FP32 Euler：共享双相机并发视觉控制，相对 resident
action 的完整策略中位数低 2.43%-2.78%，已测逐步输出逐位一致，非闭环结果。
本页保持原合成 prefix 实验口径；C17 最终数值与强对照以续验页为准。

2026-09-28。回答的问题是：C14 的紧凑/展开 K/V 取舍，是否需要新系统机制，
还是普通不变量外提与合适缓存布局就能解决？结论先行：**当前版本降为强工程
基线，不作为独立论文主线。** 展开且消费就绪的普通缓存通过完整 action flow
验证；紧凑与 grouped 版本在实测子图中没有胜过它。

2026-09-29 后续更新见 [C17 完整边界验证](SMOLVLA_CROSS_BOUNDARY_VALIDATION.md)：
本页失败结论限于原消费子图。手工贯通真实 Q/RoPE/output 表示、配合紧凑消费
就绪条件后出现 11%-12% 单块收益，但有少量末位差异，未接完整 flow，也未
成为论文主线。普通 C06 的严格逐位 baseline 保持不变。

## 1. 保持的协议与边界

- `HuggingFaceVLA/smolvla_libero` 的原 32 层动作网络，15 Q heads / 5 KV heads，
  head dimension 64，149 prefix tokens，50 action tokens，十个依赖去噪步骤。
- 不删视觉条件、Q heads、动作或步骤，不改权重、训练、目标精度和 Euler 协议。
  CPU suffix embedding、latent 和 Euler 更新仍为 FP32；原 RKNN 网络为 FP16。
- 数据是原导出过程产生的**合成固定 prefix**，以及真实 checkpoint 对两个
  suffix 产生的 layer-1 query；不是来自真实机器人图像的评测分布。
- 完整 flow 测量包含 32 层网络、动作输出投影、每步速度取回、CPU embedding
  和 FP32 solver；**不包含视觉、prefill、上下文初始化或闭环任务**。
- NPU preparation 每次 flow 都重新执行一次，成本包含在缓存方案总计时中。
  所有模式共同的一次原始 prefix native 打包另报，第一轮为 3.397 ms。
- RK3588，RKNN Toolkit/Lite/Runtime 2.3.2，driver 0.9.8；CPU affinity 4-7，
  Torch 四线程。测量期采样 NPU 为 1 GHz，DRAM 为 2.112 GHz。
  未连续记录 CPU 频率和温度，也没有多设备/长期热稳态验证。

## 2. 子图对照没有省略强基线

从真实 ONNX 的第一个 cross 层（layer 1）提取原投影、GQA 展开、mask/scale
和 Attention；scale 核验为 1/8，cross additive mask 恒零。不是随机矩阵替代。

| 方案 | 每步保留的准备工作 | 常驻 K/V |
|---|---|---|
| fused recompute | 原投影、展开与布局变换 | 原始 prefix |
| compact repeat cache | 原 GQA 展开及后续转置 | 5-head projected K/V |
| expanded cache | 后续转置 | token-major 15-head K/V |
| compact grouped cache | 合并共享 KV 的三个 query head，再恢复输出布局 | 5-head projected K/V |
| consumer-ready cache | 只裁掉物理 padding，保留原 Attention | head-major 15-head K/V |

Grouped 是已有 GQA packing 思路的控制：把三个 Q head 的 50 行拼成 150 行，
仍计算全部 15 个 Q heads，不是减少 head 数。它不是本文新发明。

准备和消费之间使用 native FD views；没有经 CPU 取回再重排 K/V。
保留默认 SDK cache 同步，查询更新的 FP16 打包/显式同步和最终输出读取
都计时。所有方案使用相同查询输入边界。

### 实际遇到的 ABI 问题

最初 `[1,320,1,149]` 接口的输出被 native padding 成 152 列，消费输入却只有
149 列，直接 alias 被尺寸检查拒绝。没有修改 SDK 标为 read-only 的 `w_stride`。
普通 compact/expanded 改用 token-as-channel、64-wide head 行，生产和消费的
channel padding 一致。consumer-ready 暴露物理 152 列输入，并在图内 Slice
恢复 149 列，**不让 padding token 参与 Attention**。这个整理成本仍被计入。
它是已解决的 ABI 工程问题，不单独包装成新研究问题。

### 40 轮轮换顺序的配对结果

单位 ms；含准备、query 打包/同步和一次最终读回。

| 方案 | 一次消费 | 同观测十次消费 |
|---|---:|---:|
| 原融合重算 | 1.893 | 17.138 |
| 紧凑缓存 + 原展开 | 2.898 | 19.516 |
| 展开缓存，尚非消费布局 | 3.212 | 19.182 |
| 紧凑缓存 + grouped | 3.781 | 28.339 |
| **普通 consumer-ready 缓存** | **2.172** | **13.775** |

十次消费缓存版节省约 19.6%，但这是**一个真实层的十个独立交替 query**，
不能写成完整十步 flow 的 19.6%。一次消费反而不适合额外缓存准备。

## 3. 为什么更少计算反而未必更快

另行采集 SDK profile，仅用于机制定位，不把 profile 延迟当生产计时：

- 原融合把 repeat 折入 K/V 投影：各投影报告 `[960,320,1,1]`。
  SDK weight region 为 1231104 bytes；只产生 compact 的 preparation 为
  411904 bytes。NPU expanded 输出与 compact 输出逐 head 重复的值逐位一致。
  这些证据强支持 repeat-folding 解释；**仍未直接读取编译权重，未测实际 MAC
  或外存流量**，不宣称已证明三倍硬件计算/DDR 流量。
- compact cache 移除投影后，出现两个 Gather（61/57 us）以及对 expanded
  K/V 的转置（297/295 us）。原融合相应两支 reshape/transpose/Conv 合计
  549 us。省下 projection 并未自动得到更快消费。
- grouped 的 Attention 核心从 862 us 降到 541 us，但 query 重排本身报告
  1206 us；它改善核心 kernel，却恶化整个子图。不是“kernel 更快就能救方案”。
- consumer-ready 把投影和主要 K/V 转置都移到一次准备；每步留下两个 Slice
  （66/64 us）。消费阶段 Attention 仍报告 878 us，收益不是删掉 Attention。

证据：[fused profile](../results/smolvla_libero/vision_probes/fused_profile.txt)、
[compact profile](../results/smolvla_libero/vision_probes/warm_compact_profile.txt)、
[grouped profile](../results/smolvla_libero/vision_probes/warm_grouped_profile.txt)、
[ready profile](../results/smolvla_libero/vision_probes/warm_ready_profile.txt)。
SDK `RW(KB)` 和 driver busy 不等于实测 DDR 带宽或 MAC occupancy。

## 4. 完整 32 层、十个依赖步骤

只将 16 个 odd cross 层的固定投影/变换外提到一次 NPU preparation。
even self 层保持原始 prefix 与动态 action K/V；warm 图保留所有动态工作。
原图和 warm 图的 SDK 表中均有 **32 个 exSDPAttention**。

强控制同时包括原图 native 持久绑定和普通 native pass-through；不以旧 FP32
提交路径作主要对手。四个 context 共存，warm/preparation 不重复分配已借用
输入，仅分配自己的输出。没有 CPU preparation 或隐藏的 CPU K/V 重排。

两次独立进程启动，各 12 轮轮换三种方案；每个 flow 都从相同 FP32 noise
开始，下一步 latent 来自本步真实 velocity，不是预先准备的 query。

| 方案 | Session 1 中位数 | Session 2 中位数 | Session 1 P95 | Session 2 P95 |
|---|---:|---:|---:|---:|
| 原图 native 持久绑定 | 747.77 | 749.43 | 848.90 | 800.55 |
| 原图普通 native pass-through | 755.80 | 756.07 | 819.57 | 864.62 |
| **普通 consumer-ready 缓存** | **692.65** | **693.32** | **761.49** | **771.29** |

相对较快的持久绑定控制，两次中位数下降 **7.37% / 7.49%**，约 55-56 ms。
相对普通 pass-through 为 8.36% / 8.30%；不能只选较慢基线突出收益。
这与此前 793/795 ms 是不同进程、CPU affinity/环境下的测量，不直接跨表相减。

Session 1 中位分项：原图 run 合计 693.65 ms；warm run 为 628.37 ms，
另付一次 preparation 10.44 ms。Session 2 为 694.14 / 629.27 / 10.46 ms。
CPU embedding 均约 49-50 ms，说明收益主要来自图内准备外提，而不是 CPU
线程配置差异。分项中位数不要求严格加和为总时延中位数。

逐轮配对，缓存对持久绑定更快 11/12、9/12 轮；paired median gain
为 55.31/55.84 ms。20,000 次配对 bootstrap 的描述性 95% 区间为
[52.92,80.35] / [11.32,57.34] ms。这些不是跨设备、任务或热状态的置信区间。

### 内存不免费

| SDK 查询 | 原图 | 一次 preparation | warm 图 |
|---|---:|---:|---:|
| weight region bytes | 207243456 | 19666944 | 187586752 |
| internal bytes | 13125120 | 10117120 | 13665280 |
| native output bytes | 3584 | 9338880 | 3584 |

prep+warm 权重区约等于原图，不能声称节省模型权重。但多图上下文、internal
buffers 和 9.339 MB expanded native state 有真实成本。按 compact shape
算的对应 native state 为 3.113 MB，不是已测可同速替换的方案。
上表是 per-graph query，不是隔离进程的 peak RSS；也不包含所有框架分配。

## 5. 数值检查

- 子图：两个真实 checkpoint query、改变 prefix 和恢复 prefix，所有五种方案
  与本次 fused FP16 anchor 逐位一致；NPU compact/expanded projection 重复
  heads 的输出逐位一致。FP16 anchor 相对 FP32 ONNX 有约 0.00166 最大差异，
  那是已存在的后端精度边界，不把 FP16 宣称成 FP32 原模型逐位相同。
- 全图：FP32 ONNX ReferenceEvaluator 与独立 ONNX Runtime CPU 后端分别
  对原 prefix、另一 suffix、改变 prefix 验证 original/split velocity：均逐位
  一致、输出有限。Mac NumPy reference 发出 matmul warning，因此另跑 ORT
  作独立核验，没有依赖那个 warning 环境单独判断。
- NPU 完整 flow：两次 session，所有十步 velocity 和每步更新后的 FP32 latent
  与原图逐位一致；真正改变 value prefix 后仍一致，恢复后仍一致。
  改变 prefix 的最终 latent 与原来的最大差异为 2.64521，避免无效失效测试。
- 原图 native C ABI 对现有 RKNNLite fixture 输出也逐位一致。

这不是对所有机器人观测的形式证明，也没有新跑闭环或成功率评估。

## 6. 科研判定与后续边界

**C06 / consumer-ready caching：已验证的强工程基线。** 可以在后续完整策略
集成/真实观测验证中使用，避免未来工作靠一个差的 action runtime 获利。

**C14 当前论文版本：停止升格。** “紧凑状态 + 新消费机制比普通缓存更好”
在这个真实子图中没有成立；普通 consumer-ready 缓存已经取得完整 flow 收益。
不通过加调度器、端云、Agent 名字来重新包装这个基线。不能因物理状态三倍
就假设容量瓶颈，也不能因 grouped kernel 更快而忽略完整转换成本。

保留的观察是：缓存位置决定每步剩余布局成本，优化某个 kernel 未必优化
完整路径。它有诊断价值，但目前没有超出已有 caching/GQA/layout 优化的
独特规律，也没有 cross-model/device 证据。要重开 C14，必须有**新的实测
障碍或不同机制**；不是继续给本次落败方案调阈值。

C04 的无损 payload/端云可行域是另一个尚未验证的问题。本轮没有测网络，
没有迁移模型、训练 drafter，也没有证明异构/投机或端云增益。

## 7. 证据与复现

- [五方案子图结果](../results/smolvla_libero/vision_probes/fragment_ready_core7.json)，
  [完整 flow Session 1](../results/smolvla_libero/vision_probes/full_flow.json)，
  [Session 2](../results/smolvla_libero/vision_probes/full_flow_repeat.json)，
  [配对统计与 clocks](../results/smolvla_libero/vision_probes/conditioning_paired_audit.json)。
- [FP32 ONNX reference](../results/smolvla_libero/vision_probes/conditioning_full_onnx_parity.json)，
  [独立 ORT](../results/smolvla_libero/vision_probes/conditioning_full_ort_parity.json)。
- 完整源码：`capture_smolvla_conditioning_fixture.py`、
  `extract_smolvla_conditioning_probe.py`、`extract_smolvla_conditioning_full.py`、
  `verify_smolvla_conditioning_onnx.py`、`smolvla_native_conditioning.cpp`、
  `smolvla_native_conditioning_full.cpp`、`probe_smolvla_native_conditioning.py`、
  `probe_smolvla_conditioning_full_flow.py`、`analyze_smolvla_conditioning_results.py`。
  均在 `scripts/`。编译仍用已有 converter，FP16、optimization level 0。
- Mac 大图缓存 `/Users/wangjiwei/.cache/smolvla-conditioning-probe/`；板端工作目录
  `/root/smolvla-conditioning-probe/`。没有替换部署中的原图，没有常驻新服务。

制品 SHA-256，Mac/board 对照一致：

```text
original: bcef2130221371e8c213ca2dca936e0d38aefdd2eefecc0e371b4e82f6ebdf7a
prepare:  15405bd818e32474415c2086ad3369a81c964e2a913fbf5282706f95b5ee50b4
warm:     d3ac3bb2e8626d44014473f9f042d9050965184882a63a844eb3ca154bb49bb2
```

先运行 full extractor，再完成输出目录的转换，避免 extractor 未写完 warm
文件时启动目录批量转换而漏编 warm 图。板端编译 native C++ 后运行 full-flow
脚本，参数见结果 JSON 的 load command；准备和 warm 图在同一个目录。
