# SmolVLA 短 Action Attention：真实边界前置审计

2026-09-28。以 checkpoint 的真实层权重和已导出合成 prefix 为输入，不是实际
观测分布、NPU 中间激活或闭环。目的：纠正随机、预展开、无 mask 探针的偏乐观
解释，而不是宣称异构加速成立。

后续排序修正：此处完整 cross-block 测试降为按需辅助诊断，不再是主线的
下一优先实验。当前优先审计条件生命周期/编译表示与端云实际 payload，见
[阅读讨论辨析](VLA_READING_ABSORPTION_2026-09-28.md)。下方保留本次测量与门槛。

## 1. 不是一种 Attention，而是交替两种依赖

从 installed policy 的 `eager_attention_forward` 截获完整 32 层输入，分别执行
原 suffix 和一个扰动 suffix。所有层都是 15 query heads、5 KV heads、head dim
64；GQA groups 为 3，query 长度 50。

| 层 | Key 长度 | 实际 mask | 两个 suffix 的 K/V |
|---|---:|---|---|
| 偶数 0,2,...,30：self | 199 = prefix 149 + action 50 | prefix 全可见；suffix causal，每层 1225 个不可见位置 | 完整 K/V 改变，不能整包常驻 |
| 奇数 1,3,...,31：cross | 149 | prefix 全可见，无被 mask 位置 | 每层 K/V 均不变，可以一次准备 |

因此，cross 的一次布局准备不等于整个 action expert 都可以静态化。self 的
prefix 部分可按真实依赖审计复用，动态 action 部分仍需更新；不能直接缓存本次
捕获的全部 self K/V。这一结构是模型事实，不是新提出的架构或新颖性证明。

## 2. 加回实际 GQA、mask 和布局后，速度线索缩小

四 A76 核、四线程 FP32，30 次，使用 layer 0/self 与 layer 1/cross 的实际
CPU 激活。各方案均返回原 `[1,50,960]` Attention 输出。模式按块计时，而非
随机轮换；无连续 CPU 频率/热稳态记录，不能作为公平 NPU 对照。

| 方案 | Q50/K199 self 中位数 | Q50/K149 cross 中位数 |
|---|---:|---:|
| installed eager，含实际 mask/GQA | 20.538 ms | 20.116 ms |
| FP32 fused SDPA，每次做 GQA 展开/连续布局 | 1.162 ms | 0.853 ms |
| FP32 fused SDPA，cross K/V 提前准备 | 不测：完整 K/V 非静态 | 0.546 ms |

cross consumer-ready K/V 单次准备约 0.320 ms（仅一个样本），展开后的 FP32
K/V 为 1144320 bytes/层，是五 KV heads 原 FP32 数据量的三倍。保留 16 个
cross 层约 18.3 MB，必须计费；不能只说“布局提前准备就免费”。本探针未测试
不展开 GQA 的分组 kernel，后者也是潜在强控制。

旧随机探针的 0.478/0.624 ms 使用已展开 heads、无实际 mask、无布局准备；
现在 self 1.162 ms 已不低于此前 NPU operator profile 的约 1.080 ms，cross
每次准备 0.853 ms 也约等于旧 NPU profile 的 0.849 ms。只有 cross 预准备的
0.546 ms 留下形状线索，仍不是同精度、同接口或配对结果。

用旧 profile 做一个**量级提醒而非收益预测**：cross 差额约 0.3 ms/层，16 层
乘 10 步约 48 ms，尚未计入任何同步、Q/output 转换、NPU graph submit 与融合
损失。相对约 0.8 s 的强 flow 控制只是约 6% 量级，不能套用 eager 到 fused
的约 20 倍来讲 NPU 加速，更不能直接搬到完整 VLA。

## 3. 算法等价不等于输出逐位一致

所有 token/head、mask、步骤与 FP32 dtype 保持不变，fused SDPA 只是改变
执行/累加顺序，不是减少模型工作的有损算法。但输出不逐位一致：

- self Attention 最大绝对差 1.49e-7；cross 最大绝对差 5.07e-7。
- 将 32 层的接口全部替换为 fused 后，两个 suffix 的 velocity 最大绝对差
  为 1.91e-6、1.43e-6；仅为 CPU 单步、同输入比较。
- 没有 NPU 中间输入、十步轨迹或闭环比较；不能据此保证动作/成功率不变。

与 [Flow 原图输入实验](SMOLVLA_FLOW_INVESTIGATION.md) 不同：后者保持同一
RKNN 图并通过 storage uint32 逐位检查，前者改变了 Attention 的执行后端/顺序。
如果采用硬性逐位约束，此 fused CPU 方案不能过关；若允许同数学计算下的正常
后端舍入差异，仍须预先固定数值/任务验证协议，不事后调误差门槛。

## 4. 下一关必须足够小，且有停止条件

不建设通用 scheduler，不先扩展 32 层。先在**一块真实 cross action block**
比较 NPU producer -> Attention -> NPU consumer 的完整关键路径：

1. 原图/原生全 NPU，普通 native pass-through 与持久绑定均作为控制。
2. 分图但仍全 NPU，以分离 graph boundary 损失与 CPU placement 收益。
3. CPU FP32 fused Attention：每观测准备静态 K/V，只交换动态 Q/结果。
4. 若 SDK/布局兼容，NPU head 分区作为额外控制，不能预设 CPU 是赢家。

先测静态准备、动态 bytes、layout/cache sync、submit/wait、融合损失与实际
kernel 的对照。若这些新增成本已吞掉约 0.3 ms 的 cross 线索，停止 CPU 搬迁，
不继续用流控制包装。若完整 block 测出净收益，再验证十步、实际观测与闭环。

这条仍是探针，不是当前推荐论文主线。更新判断：**C13 保留，但下调预期，
不能拿短 query 故事自动替代已被强对照吸收的 C09。**

证据：[真实 mask/GQA 与数值报告](../results/smolvla_libero/vision_probes/action_attention_real_mask_cpu.json)。
脚本 `scripts/probe_smolvla_action_attention.py`，参数为 model/vlm/data-dir、
repeats、output；板上独立 policy 进程执行，结束后未留下服务。
