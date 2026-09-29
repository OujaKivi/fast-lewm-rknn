# SmolVLA 常驻 Flow 输入调查

后续图内准备外提已完成：普通 consumer-ready 缓存通过完整 32 层十步验证，
配对约 7.4% action-flow 收益；C14 紧凑版本没有子图优势。最新数据与边界见
[conditioning 实际验证](SMOLVLA_CONDITIONING_VALIDATION.md)。本文保留原图输入
生命周期对照，不能与新实验跨进程绝对时延直接相减。

2026-09-28。保持原 checkpoint、原 RKNN 图、全部十步、原 FP32 suffix embedding
和 latent/Euler 更新。这里只测预先导出的合成固定 prefix 下的十步真实依赖链，
不含视觉、prefill、初始化、动作执行和闭环。不是已成立的论文贡献。

## 1. 先区分三层重复成本

- Python 侧 K/V 转置/复制：已有约 8.204 ms/步控制。
- RKNN 输入 API 侧的 dtype、layout、复制及提交：必须单独测，不能用前者封顶。
- 图内静态 K/V 变换/投影：本轮完全不改图，也没有消除这些计算。

初轮四模式十步测量为 legacy 预整理 FP32 输入 1028.98 ms、每步重新准备
native prefix 831.88 ms、native 持久绑定并保留默认同步 760.63 ms、显式 dirty
同步 823.45 ms。输入准备累计中位数约 252.77/74.05/0.42/0.67 ms；run 累计
约 708.60/708.96/711.70/758.79 ms。静态 native 准备约 6.4 ms/观测另计。
这说明 native 路径主要减少接口工作，而非改变模型计算。

**这 26% 不能作为最终收益。** 初轮 legacy 是预整理的 FP32 输入，尚未包括
预转换 FP16、预整理 native layout 后普通 pass-through 提交的更强对照。
显式 dirty 同步初轮更慢，不能假定省掉自动 flush 就一定更快；原因未定位。
初轮 prefix switch 使用配对 K/V token permutation，语义上可能保持 Attention
不变，不是最强失效检查；补充轮改为修改 value 并要求轨迹确实变化。

证据：[初轮十步](../results/smolvla_libero/vision_probes/native_flow_paired.json)、
[初轮频率与 busy](../results/smolvla_libero/vision_probes/native_flow_load.json)。

## 2. 强对照结论：普通 native pass-through 已吸收主要收益

六模式各 12 次轮换，全部同图、同 prefix、同噪声、十步真依赖。下表单位为 ms；
“输入”和“run”分别取各阶段累计中位数，不能与总耗时中位数直接相加。

| 模式 | 十步中位数 | 十步 P95 | 输入累计中位数 | run 累计中位数 |
|---|---:|---:|---:|---:|
| 预整理逻辑 FP32，普通提交 | 1029.21 | 1140.16 | 271.68 | 697.38 |
| 每步重写 native prefix | 828.45 | 980.40 | 84.13 | 695.59 |
| native 持久绑定，默认同步 | 793.24 | 821.77 | 0.51 | 729.05 |
| native 持久绑定，显式 dirty 同步 | 810.64 | 845.77 | 0.66 | 739.23 |
| 预转换逻辑 FP16，普通提交 | 1085.06 | 1147.76 | 323.65 | 698.09 |
| 预整理 native FP16，普通 pass-through | 794.65 | 881.36 | 9.64 | 729.62 |

**持久绑定不是当前可立题的加速机制。** 相对最强普通提交，总耗时中位数只差
1.40 ms（0.18%）。它确实进一步减少输入阶段工作，但目前没有稳定的十步净收益。
按轮次配对，持久绑定在 8/12 轮更快，差值中位数为 20.07 ms；样本波动较大，
不能将这个数与“两个中位数之差”混用。20,000 次按轮次重采样的差值中位数
区间为 [-29.92, 61.41] ms（seed 812），跨零。这只是单次短测量的描述性
稳定性检查，不是跨独立会话/热状态的总体置信结论。两次重复的位一致检查轮
也出现普通 pass-through 更快，不使用那一轮的时延作正式收益数字。

一次性准备另计：host FP32 storage copy 1.05 ms；native pack/sync 6.35 ms；
逻辑 FP16 conversion 2.36 ms；从已 pack 的 native buffer 复制为 host array
0.59 ms。**pass-through 的 0.59 ms 不是从逻辑输入完成 native packing 的成本**；
当前控制复用持久路径已生成的 bytes，若独立部署需额外完成等价 packing。
不把所有 context 的准备时间加到任一单独方案，也不把常驻解释成 SRAM 驻留。

预转换 FP16 没变快，说明仅 dtype conversion 不能解释现象；绕过逻辑布局
适配的普通 pass-through 吸收主要收益。但未拆 SDK 内部转换/复制/提交成本，
不能进一步声称是哪一个内部 kernel、cache flush 或 DDR 瓶颈。
dirty 同步没有测出收益，默认保守路径继续作为基线。run 的变化也没有被
定位成因，不用单次 stage 时间归因编译器或硬件。

这一结果否定的是“持久绑定相对强普通提交有大收益”，**不否定图内静态
K/V 投影 hoisting**；后者仍需单独 cold/warm 图验证，不能借接口的 23% 升格。
后续异构路径必须同时比较 native pass-through 与持久绑定控制。

数值审计：六模式每步 velocity/latent 最大绝对差均为 0；改变 prefix 后
最终 latent 与原来相差 2.64521，证明失效测试真的改变了计算。随后补充轮用
FP32 storage 的 uint32 视图逐位比较，覆盖固定 suffix、十步、改变 prefix、
恢复 prefix 和上下文切换；全部通过。与 CPU 原 checkpoint 的差异和闭环未测。

证据：[六模式正式计时](../results/smolvla_libero/vision_probes/native_flow_strong_paired.json)、
[频率与 busy](../results/smolvla_libero/vision_probes/native_flow_strong_load.json)、
[逐位验证轮](../results/smolvla_libero/vision_probes/native_flow_strict_parity.json)。
正式计时标记内 NPU/DRAM 采样固定为 1 GHz/2.112 GHz；CPU 未连续采样且该
flow probe 未绑核，热稳态/后台干扰控制仍不充分。
六模式控制共用一个 probe 进程，legacy/native/dirty 三个 context 同时存在；
这是排除简单机制的实验，不是单 context 部署的 RSS、能耗或长期吞吐结果。

## 3. 原图 I/O ABI 与同步

原 `/root/smolvla-libero-rknn/denoise32/denoise_step_addmask.rknn` 为 65 输入、
1 输出。suffix 逻辑/native 都是 FP16 UNDEFINED `[1,50,480]`，48000 bytes。
64 个 key/value 逻辑接口均为 NHWC `[1,5,64,149]`；这对应原 `[1,149,5,64]`
fixture 经过 `transpose(0,2,3,1)` 的连续数组。

每个 native prefix 是 NC1HWC2 `[1,19,5,64,8]`，97280 bytes，149 个 channel
padding 到 152。实现逐字段检查 format、dim、width stride、height stride 和
size，保留零 padding；不假定任意张量都能直接绑定。

64 个 prefix native buffer 合计 6225920 bytes；逻辑 FP16 为 6103040 bytes；
逻辑 FP32 为 12206080 bytes。它们不是模型权重、整个进程 RSS 或每步 DDR 流量。
完整属性见 [I/O 审计](../results/smolvla_libero/vision_probes/native_denoise_io.json)。

默认持久模式在每次新 prefix 后显式 pack/sync 一次，每步更新 suffix；
仍保留 SDK 默认输入同步。dirty 模式仅使用 SDK 文档允许的
`RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE`，对每个 CPU 写入的 prefix 和 suffix
执行 `rknn_mem_sync(TO_DEVICE)`。dirty context 从不调用 `rknn_inputs_set`。
输出同步未关闭，依旧 `rknn_outputs_get(want_float=1)`。

## 4. 六个强控制

| 模式 | 静态条件 | 每步输入动作 | 用途 |
|---|---|---|---|
| legacy_c_api_prepacked | 预整理逻辑 FP32 | 普通 inputs_set 全输入 | 与原 RKNNLite API 数值校验，去掉 Python 重排的控制 |
| native_rewrite_prefix | 每步 pack 原生 FP16 | 重写/sync prefix 与 suffix | 同一种 native packing 的 lifetime 控制 |
| native_resident_default_sync | 一次 pack、持久绑定 | 只 CPU 写 suffix，默认 SDK 同步保留 | 最保守常驻路径 |
| native_resident_dirty_sync | 同上，显式管理写后同步 | 只写/sync suffix，禁自动输入 flush | 检验 dirty 输入同步，而非假设它更快 |
| legacy_fp16_prefix | 一次转逻辑 FP16 | 普通 inputs_set 全输入 | 排除每次 FP32 -> FP16 转换这个简单替代 |
| legacy_native_prefix_passthrough | 一次整理 native FP16 host array | 普通 inputs_set，prefix pass_through=1 | 排除一次布局整理但仍重复提交这个强替代 |

suffix 输入在 legacy 模式均为 FP32；native 模式显式转 FP16，均与原图实际
native precision 一致。所有模式保留相同 CPU FP32 solver，不把 Euler 塞进
FP16 图。mode 5 的 native pass-through 是逐输入标志，不是 zero-copy binding。

## 5. 数值与计时门槛

- C API legacy fixture 与 RKNNLite 原路径先做输出位一致检查。
- 每种模式、每一步 velocity 和 FP32 latent 与 legacy 控制位一致；不放宽阈值。
- 换 prefix 后 value 改变，要求最终动作确实变化；所有路径匹配新的 legacy
  轨迹，恢复 prefix 后回到原轨迹。正式计时中每个最终结果再次校验。
- 模型/context 常驻；轮换计划顺序。测 suffix embedding、输入准备/同步、
  run、取回和 FP32 solver；整个十步计时包含控制器与检查的共同开销。
- 一次性 prefix 转换/同步另测；common NumPy 初始重排与初始化不在十步数字内。
- NPU/DRAM 频率由 sampler 记录；driver busy 不是 MAC occupancy/带宽。
- 合成 prefix 的位一致不证明实际观测分布或闭环质量，更不证明完整 VLA 的收益。

## 6. 复现

`scripts/smolvla_native_denoise.cpp` 在板上编为共享库，链接当前 `librknnrt`：

```sh
g++ -O2 -std=c++17 -fPIC -shared smolvla_native_denoise.cpp \
  -I/root/rk-llama.cpp/ggml/src/ggml-rknpu2/libs/include \
  -L/root/rk-llama.cpp/ggml/src/ggml-rknpu2/libs -lrknnrt \
  -Wl,-rpath,/root/rk-llama.cpp/ggml/src/ggml-rknpu2/libs \
  -o libsmolvla_native_denoise.so
```

`scripts/probe_smolvla_native_flow.py` 使用已安装 board policy 的 `embed_suffix`，
按实际 `sample_actions` 的 `time = 1 + step * dt`、`x_t = x_t + dt * velocity`
更新十次。参数包括 `--library`、`--rknn-model`、`--data-dir`、`--model-path`、
`--vlm-path`、`--repeats 12`、`--output`。以 `sample_rknpu_load.py` 包装记录
阶段标记与频率。大型 weights/fixtures/library 留在 board，不入 Git。

`scripts/inspect_rknn_io.cpp` 输出逻辑与 native attrs；原生 helper 的 init flags
参数默认为 0，旧视觉调用语义不变，不将 dirty 同步扩散到视觉路径。
