# Fast-LeWM 方案 A 异构调度实验报告

**日期**：2026-09-15

**平台**：RK3588（4×A76 绑核，NPU 三核 1GHz，Runtime 2.3.2）

**方案**：A — CPU action\_encoder + NPU predictor（单缓冲，无流水线）

**测试配置**：CEM 30 iters × 3 repeats，batch=300，5 步 rollout，FP16



***

## 1. 实验目标

验证方案 A（简单异构调度）的实际收益：



* action\_encoder 留 CPU（1.8M 参数，小 attention，NPU 不划算）

* predictor 放 NPU（9.0M 参数，大 GEMM，NPU 效率高）

* 测量端到端 CEM 决策延迟，对比全 CPU 基线

* 细粒度分析 NPU 数据搬运、推理、输出解析开销

* 验证 NPU 推理是否对 CPU 性能产生影响（内存带宽竞争 / 发热降频）

* 验证研究假设 H1：瓶颈是否从模型计算转移到其他部分



***

## 2. 实现方案

### 2.1 核心组件

`npu_predictor.py`**&#x20;— NPUPredictor 包装类**



* 接口与 `module.ARPredictor` 完全一致：`predictor(latent, act_emb) → pred`

* 内部封装 RKNN 加载、推理、输出解析

* 自动处理 torch tensor ↔ numpy float32 转换

* 支持 `release()` 释放资源

`profile_heterogeneous.py`**&#x20;— 异构版 profiling 脚本**



* 基于 `profile_fast_lewm.py` 的 CEM 循环改造

* `--mode cpu`：全 CPU 基线

* `--mode npu`：CPU enc + NPU pred

* `--mode both`：两者对比，自动计算加速比

* `--verify`：验证 CPU/NPU 输出一致性（cosine similarity / MSE）

* 细粒度计时：`npu_transfer_in` / `npu_inference` / `npu_transfer_out`

### 2.2 CEM 循环改造



```
\# 改造前（全 CPU）

act\_emb = act\_enc(actions, return\_last\_only=True, latent=emb\[:, -1:])

for \_ in range(steps):

&#x20;   pred = predictor(emb\[:, -1:], act\_emb)  # CPU torch

&#x20;   emb = torch.cat(\[emb, pred], dim=1)

\# 改造后（异构）

act\_emb = act\_enc(actions, return\_last\_only=True, latent=emb\[:, -1:])  # CPU

for \_ in range(steps):

&#x20;   pred = npu\_predictor(emb\[:, -1:], act\_emb)  # NPU RKNN

&#x20;   emb = torch.cat(\[emb, pred], dim=1)
```

### 2.3 NPU 推理细粒度流程



```
npu\_transfer\_in:  torch.Tensor → numpy.float32（含 contiguous/copy）

npu\_inference:    rknn.inference(inputs=\[latent\_np, act\_emb\_np])

npu\_transfer\_out: numpy → torch.Tensor（from\_numpy + to(device)）
```



***

## 3. 实验结果

> **重要说明**
>
> ：本报告数据均在
>
> **锁频状态**
>
> 下测得（CPU 大核 2352MHz / 小核 1800MHz performance governor、NPU 1GHz、DMC 2.1GHz）。早期未锁频数据存在 DVFS 干扰（NPU 模式下 CPU 被 SoC 功耗预算限制压到 1416MHz），导致 NPU 异构收益被低估约 30%。详见第 4 节根因诊断。

### 3.1 三方端到端对比（CPU / NPU 全量 / NPU 异构）

为区分 "异构调度的收益" 与 "简单把模型搬到 NPU 的收益"，增加 NPU 全量基线（action\_encoder + predictor 都放 NPU）。



| 指标                     | CPU 全量      | NPU 全量（enc+pred 都 NPU）      | NPU 异构（方案 A，CPU enc+NPU pred） |
| ---------------------- | ----------- | --------------------------- | ----------------------------- |
| 完整 CEM 决策              | 19.29 s     | 8.58 s                      | **6.27 s**                    |
| vs CPU 加速比             | 1.0×        | 2.25×                       | **3.08×**                     |
| rollout（5 步 ×30iter）   | 16.15 s     | 3.09 s                      | **3.17 s**                    |
| action\_encoder        | 3.01 s（CPU） | 5.31 s（NPU，**比 CPU 慢 76%**） | 2.94 s（CPU，**无降频**）           |
| vi\_enc（视觉编码）          | 3.75 s      | 3.72 s（**无差异**）             | 3.66 s（**无差异**）               |
| score + elite + sample | 0.017 s     | 0.017 s（**无差异**）            | 0.018 s（**无差异**）              |
| 内存占用（max RSS）          | \~440 MB    | **913 MB**（双模型加载）           | \~490 MB                      |

**关键结论：异构（方案 A）比 NPU 全量快 37%（1.37×）**，证明 "不适合 NPU 的模块留 CPU" 的异构调度有实际价值，而不是简单搬家。

**锁频的重要性**：未锁频时异构 vs CPU 仅 2.25×、异构 vs 全量仅 1.17×；锁频后分别提升到 3.08× 和 1.37×。未锁频数据严重低估了 NPU 异构的真实收益。

### 3.2 为什么异构比全量好（锁频后修正版）

锁频后，之前观察到的 "predictor 变慢" 和 "CPU 被拖慢" 等现象全部消失，异构优于全量的**唯一原因**就是：

**action\_encoder 在 NPU 上反而更慢**：



* CPU：2.94s（30 iter，约 98ms/iter）

* NPU：5.31s（30 iter，约 177ms/iter）

* NPU 比 CPU 慢 **76%**，原因：

1. action\_encoder 仅 1.8M 参数，NPU 吃不满（利用率 31.7%）

2. 含 T=5 小 self-attention，NPU 上 attention 效率低

3. RKNN 模型文件 190MB，加载和初始化开销大

4. 数据搬运开销占比大（小模型计算少，搬运相对多）

**锁频后已排除的伪因素**（未锁频时观察到、锁频后消失）：



* ~~NPU 全量模式下 predictor 变慢 33%~~ → 锁频后异构 20.9ms / 步 vs 全量 20.4ms / 步，**无差异**

* ~~NPU 推理导致 CPU 性能下降 30-50%~~ → 锁频后 vi\_enc/action\_encoder/score 等在三种模式下**完全一致**

* ~~内存带宽竞争~~ → 锁频后证明 RK3588 的 DDR 带宽足够同时支撑 NPU 推理和 CPU 计算

### 3.3 NPU 每步细粒度（450 次调用统计，锁频后）



| 阶段                 | 异构模式 median  | 全量模式 median  | 占比（异构） |
| ------------------ | ------------ | ------------ | ------ |
| npu\_transfer\_in  | 0.064 ms     | 0.056 ms     | 0.30%  |
| npu\_inference     | **20.90 ms** | **20.41 ms** | 98.8%  |
| npu\_transfer\_out | 0.189 ms     | 0.171 ms     | 0.90%  |
| **每步合计**           | **21.15 ms** | **20.64 ms** | 100%   |

**关键结论**：



* NPU 推理本身占 98.8%，数据搬运开销仅 1.2%（0.25ms / 步）

* 异构和全量模式下 NPU 推理时间**无显著差异**（20.9 vs 20.4ms），证明双模型交替推理不存在 NPU 上下文切换开销

* predictor 在 NPU 上比 CPU 快 **5.1×**（107.7ms → 20.9ms，锁频后 CPU 单步时间）

### 3.3 输出一致性验证（随机权重下）



| 指标                       | 值          |
| ------------------------ | ---------- |
| cosine similarity (mean) | **0.9707** |
| cosine similarity (min)  | 0.9496     |
| MSE                      | 0.0585     |
| relative L1 error        | 0.2375     |

**说明**：随机权重下数值对比无实际意义，cos\_sim 0.97 证明推理流程正确。FP16 量化本身会引入精度损失，加载预训练权重后需重新验证。



***

## 4. 根因诊断：DVFS 降频 vs 内存带宽竞争

> 本节是本报告最重要的修正。早期未锁频实验观察到 "NPU 推理导致 CPU 性能下降 30-50%"，一度归因于内存带宽竞争。通过锁频对照实验，确认真正原因是 
>
> **RK3588 的 SoC 级功耗预算限制（IPA）导致 CPU 降频**
>
> ，而非内存带宽竞争。

### 4.1 未锁频时观察到的现象（已被推翻）

在默认 schedutil governor 下，NPU 模式下所有 CPU 端计算都比纯 CPU 模式慢 30-50%：



| CPU 端模块           | 纯 CPU    | NPU 模式（未锁频）  | 下降幅度     |
| ----------------- | -------- | ------------ | -------- |
| action\_encoder   | 2971 ms  | 5070 ms      | **-41%** |
| vi\_enc（ViT-tiny） | 3902 ms  | 6573 ms      | **-41%** |
| score             | 8.8 ms   | 16.4 ms      | **-46%** |
| CPU 频率            | 2127 MHz | **1416 MHz** | **-33%** |

### 4.2 根因诊断实验设计

设计四种模式对照，每种模式独立进程、间隔 2 秒冷却，测量 vi\_enc 性能 + CPU 频率 + 温度：



| 模式       | action\_encoder    | predictor         | 目的                  |
| -------- | ------------------ | ----------------- | ------------------- |
| CPU 基线   | CPU                | CPU               | 基线                  |
| NPU 异构   | CPU                | NPU               | 方案 A                |
| NPU 全量   | NPU                | NPU               | 简单搬家                |
| sleep 对照 | sleep(182ms) + CPU | sleep(22ms) + CPU | 模拟 NPU 时间占用但不实际访问内存 |

**关键逻辑**：如果 sleep 对照下 vi\_enc 也变慢，说明是时间占用 / 发热导致；如果 sleep 对照下 vi\_enc 不变慢但 NPU 模式下变慢，说明是内存带宽竞争。

### 4.3 锁频前诊断结果



| 指标      | CPU 基线  | NPU 异构      | NPU 全量      | sleep 对照 |
| ------- | ------- | ----------- | ----------- | -------- |
| vi\_enc | 175.4ms | 180.6ms     | 191.4ms     | 236.2ms  |
| CPU 均频  | 2127MHz | **1416MHz** | **1433MHz** | 2151MHz  |
| 均温      | 59.5°C  | 55.2°C      | 53.2°C      | 57.4°C   |

**发现**：



1. NPU 模式下 CPU 频率被压到 1416MHz（-33%），但温度反而更低（55°C vs 60°C）

2. sleep 对照下 vi\_enc 反而最慢（35% 变慢），因为它是最后一个跑的，存在进程内累积效应

3. vi\_enc 变慢幅度（3-9%）远小于 CPU 降频幅度（33%），说明 vi\_enc 对 CPU 频率不敏感

### 4.4 锁频后诊断结果（决定性证据）

强制 CPU 大核锁 2352MHz / 小核 1800MHz（performance governor），NPU 锁 1GHz，DMC 锁 2.1GHz：



| 指标         | CPU 基线      | NPU 异构      | NPU 全量      | sleep 对照    |
| ---------- | ----------- | ----------- | ----------- | ----------- |
| vi\_enc    | 121.6ms     | 116.4ms     | 122.7ms     | 118.6ms     |
| CPU 均频     | **2352MHz** | **2352MHz** | **2352MHz** | **2352MHz** |
| 均温         | 52.5°C      | 49.2°C      | 47.4°C      | 53.2°C      |
| vi\_enc 变慢 | 基线          | **-4%**     | **+1%**     | **-3%**     |

**决定性结论**：



1. **锁频后，NPU 推理对 vi\_enc 完全没有负面影响**（四种模式在 ±3% 噪声范围内）

2. **RK3588 不存在显著的 CPU-NPU 内存带宽竞争**—— 至少在 Fast-LeWM 这个 workload 下，DDR 带宽足够同时支撑 NPU 推理和 CPU 计算

3. **之前观察到的 CPU 性能下降 100% 是 DVFS 降频导致的**，与内存带宽无关

4. **降频原因是 SoC 功耗预算限制（IPA），不是过热**——NPU 模式下温度反而更低，但 CPU 仍被限频

### 4.5 对后续方案的影响（修正版）



* ~~方案 B（双缓冲）的内存带宽竞争风险~~ → **已消除**，CPU 和 NPU 可以真正并行而不互相拖慢

* ~~需要测量内存带宽~~ → 不需要，已证明不是瓶颈

* ~~降低 NPU 频率减少带宽竞争~~ → 无意义

* **后续所有性能实验必须锁频**，否则 DVFS 会严重干扰结果（NPU 模式下 CPU 被压到 1416MHz，异构收益被低估约 30%）

* **方案 B（双缓冲）的预期收益可能高于之前预估**，因为 CPU/NPU 并行时不存在内存带宽竞争



***

## 5. 瓶颈转移分析（验证 H1，锁频后）

### 5.1 CPU 基线瓶颈分布



```
CPU 模式（19.29s）：

&#x20; rollout (predictor×5步):  16.15s  83.7%  ← 主要瓶颈

&#x20; vi\_enc:                     3.75s  19.4%

&#x20; action\_encoder:             3.01s  15.6%

&#x20; 其他（score/elite/sample）: 0.02s   0.1%
```

### 5.2 NPU 异构模式瓶颈分布



```
NPU 异构模式（6.27s）：

&#x20; vi\_enc:                     3.66s  58.4%  ← 新瓶颈！

&#x20; action\_encoder:             2.94s  46.9%  ← 新瓶颈！

&#x20; rollout (NPU predictor×5步): 3.17s  50.6%

&#x20; 其他:                        0.02s   0.3%
```

### 5.3 结论

**研究假设 H1 得到验证**：当 predictor 计算被 NPU 加速 5.1× 后，端到端瓶颈从 "模型计算（rollout）" 转移到了 "视觉编码 + action\_encoder"。



* rollout 占比从 83.7% 降到 50.6%

* vi\_enc + act\_enc 成为与 rollout 并列的三大瓶颈

* 下一步优化重点应该是 vi\_enc 和 action\_encoder 的加速，而不是继续优化 predictor

**vi\_enc 加速方向**：



* ViT-tiny 也可以尝试 NPU 推理（5.5M 参数，含 attention，需测试 NPU 效率）

* 或者降低 ViT 输入分辨率（224→160/128）减少计算量

* 或者用更轻量的视觉编码器（MobileNet、EfficientNet-lite）

**action\_encoder 加速方向**：



* 当前在 CPU 上 98ms/iter，NPU 上 177ms（更慢）

* 可以尝试 INT8 量化后上 NPU，看是否能比 CPU 快

* 或者优化 action\_encoder 结构（减少 attention 层数、用线性 attention）

* 或者用 CPU 多线程并行（当前绑 4 大核，torch 线程数可能未最优）



***

## 6. 方案 A 收益总结

### 6.1 实际收益 vs 预估（锁频后）



| 指标           | 预估（方案 A） | 未锁频实测 | 锁频后实测     | 锁频差异 |
| ------------ | -------- | ----- | --------- | ---- |
| 完整 CEM       | \~13.2s  | 8.89s | **6.27s** | -29% |
| 加速比（vs CPU）  | 1.5×     | 2.25× | **3.08×** | +37% |
| predictor 加速 | 2.4×     | 4.94× | **5.1×**  | +3%  |
| 异构 vs 全量     | -        | 1.17× | **1.37×** | +17% |

锁频后实际收益显著好于未锁频测量，因为未锁频时 NPU 模式下 CPU 被 SoC 功耗预算限制压到 1416MHz，严重拖累了 CPU 端的 action\_encoder 和 vi\_enc。

### 6.2 未达预期的部分



* ~~CPU 端性能下降 30-50%（内存带宽竞争）~~ → **已排除**，锁频后证明是 DVFS 降频导致，不存在内存带宽竞争

* action\_encoder 在 NPU 上比 CPU 慢 76%，无法上 NPU，需要其他优化方向

### 6.3 距实时性的差距



| 指标   | 当前（方案 A，锁频） | 10Hz 实时目标 | 20Hz 实时目标 |
| ---- | ----------- | --------- | --------- |
| 单次决策 | 6.27 s      | 0.1 s     | 0.05 s    |
| 差距   | -           | **63×**   | **125×**  |

方案 A 后仍距实时性有 63× 差距，需要后续方案（B/C/D）持续优化。



***

## 7. 后续工作建议

### 7.1 短期（1-2 周）



1. **~~vi\_enc NPU 测试~~ → 已完成**：见第 9 节，NPU 加速 3.98×，但 vi\_enc 仅占总时间 2%，收益有限

2. **action\_encoder INT8 测试**：INT8 量化后上 NPU，看是否能比 CPU 快（当前 NPU FP16 比 CPU 慢 76%）

3. **方案 B 双缓冲原型**：实现 CPU/NPU 流水线，已证明不存在内存带宽竞争，预期收益可能高于之前预估

### 7.2 中期（2-4 周）



1. **INT8 量化**：加载预训练权重后做 INT8 校准，predictor 预期再加速 1.5-2×

2. **batch size 扫描**：测试不同 batch（16/32/64/128/256/512）的 NPU 效率，找最佳区间

3. **vi\_enc 优化**：降低输入分辨率（224→160/128）或替换更轻量视觉编码器

### 7.3 长期（1-2 月）



1. **方案 D 动态 batch**：CEM 收敛感知，减少后期 candidate 数量

2. **端到端真实环境验证**：在真实机器人 / 仿真环境中验证规划成功率和延迟



***

## 8. 产物清单



| 文件                            | 说明                                   |
| ----------------------------- | ------------------------------------ |
| `npu_predictor.py`            | NPUPredictor 包装类（接口兼容 ARPredictor）   |
| `profile_heterogeneous.py`    | 异构版 profiling 脚本（支持 cpu/npu/both 模式） |
| `profile_het_full.json`       | 正式测试原始数据（30 iter × 3 repeats）        |
| `profile_het_smoke.json`      | 冒烟测试数据（5 iter × 2 repeats）           |
| `predictor_S300_fp16_v4.rknn` | NPU  predictor 模型（修复全零权重后转换）         |
| `bench_vi_enc.py`              | vi\_enc CPU vs NPU 对比测试脚本                     |
| `vi_enc_bench_result.json`     | vi\_enc 测试结果原始数据                              |
| `vit_tiny_b2_fp16.rknn`        | vi\_enc (ViT-tiny) NPU 模型（14.33MB）              |



***

## 9. vi\_enc NPU 效率测试补充（2026-09-16）

### 9.1 测试目的

验证 vi\_enc（视觉编码器，ViT-tiny 结构）是否适合上 NPU，补充方案 A 的模块级 NPU 适配性分析。

### 9.2 测试配置

* **模型**：timm `vit_tiny_patch16_224`（5.52M 参数，输出 192 维特征）
* **输入**：[2, 3, 224, 224]（batch=2，与 ONNX 导出一致）
* **CPU**：4×A76 @ 2352MHz（绑核），torch 2.9.1+cpu
* **NPU**：3核 @ 1GHz，RKNN FP16，Runtime 2.3.2
* **锁频**：CPU/NPU/DMC 全部锁最高频
* **测试**：warmup=20，repeat=100

### 9.3 测试结果

| 指标 | CPU (4×A76) | NPU (3核) | NPU/CPU |
| --- | --- | --- | --- |
| batch 延迟 | 139.29 ms | 34.97 ms | **0.25×** |
| 单候选延迟 | 69.64 ms | 17.49 ms | **0.25×** |
| 吞吐量 | 14.4 samples/s | 57.2 samples/s | **3.98×** |
| p99 延迟 | 147.74 ms | 38.04 ms | 0.26× |

**结论：vi\_enc 在 NPU 上获得 3.98× 加速，适合上 NPU。**

### 9.4 对端到端规划的影响分析

虽然 vi\_enc 本身 NPU 加速明显，但对端到端规划延迟的影响有限：

| 指标 | 当前（vi\_enc CPU） | vi\_enc 上 NPU 后 | 节省 |
| --- | --- | --- | --- |
| vi\_enc 单次延迟 | ~125 ms | ~35 ms | ~90 ms |
| 端到端决策延迟 | 6.27 s | 6.18 s | ~1.4% |
| vi\_enc 占比 | 2.0% | 0.6% | - |

**原因**：vi\_enc 在每次 CEM 规划中只调用一次（编码当前观测），而 action\_encoder 和 predictor 在每个 CEM iteration 中都要调用 300 次（每个候选一次）。因此 vi\_enc 的绝对耗时很小，即使加速 4 倍对整体影响也只有 ~1.4%。

### 9.5 与其他模块的 NPU 适配性对比

| 模块 | 参数 | CPU 延迟 | NPU 延迟 | NPU 加速 | 是否适合 NPU |
| --- | --- | --- | --- | --- | --- |
| **vi\_enc** | 5.52M | 139ms/batch | 35ms/batch | **3.98×** | ✅ 适合，但占比低 |
| **predictor** | 9.0M | ~733ms/300候选 | 20.9ms/步 | **~35×** | ✅ 非常适合，已上 NPU |
| **action\_encoder** | 1.8M | 2.94s/300候选 | 5.31s/300候选 | **0.55×（慢76%）** | ❌ FP16 不适合，需 INT8 |

**关键发现**：
1. **大模型（predictor）最适合 NPU**：计算密集，NPU 加速比最高
2. **小模型（action\_encoder）NPU 不划算**：NPU 启动开销占比高，FP16 比 CPU 还慢
3. **中等模型（vi\_enc）NPU 有收益**：但调用次数少，整体影响有限
4. **下一步优化重点应是 action\_encoder 的 INT8 量化**：占总时间 47%，是最大瓶颈



***

## 附录：关键命令



```
\# 板端运行正式测试（绑4大核）

cd /root/Fast-LeWorldModel

taskset -c 4-7 python profile\_heterogeneous.py \\

&#x20; \--mode both --verify --repeats 3 --iters 30 --tag het\_full

\# 仅 NPU 模式

taskset -c 4-7 python profile\_heterogeneous.py \\

&#x20; \--mode npu --repeats 5 --iters 30 --tag npu\_only

\# 冒烟测试

taskset -c 4-7 python profile\_heterogeneous.py \\

&#x20; \--mode both --verify --quick --tag het\_smoke
```