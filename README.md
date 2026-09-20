# Fast-LeWM RK3588 端侧异构执行优化

## 项目简介

在 RK3588 开发板上部署 Fast-LeWorldModel，通过 NPU 加速视觉编码器，探索端侧异构执行优化的可行性与瓶颈。

---

## 核心 Profiling 结果（对齐后正确版本）

![对齐后性能曲线](aligned_batch_performance_curve.png)

**关键发现：**

- ✅ **ViT NPU 加速 2.39x**（138ms → 58ms）
- ✅ **Predictor NPU 只慢 8%**（424ms → 460ms，对齐后正确结果）
- ✅ **拐点在 batch=192**：小 batch NPU 更快，大 batch CPU 略快
- 📉 **数据传输仅 0.15ms**，根本不是瓶颈
- 🎯 **总耗时 NPU 快 21%**（655ms → 518ms）

---

## 后续规划

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| P0 | 端到端重新测试（对齐后正确实现） | 验证 NPU 真实性能提升 |
| P0 | ViT NPU + Predictor NPU 全异构 | 预期端到端加速 20%+ |
| P1 | INT8 量化 Predictor NPU | 预期再快 30% |
| P1 | 扩大测试规模到 50 episodes | 获得更可靠的成功率数据 |
| P2 | 三核 NPU 并行拆分 | 理论 1.5-2x 加速 |
| P2 | 双缓冲流水线 | 减少空等，预期 10-15% |

---

## 环境信息

- **硬件**：RK3588（16GB RAM，6TOPS NPU）
- **软件**：Ubuntu 22.04/aarch64, RKNN Runtime 2.3.2
- **模型**：Fast-LeWM（PushT 任务，预训练权重）

---

**完整进度记录见 [PROGRESS.md](PROGRESS.md)**
