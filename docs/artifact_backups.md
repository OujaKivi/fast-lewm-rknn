# RKNN 制品备份（2026-09-24）

本快照只收录已用于当前结果或明确作为对照的模型图，不收集板上的全部历史实验文件。大文件不进入 Git；Git 中的 [`artifacts/2026-09-24.sha256`](../artifacts/2026-09-24.sha256) 是 33 个文件的校验清单。

两份板外副本使用相同的相对目录：

| 位置 | 根目录 |
|---|---|
| Mac | `/Users/wangjiwei/Doubao/model-artifact-backups/fast-lewm-rknn/2026-09-24/` |
| RTX 主机 | `/home/wang/model-artifact-backups/fast-lewm-rknn/2026-09-24/` |

## 按模型划分

| 模型 | 快照内容 | 板上来源 | 用途 |
|---|---|---|---|
| Fast-LeWM / PushT | `fast-lewm/Fast-lewm_pusht_object.ckpt` | `/root/Fast-LeWorldModel/weights/` | 导出权重 |
| Fast-LeWM / PushT | `fast-lewm/full_model_state.pt` | `/root/Fast-LeWorldModel/weights/` | 当前规划服务实际加载的运行时权重 |
| Fast-LeWM / PushT | `fast-lewm/vit_encoder_projected_fp16.rknn` | `/root/Fast-LeWorldModel/` | 当前 NPU 图像路径 |
| Fast-LeWM / PushT | `fast-lewm/predictor_terminal_with_proj_b{300,150,64}_fp16.rknn` | `/root/Fast-LeWorldModel/` | 当前 300/150/64 候选调度 |
| Fast-LeWM / PushT | `fast-lewm/action_encoder_terminal_b300_fp16_conv.rknn` | `/root/Fast-LeWorldModel/` | 完整 NPU 动作编码器精度、性能对照；不是当前纯 NPU 预测模式的默认动作路径 |
| Fast-LeWM / PushT | `fast-lewm/action_encoder_terminal_b{64,100}_fp16_conv.rknn` | `/root/Fast-LeWorldModel/` | 候选级 CPU/NPU 协同对照 |
| SmolVLA base | `smolvla-base/prefill_addmask.rknn`, `smolvla-base/denoise_step_addmask.rknn` | `/root/smolvla-rknn-prefill/`, `/root/smolvla-rknn-denoise/` | 单相机、70-token Prefill、16 层动作专家的合成输入剖析 |
| SmolVLA LIBERO | `smolvla-libero/prefill_addmask.rknn`, `smolvla-libero/denoise_step_addmask.rknn` | `/root/smolvla-libero-rknn/prefill32/`, `/root/smolvla-libero-rknn/denoise32/` | 双相机、149-token Prefill、32 层动作专家的 task 0 闭环 |
| SmolVLA base 与 LIBERO 共用 | `shared-vision/vision_connector_fp16.rknn` | `/root/smolvla-rknn-vision/` | 已验证两种 checkpoint 的视觉/连接器权重一致，只保存一份图 |
| SmolVLA 权重 | `checkpoints/smolvla_base/`, `checkpoints/smolvla_libero/`, `checkpoints/smolvlm_config/` | `/root/models/` 中同名目录 | 对应权重、预处理/后处理参数、VLM 配置与 tokenizer |

这里“共用视觉图”只指模型部署：LIBERO 模拟器运行在 RTX 主机，产生的两路相机图像送往 RK3588 做策略推理；RK 上对两路图像分别调用同一张视觉 RKNN 图。base 与 LIBERO checkpoint 中共享的视觉/连接器权重已逐张量核对相同，并非把模拟器移到 RK 或让两个实验共用图像。LIBERO 源模型 revision 为 `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。实际备份文件的 SHA-256 是更直接的身份依据。SmolVLA base 使用的是实验时板上的 checkpoint 快照，其精确字节也由校验清单固定。两者的 Prefill/去噪图**不可互换**。

## 恢复与再生成

在 Mac 副本根目录校验：

```sh
shasum -a 256 -c SHA256SUMS
```

在 RTX 主机副本根目录校验：

```sh
sha256sum -c SHA256SUMS
```

校验后按上表复制回板上来源目录。恢复时保留文件名；Fast-LeWM 的运行时路径见 [`rk3588_planner_server.py`](../rk3588_planner_server.py)，SmolVLA 的运行参数见[机器交接文档](test_hosts.md)。不要把 base 的 Prefill/去噪图用于 LIBERO。

源代码保留了 Fast-LeWM 的 [`export_vit_encoder.py`](../Fast-LeWorldModel/export_vit_encoder.py)、[`export_terminal_predictor.py`](../Fast-LeWorldModel/export_terminal_predictor.py)、[`export_action_encoder.py`](../Fast-LeWorldModel/export_action_encoder.py) 与 [`convert_to_rknn.py`](../Fast-LeWorldModel/convert_to_rknn.py)，以及 SmolVLA 的视觉、Prefill、去噪导出与 mask 改写脚本。SmolVLA 的图形状和转换顺序见 [`smolvla_rknn.md`](smolvla_rknn.md) 与 [`smolvla_libero_closed_loop.md`](smolvla_libero_closed_loop.md)。板上 RKNN Lite2 为 2.3.2；文档使用 Toolkit2 2.3.2。保存的已验证图用于**直接恢复实验**；由于未保存每张图的原始及改写后 ONNX、全部导出参数与转换日志，目前不能声称从权重重新转换会得到逐字节相同的 RKNN。这是明确的再生成限制，不应把新图默认视作旧图。

精度校验记录和闭环结果仍在 `results/`。再次转换时，应先比对对应 ONNX 与 PyTorch、RKNN 与 ONNX，再跑任务；旧图的校验结论不能自动移植给新图。
