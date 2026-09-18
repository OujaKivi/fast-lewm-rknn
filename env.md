# Fast-LeWM 端侧异构部署环境记录

> 本文档记录所有设备、环境、编译、执行信息，确保后续复现可对齐。
> 最后更新：2026-09-18

---

## 1. 设备清单

### 1.1 Mac 开发机（环境模拟 + 评估客户端）
| 项目 | 值 |
|------|-----|
| 设备 | Apple Mac mini / MacBook Pro |
| CPU | Apple M5 Pro (arm64) |
| 内存 | 统一内存（约 36GB+） |
| 操作系统 | macOS |
| 角色 | PushT 环境模拟、评估客户端、数据可视化、代码开发 |
| 项目目录 | `/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/` |

### 1.2 RK3588 开发板（端侧推理 + 完整规划）
| 项目 | 值 |
|------|-----|
| 设备 | RK3588 开发板 (hostname: YY3588) |
| IP | 192.168.77.2 |
| 用户 | root |
| 密码 | 123456 |
| CPU | 4×Cortex-A76 (大核) + 4×Cortex-A55 (小核), aarch64 |
| 内存 | 16GB LPDDR4 |
| NPU | 三核 6TOPS INT8, freq=1GHz, Runtime 2.3.2, driver v0.9.8 |
| 操作系统 | Ubuntu 22.04 LTS |
| 角色 | 完整规划 pipeline（ViT + ActionEnc + Predictor + CEM），支持 CPU/NPU 模式 |
| 代码目录 | `/root/Fast-LeWorldModel/` |
| 绑核命令 | `taskset -c 4-7`（绑大核） |

### 1.3 x86 Ubuntu 转换机（RKNN 模型编译）
| 项目 | 值 |
|------|-----|
| 设备 | x86 Ubuntu 服务器 (hostname: seu-server) |
| IP | 10.128.201.131 |
| 用户 | wjw |
| 密码 | wjww118xxx |
| 操作系统 | Ubuntu (x86_64) |
| 角色 | ONNX → RKNN 模型转换（需要 x86 环境运行 rknn-toolkit2） |
| 工作目录 | `/home/wjw/fast-lewm-npu/` |
| 约束 | 未经允许不要用 docker |

---

## 2. Conda 环境

### 2.1 Mac: stable-wm
| 项目 | 值 |
|------|-----|
| 环境名 | stable-wm |
| Python | 3.10.21 |
| torch | 2.4.0+cpu |
| transformers | 4.35.2 |
| stable-worldmodel | 0.1.1 |
| 其他 | pygame-ce, pymunk, lancedb, scikit-learn, numpy |
| 用途 | PushT 环境模拟、官方评估流程、Mac CPU 基线测试 |

### 2.2 RK3588: fast-lewm
| 项目 | 值 |
|------|-----|
| 环境路径 | `/root/miniconda3/envs/fast-lewm/bin/python` |
| Python | 3.10.x |
| torch | 2.9.1+cpu |
| rknn-toolkit-lite2 | 2.3.2（板端推理 runtime） |
| timm | 1.0.29 |
| transformers | 4.35.2（已从 5.17.0 降级，5.x 不兼容） |
| numpy | 1.26.x |
| 用途 | 板端完整规划 pipeline、CPU/NPU 推理、性能测试 |

### 2.3 x86: rknn-py310
| 项目 | 值 |
|------|-----|
| 环境路径 | `/home/wjw/miniconda3/envs/rknn-py310/bin/python` |
| Python | 3.10.21 |
| torch | 2.4.0+cpu |
| rknn-toolkit2 | 2.3.2（模型转换工具链） |
| numpy | 1.26.4 |
| onnx | 1.16.1 |
| onnxruntime | 已安装（用于 ONNX 精度验证） |
| 用途 | ONNX 导出、RKNN 模型转换（FP16/INT8） |

---

## 3. 模型参数（正确配置）

### 3.1 ViT (Vision Transformer)
| 参数 | 值 |
|------|-----|
| hidden_size | 192 |
| num_attention_heads | 3 |
| num_hidden_layers | 12 |
| intermediate_size | 768 |
| patch_size | 14 |
| image_size | 224 |
| add_pooling_layer | False |
| 输出维度 | 192 (latent) |

### 3.2 ActionPrefixEmbedder
| 参数 | 值 |
|------|-----|
| input_dim | 10 |
| emb_dim | 192 |
| use_latent_condition | True |
| latent_dim | 192 |
| transformer_depth | 3 |
| transformer_heads | 3 |
| transformer_dim_head | 64 |

### 3.3 ARPredictor（多步预测核心）
| 参数 | 值 |
|------|-----|
| depth | 6 |
| mlp_dim | 2048 |
| input_dim | 192 |
| hidden_dim | 192 |
| output_dim | 192 |
| heads | 8 |
| dim_head | 128 |
| 参数量 | 9,023,424 |
| 输入 | latent (B,1,192) + act_emb (B,5,192) |
| 输出 | pred_latent (B,5,192) —— 一次 forward 完成 5 步预测 |

> ⚠️ **历史错误**：之前转换 NPU 模型时误用了 heads=3, dim_head=64, mlp_dim=768，导致权重加载不正确，所有旧 NPU 性能测试结果无效。

### 3.4 PredProj (Projector)
| 参数 | 值 |
|------|-----|
| 结构 | MLP 192→2048→192 + BatchNorm1d |
| 参数量 | 792,768 |

---

## 4. 预训练权重

| 项目 | 值 |
|------|-----|
| HuggingFace repo | `naiverer/fast-leworldmodel` |
| 权重文件 | `Fast-lewm_pusht_object.ckpt` (69MB) |
| 完整模型权重 | `full_model_state.pt` (69MB, 349层) |
| 板端路径 | `/root/Fast-LeWorldModel/weights/full_model_state.pt` |
| x86 路径 | `/home/wjw/fast-lewm-npu/full_model_state.pt` |
| Mac 临时路径 | `/tmp/full_model_state.pt` |

---

## 5. 数据集

| 项目 | 值 |
|------|-----|
| HuggingFace repo | `galilai-group/lewm-pusht` |
| 格式 | Lance |
| 大小 | 13.6GB |
| 行数 | 2,336,736 |
| episodes | 18,685 |
| fps | 10 |
| 本地路径 | `/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/data/datasets/datasets--galilai-group--lewm-pusht/snapshots/ea321e392348e3c65a18ab0d685f00e57be2c3e0/pusht_expert_train.lance` |
| pixels 字段 | JPEG 编码的 bytes（需要 PIL 解码，不是 numpy array） |
| state 字段 | 7维 [agent_x, agent_y, block_x, block_y, block_angle, vel_x, vel_y] |

### StandardScaler 参数（从 233 万帧数据集拟合）
| 字段 | mean | std |
|------|------|-----|
| action | [-0.0078, 0.0069] | [0.2085, 0.2067] |
| proprio | [228.81, 292.23, -2.92, 2.55] | [103.37, 98.93, 77.50, 76.86] |
| state | [228.81, 292.23, 243.61, 275.25] | [103.37, 98.93, 71.62, 73.38] |

---

## 6. 官方评估配置

| 项目 | 值 |
|------|-----|
| 配置文件 | `config/eval/pusht.yaml` |
| num_eval | 50 |
| goal_offset_steps | 25 |
| eval_budget | 50 |
| max_episode_steps | 100 |
| CEM num_samples | 300 |
| CEM n_steps | 30 |
| CEM topk | 30 |
| CEM var_scale | 1.0 |
| PlanConfig horizon | 1 |
| PlanConfig receding_horizon | 1 |
| PlanConfig action_block | 25 |
| PlanConfig action_num_blocks | 5 |
| PlanConfig action_block_size | 5 |

### action_block 关系
```
packed_action_dim = env_action_dim(2) × action_block(25) = 50
action_num_blocks = packed_dim(50) // action_encoder_input_dim(10) = 5
CEM 采样动作维度 = 50
```

---

## 7. NPU 模型转换流程

### 7.1 转换环境
- 机器：x86 Ubuntu (10.128.201.131)
- conda 环境：rknn-py310
- 工具链：rknn-toolkit2 2.3.2

### 7.2 转换步骤
1. **导出 ONNX**：`export_multistep_onnx.py`
   - 用正确参数构建 ARPredictor + PredProj
   - 加载预训练权重
   - 导出多步预测 ONNX（输入 latent+act_emb，输出 pred_latent）
   - ONNX opset_version=12
   - 验证 ONNX vs PyTorch 精度（max diff < 1e-6）

2. **转换 RKNN FP16**：`convert_rknn.py`
   - `rknn.config(target_platform='rk3588')`
   - `rknn.load_onnx(inputs=['latent','act_emb'], input_size_list=[[1,1,192],[1,5,192]])`
   - `rknn.build(do_quantization=False)` → FP16
   - `rknn.export_rknn()` → .rknn 文件

3. **板端测试**：`test_multistep_npu.py`
   - 加载 RKNN 模型，init_runtime(core_mask=NPU_CORE_0_1_2)
   - 精度对比：NPU vs PyTorch（SNR, max diff, mean diff）
   - 性能对比：NPU vs CPU（推理延迟）

### 7.3 模型文件
| 文件 | 大小 | 说明 |
|------|------|------|
| `predictor_multistep.onnx` | 37.60 MB | 多步预测 ONNX（正确参数） |
| `predictor_multistep_fp16.rknn` | 20 MB | 多步预测 RKNN FP16（正确参数） |
| ~~`predictor_S300_pretrained_fp16.rknn`~~ | ~~19MB~~ | ~~旧模型（参数错误，已弃用）~~ |
| ~~`predictor_S300_pretrained_int8.rknn`~~ | ~~11MB~~ | ~~旧模型（参数错误，已弃用）~~ |

### 7.4 板端测试结果（2026-09-18）
| 指标 | PyTorch CPU | NPU FP16 | 加速比 |
|------|-------------|----------|--------|
| 推理延迟 | 31.40 ms | 3.24 ms | **9.7x** |
| 精度 SNR | - | 46.39 dB | 优秀 |
| 最大绝对误差 | - | 0.0095 | 很小 |

---

## 8. 锁频配置（板端性能实验必须）

> 用户明确要求：后续所有实验都锁最高频，避免原生频率调度策略影响实验。

```bash
# 小核 (A55, cpu0-3)
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo 1800000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq  # 1800MHz

# 大核 (A76, cpu4-7)
echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
echo 2352000 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq  # 2352MHz

# NPU
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/max_freq  # 1GHz

# DMC (内存控制器)
echo 2112000000 > /sys/class/devfreq/dmc/max_freq  # 2112MHz
```

---

## 9. SSH 配置（连接复用）

> 用户明确要求：SSH 连接必须复用，不要反复 sshpass（会触发 fail2ban）。

配置文件：`~/.ssh/config_rknn`

```
Host x86-build
    HostName 10.128.201.131
    User wjw
    Port 22
    ControlMaster auto
    ControlPath ~/.ssh/cm/x86-%r@%h:%p
    ControlPersist 3600
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    PreferredAuthentications password
    PubkeyAuthentication no
    ServerAliveInterval 30
    ServerAliveCountMax 3

Host rk3588
    HostName 192.168.77.2
    User root
    Port 22
    ControlMaster auto
    ControlPath ~/.ssh/cm/rk-%r@%h:%p
    ControlPersist 3600
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

### 使用方式
```bash
# 首次建立 master 连接（只需一次）
sshpass -p 'wjww118xxx' ssh -F ~/.ssh/config_rknn -Nf x86-build
sshpass -p '123456' ssh -F ~/.ssh/config_rknn -Nf rk3588

# 后续直接使用（自动复用）
ssh -F ~/.ssh/config_rknn x86-build 'command'
ssh -F ~/.ssh/config_rknn rk3588 'command'
scp -F ~/.ssh/config_rknn file x86-build:/path/

# 检查 master 连接
ls ~/.ssh/cm/
```

---

## 10. Git 仓库

| 项目 | 值 |
|------|-----|
| 仓库 | `git@github.com:OujaKivi/fast-lewm-rknn.git` |
| 分支 | main |
| 本地目录 | `/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/` |
| .gitignore | 排除 *.rknn, *.onnx, *.pt, *.ckpt, data/ |

---

## 11. 关键脚本清单

| 脚本 | 位置 | 用途 |
|------|------|------|
| `rk3588_planner_server.py` | 板端 `/root/Fast-LeWorldModel/` + Mac 项目目录 | 板端完整规划 server（ViT+ActionEnc+Predictor+CEM，支持CPU/NPU） |
| `run_edge_comparison.py` | Mac 项目目录 | Mac 端评估客户端（PushT环境+板端规划，端到端对比） |
| `export_multistep_onnx.py` | x86 `/home/wjw/fast-lewm-npu/` | 多步预测 ONNX 导出脚本 |
| `convert_rknn.py` | x86 `/home/wjw/fast-lewm-npu/` | ONNX → RKNN 转换脚本 |
| `test_multistep_npu.py` | 板端 `/root/Fast-LeWorldModel/` | 板端 NPU 精度和性能测试 |
| `eval_official_v2.py` | Mac 项目目录 | 对齐官方原生流程的评估脚本 |
| `PROGRESS.md` | Mac 项目目录 | 详细进度记录 |
| `env.md` | Mac 项目目录 | 本文档，环境记录 |

---

## 12. 已验证"做不通"的方向

| 方向 | 原因 | 结论 |
|------|------|------|
| 自建 PushT 环境（pymunk+PIL） | 与官方预训练权重对不齐 | 已弃用，使用官方 stable-worldmodel 环境 |
| action_encoder 上 NPU | CPU 97ms vs NPU 130ms，慢 34%，且预训练权重 NPU 转换精度有问题 | 已确认留 CPU |
| 方案 B 双缓冲流水线 | 仅加速 7%，因小 batch NPU 效率低+分组 overhead | 收益有限 |
| SSH offload 架构（Mac跑CEM+环境，RK3588只做predictor） | 通信开销占 70% | 已确认不可行，正确架构是 RK3588 跑完整规划 pipeline |
| 旧 NPU 模型（heads=3, 单步循环） | 参数错误，权重加载不正确，NPU 66.8ms 比 CPU 还慢 | 已弃用，重新转换正确参数的多步预测模型 |

---

## 13. 架构设计

### 正确架构：Mac 环境 + RK3588 完整规划
```
Mac (M5 Pro)                    RK3588 (端侧)
┌─────────────────┐             ┌──────────────────────────────┐
│ PushT 环境模拟   │  stdin/stdout │ 完整规划 pipeline            │
│ (stable-worldmodel)│ ◄──JSON──► │  ViT 编码 (CPU)             │
│                 │             │  Action Encoder (CPU)        │
│ 评估客户端       │             │  Predictor (CPU/NPU)         │
│ (run_edge_      │             │  CEM 采样+更新 (CPU)         │
│  comparison.py) │             │                              │
└─────────────────┘             └──────────────────────────────┘
```

- 通信开销 < 0.1s（验证架构可行）
- RK3588 负责完整规划，避免频繁数据传输
- Mac 负责环境模拟和评估，不占用端侧资源

---

## 14. 常见问题

### Q: 为什么需要 x86 机器编译 RKNN？
A: rknn-toolkit2（模型转换工具链）只支持 x86 环境，不支持 aarch64。板端只能用 rknn-toolkit-lite2 做推理，不能做转换。

### Q: 为什么一定要 ONNX？
A: RKNN 工具链支持 ONNX/TensorFlow/Caffe 等格式，ONNX 是最通用的中间格式，PyTorch 导出 ONNX 最成熟。

### Q: fail2ban 是什么？
A: Linux 下的防暴力破解工具，会监控 SSH 登录失败次数，超过阈值后封禁 IP。反复 sshpass 登录失败会触发封禁。

### Q: 为什么 Mac 成功率低（28%）？
A: 可能 CEM 参数或采样偏差，论文约 80-90%。需进一步排查（可能是 num_samples 或 n_steps 设置，或仍有细微不对齐）。

---

*本文档随项目进展持续更新。*
