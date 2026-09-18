#!/usr/bin/env python3
"""
Mac 端 PushT 官方完整流程评估（CPU 推理）
- 用官方 stable-worldmodel 环境
- 用官方 AutoCostModel + CEMSolver + WorldModelPolicy
- 验证环境和规划是否正常工作
"""
import sys
sys.path.insert(0, '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel')

import os
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.wrapper import AddPixelsWrapper
import gymnasium as gym

# 配置
CKPT_DIR = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel/weights'
RUN_NAME = 'Fast-lewm_pusht'
IMG_SIZE = 224
NUM_SAMPLES = 300
N_STEPS = 30
TOPK = 30
HORIZON = 1
ACTION_NUM_BLOCKS = 5  # 配置文件中的值
ACTION_BLOCK_SIZE = 5
RECEDING_HORIZON = 1
MAX_EPISODE_STEPS = 100


def build_img_transform():
    """构建图像预处理 transform（与官方 eval.py 一致）"""
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    ])
    return transform


def main():
    print("=== 加载模型 ===")
    model = swm.policy.AutoCostModel(RUN_NAME, cache_dir=CKPT_DIR)
    model = model.to('cpu')
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True  # 官方 eval.py 第314行，ViT 位置编码插值
    
    # 修复 checkpoint 中 ViTConfig 不完整的问题
    # checkpoint 保存的 config 缺少多个公共属性，手动补全
    cfg = model.encoder.config
    if not hasattr(cfg, 'output_attentions'):
        cfg.output_attentions = getattr(cfg, '_output_attentions', False)
    if not hasattr(cfg, 'torchscript'):
        cfg.torchscript = False
    if not hasattr(cfg, 'use_return_dict'):
        cfg.use_return_dict = getattr(cfg, 'return_dict', True)
    
    # 修复 checkpoint 中 ViTSelfAttention 缺少 dropout 模块的问题
    # checkpoint 是用修改版 transformers 训练的，ViTSelfAttention 没有 dropout 层
    for name, module in model.encoder.named_modules():
        if type(module).__name__ == 'ViTSelfAttention':
            if not hasattr(module, 'dropout'):
                module.dropout = nn.Dropout(p=getattr(module, 'dropout_prob', 0.0))
    
    print(f"模型类型: {type(model).__name__}")
    
    # 构建图像 transform
    transform = {
        "pixels": build_img_transform(),
        "goal": build_img_transform(),
    }
    
    # 构建动作标准化（PushT 动作空间 [-1,1]，用 mean=0, std=0.5 做标准化）
    from sklearn.preprocessing import StandardScaler
    action_scaler = StandardScaler()
    action_scaler.mean_ = np.array([0.0, 0.0])
    action_scaler.scale_ = np.array([0.5, 0.5])
    action_scaler.var_ = np.array([0.25, 0.25])
    action_scaler.n_features_in_ = 2
    action_scaler.n_samples_seen_ = 1000
    # 不在 policy 内部做 inverse_transform（因为 env 不是 vectorized），而是在 get_action 后手动处理
    process = {}
    
    print("\n=== 创建 CEM Solver ===")
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=NUM_SAMPLES,
        n_steps=N_STEPS,
        topk=TOPK,
        device='cpu',
        seed=42,
    )
    print(f"CEM Solver 创建成功: num_samples={NUM_SAMPLES}, n_steps={N_STEPS}, topk={TOPK}")
    
    print("\n=== 创建 PlanConfig ===")
    action_block = ACTION_NUM_BLOCKS * ACTION_BLOCK_SIZE  # 25
    config = swm.PlanConfig(
        horizon=HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=action_block,
    )
    print(f"PlanConfig: horizon={HORIZON}, action_block={action_block}, receding_horizon={RECEDING_HORIZON}")
    
    print("\n=== 创建 WorldModelPolicy ===")
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
    print("WorldModelPolicy 创建成功")
    
    print("\n=== 创建 PushT 环境 ===")
    env = gym.make('swm/PushT-v1', max_episode_steps=MAX_EPISODE_STEPS)
    env = AddPixelsWrapper(env, pixels_shape=(IMG_SIZE, IMG_SIZE))
    # 给 env 添加 num_envs 和 single_action_space 属性（WorldModelPolicy 需要）
    env.num_envs = 1
    env.single_action_space = env.action_space  # shape (2,)
    # CEMSolver 期望 env.action_space.shape 是 (n_envs, action_dim)，所以需要包装成 (1, 2)
    from gymnasium import spaces
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
    policy.set_env(env)
    print("PushT 环境创建成功")
    
    # 运行一个 episode
    print("\n=== 运行 Episode ===")
    obs, info = env.reset(seed=42)
    print(f"初始: agent_pos={info['pos_agent']}, block_pos={info['block_pose'][:2]}, goal_pos={info['goal_pose'][:2]}")
    
    success = False
    for step in range(MAX_EPISODE_STEPS):
        # 用官方 policy 获取动作
        # policy.get_action 需要 info_dict，包含 pixels 和 goal
        # _prepare_info 期望 [e, t, H, W, C] 格式（5维）
        # get_cost 还需要 action 键（即使是 dummy）
        info_dict = {
            "pixels": info["pixels"][np.newaxis, np.newaxis, ...],  # [1, 1, 224, 224, 3]
            "goal": info["goal"][np.newaxis, np.newaxis, ...],      # [1, 1, 224, 224, 3]
            "action": np.zeros((1, 1, 2), dtype=np.float32),         # dummy action
        }
        
        try:
            action = policy.get_action(info_dict)
            action = np.asarray(action).reshape(-1)[:2]  # 取前 2 维
            # 用从 pusht_expert_train 数据集拟合的 StandardScaler 做 inverse_transform
            # mean=[-0.0078, 0.0069], std=[0.2085, 0.2067]
            action = action * np.array([0.2085, 0.2067]) + np.array([-0.0078, 0.0069])
            # 动作空间是 [-1, 1]，clip 防止飞出画面
            action = np.clip(action, -1.0, 1.0)
        except Exception as e:
            print(f"  step {step}: policy.get_action 失败: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        
        if step % 5 == 0:
            print(f"  step {step}: reward={reward:.1f}, action=[{action[0]:.3f},{action[1]:.3f}], "
                  f"agent_pos={info['pos_agent']}, block_pos={info['block_pose'][:2]}")
        
        if terminated or truncated:
            success = info.get("success", reward > 0)
            print(f"\nEpisode 结束: success={success}, steps={step+1}, final_reward={reward:.1f}")
            break
    
    if not success and step == MAX_EPISODE_STEPS - 1:
        print(f"\nEpisode 超时: steps={MAX_EPISODE_STEPS}, final_reward={reward:.1f}")
    
    env.close()
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
