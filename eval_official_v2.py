"""
官方原生流程对齐版评估脚本
关键对齐点：
1. goal 图像直接从数据集取（不是手动渲染）
2. process 拟合 action+proprio+state 三列
3. 每步更新 infos 中的 goal 图像
4. 保留数据集中的 velocity
5. action_num_blocks_per_step=[2,3]
"""
import sys
sys.path.insert(0, '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel')

import os
import json
import time
import numpy as np
import torch
import lance
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import gymnasium as gym
from gymnasium import spaces
import stable_worldmodel as swm
from stable_worldmodel.wrapper import AddPixelsWrapper
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel import PlanConfig

# 配置
CKPT_DIR = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel/weights'
RUN_NAME = 'Fast-lewm_pusht'
DATASET_PATH = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/data/datasets/datasets--galilai-group--lewm-pusht/snapshots/ea321e392348e3c65a18ab0d685f00e57be2c3e0/pusht_expert_train.lance'
NUM_EPISODES = 5
GOAL_OFFSET = 25
EVAL_BUDGET = 50
MAX_STEPS = 2 * EVAL_BUDGET  # 100

# ImageNet 归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.Resize(size=(224, 224)),
    ])

def load_model():
    """加载预训练模型（对齐官方 AutoCostModel）"""
    import torch.nn as nn
    
    model = swm.policy.AutoCostModel(RUN_NAME, cache_dir=CKPT_DIR)
    model = model.to('cpu')
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True  # 官方 eval.py 第314行
    
    # 修复 checkpoint 中 ViTConfig 不完整的问题
    cfg = model.encoder.config
    if not hasattr(cfg, 'output_attentions'):
        cfg.output_attentions = getattr(cfg, '_output_attentions', False)
    if not hasattr(cfg, 'torchscript'):
        cfg.torchscript = False
    if not hasattr(cfg, 'use_return_dict'):
        cfg.use_return_dict = getattr(cfg, 'return_dict', True)
    
    # 修复 ViTSelfAttention 缺少 dropout 模块
    for name, module in model.encoder.named_modules():
        if type(module).__name__ == 'ViTSelfAttention':
            if not hasattr(module, 'dropout'):
                module.dropout = nn.Dropout(p=getattr(module, 'dropout_prob', 0.0))
    
    # GRU 设为 train mode（对齐官方）
    for m in model.modules():
        if isinstance(m, torch.nn.GRU):
            m.train()
    
    return model

def fit_process(dataset):
    """拟合 process（对齐官方：action+proprio+state 三列）"""
    process = {}
    
    # 读取所有数据用于拟合
    print("拟合 StandardScaler...")
    df = dataset.to_pandas(columns=['action', 'proprio', 'state'])
    
    for col in ['action', 'proprio', 'state']:
        data = np.stack(df[col].values)
        # 去除 NaN
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        process[col] = scaler
        print(f"  {col}: mean={scaler.mean_[:4]}, std={scaler.scale_[:4]}")
        
        # goal_* 也用同一个 scaler
        if col != 'action':
            process[f'goal_{col}'] = scaler
    
    return process

def sample_episodes(dataset, num_episodes, goal_offset):
    """从数据集中采样评估 episode（对齐官方）"""
    df = dataset.to_pandas(columns=['episode_idx', 'step_idx'])
    
    # 计算每个 episode 的长度
    episode_lengths = df.groupby('episode_idx')['step_idx'].max() + 1
    
    # 筛选有足够长度的 episode
    valid_episodes = episode_lengths[episode_lengths > goal_offset + 1].index
    
    # 随机采样
    rng = np.random.default_rng(42)
    selected_episodes = rng.choice(valid_episodes, size=num_episodes, replace=False)
    
    episodes = []
    for ep_id in selected_episodes:
        ep_data = df[df['episode_idx'] == ep_id]
        max_start = len(ep_data) - goal_offset - 1
        start_step = rng.integers(0, max_start)
        episodes.append((int(ep_id), int(start_step)))
    
    return episodes

def get_episode_data(dataset, episode_id, start_step, goal_offset):
    """获取单个 episode 的起始状态和 goal 状态（对齐官方 _extract_init_goal）"""
    # 先获取 episode_idx 和 step_idx 列，找到需要的行索引
    idx_df = dataset.to_pandas(columns=['episode_idx', 'step_idx'])
    mask = (idx_df['episode_idx'] == episode_id) & (idx_df['step_idx'] >= start_step) & (idx_df['step_idx'] <= start_step + goal_offset)
    row_indices = np.where(mask)[0]
    
    if len(row_indices) == 0:
        return None
    
    # 只读取需要的行
    # lance 不支持按行索引随机读取，我们需要用 offset/limit
    # 但行可能不连续，所以读取 min~max 范围
    min_row = row_indices[0]
    max_row = row_indices[-1]
    num_rows = max_row - min_row + 1
    
    df = dataset.to_pandas(
        columns=['action', 'proprio', 'state', 'pixels', 'episode_idx', 'step_idx'],
        offset=min_row,
        limit=num_rows
    )
    
    # 再次筛选
    mask = (df['episode_idx'] == episode_id) & (df['step_idx'] >= start_step) & (df['step_idx'] <= start_step + goal_offset)
    ep_data = df[mask].reset_index(drop=True)
    
    if len(ep_data) == 0:
        return None
    
    # 起始状态 = 第一帧
    init_state = {
        'action': np.array(ep_data['action'].iloc[0]),
        'proprio': np.array(ep_data['proprio'].iloc[0]),
        'state': np.array(ep_data['state'].iloc[0]),
        'pixels': ep_data['pixels'].iloc[0],  # bytes
    }
    
    # goal 状态 = 最后一帧
    goal_state = {
        'goal_action': np.array(ep_data['action'].iloc[-1]),
        'goal_proprio': np.array(ep_data['proprio'].iloc[-1]),
        'goal_state': np.array(ep_data['state'].iloc[-1]),
        'goal': ep_data['pixels'].iloc[-1],  # goal 图像直接从数据集取！
    }
    
    return init_state, goal_state

def decode_pixels(pixels_bytes):
    """解码 pixels（PNG bytes -> numpy array）"""
    import cv2
    img_array = np.frombuffer(pixels_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def run_episode(env, model, policy, init_state, goal_state, max_steps):
    """运行单个 episode（对齐官方 _evaluate_from_dataset）"""
    obs, info = env.reset(seed=42)
    unwrapped = env.unwrapped
    
    # 设置起始状态和 goal 状态（对齐官方 callables）
    unwrapped._set_state(state=init_state['state'])
    unwrapped._set_goal_state(goal_state=goal_state['goal_state'])
    
    # 解码 goal 图像（直接从数据集取）
    goal_img = decode_pixels(goal_state['goal'])
    
    total_reward = 0
    success = False
    
    for step in range(max_steps):
        # 关键：每步都更新 infos 中的 goal 图像（对齐官方 on_step）
        info['goal'] = goal_img
        
        # 把 info 整理成 vectorized 格式（num_envs=1）
        # 只传入模型需要的键，过滤掉标量值（n_contacts, render_time, env_name 等）
        info_dict = {}
        for k in ['pixels', 'goal', 'state', 'proprio']:
            if k not in info:
                continue
            v = info[k]
            if k in ['pixels', 'goal']:
                # 图像: (H, W, C) -> (1, 1, H, W, C)
                info_dict[k] = v[np.newaxis, np.newaxis, ...]
            else:
                # 向量: (n,) -> (1, n)
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    info_dict[k] = v[np.newaxis, ...]
                else:
                    info_dict[k] = v
        
        # 添加 dummy action（get_cost 期望有 action 键）
        info_dict['action'] = np.zeros((1, 1, 2), dtype=np.float32)
        
        # 获取动作（policy 内部会做 _prepare_info、CEM 规划、inverse_transform）
        action = policy.get_action(info_dict)
        
        # action 形状是 (1, 2)，取第一个环境的动作
        action = action[0]
        
        # clip 动作到 [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"  step {step}: reward={reward:.1f}, action={action}, agent={info['pos_agent']}, block={info['block_pose'][:2]}")
        
        if terminated:
            success = True
            print(f"  成功！step {step}")
            break
        
        if truncated:
            break
    
    return success, total_reward, step + 1

def main():
    print("=" * 60)
    print("Fast-LeWM 官方原生流程对齐版评估")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n1. 加载数据集...")
    dataset = lance.dataset(DATASET_PATH)
    print(f"  数据集行数: {dataset.count_rows()}")
    
    # 2. 拟合 process（action+proprio+state）
    print("\n2. 拟合 process...")
    process = fit_process(dataset)
    
    # 3. 加载模型
    print("\n3. 加载模型...")
    model = load_model()
    print("  模型加载成功")
    
    # 4. 创建环境
    print("\n4. 创建环境...")
    env = gym.make('swm/PushT-v1', max_episode_steps=MAX_STEPS)
    env = AddPixelsWrapper(env, pixels_shape=(224, 224))
    # 包装 action_space（CEMSolver 需要 vectorized 格式）
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
    env.num_envs = 1
    env.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    print("  环境创建成功")
    
    # 5. 创建 solver 和 policy
    print("\n5. 创建 solver 和 policy...")
    solver = CEMSolver(
        model=model,
        num_samples=300,
        n_steps=30,
        topk=30,
        var_scale=1.0,
        device='cpu',
        seed=42,
    )
    
    config = PlanConfig(
        horizon=1,
        receding_horizon=1,
        action_block=25,  # action_num_blocks(5) * action_block_size(5)
    )
    
    transform = {
        'pixels': img_transform(),
        'goal': img_transform(),
    }
    
    policy = WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
    policy.set_env(env)
    print("  Policy 创建成功")
    
    # 6. 采样评估 episode
    print("\n6. 采样评估 episode...")
    episodes = sample_episodes(dataset, NUM_EPISODES, GOAL_OFFSET)
    for i, (ep_id, start_step) in enumerate(episodes):
        print(f"  Episode {i+1}: episode_id={ep_id}, start_step={start_step}, goal_step={start_step + GOAL_OFFSET}")
    
    # 7. 运行评估
    print("\n7. 运行评估...")
    success_count = 0
    total_rewards = []
    
    for i, (ep_id, start_step) in enumerate(episodes):
        print(f"\n=== Episode {i+1}/{NUM_EPISODES} (episode_id={ep_id}) ===")
        
        # 获取 episode 数据
        result = get_episode_data(dataset, ep_id, start_step, GOAL_OFFSET)
        if result is None:
            print("  无法获取 episode 数据，跳过")
            continue
        
        init_state, goal_state = result
        print(f"  起始 state: {init_state['state']}")
        print(f"  goal state: {goal_state['goal_state']}")
        print(f"  goal 图像 shape: {decode_pixels(goal_state['goal']).shape}")
        
        # 运行 episode
        success, reward, steps = run_episode(env, model, policy, init_state, goal_state, MAX_STEPS)
        
        print(f"  Episode 结束: success={success}, steps={steps}, total_reward={reward:.1f}")
        
        if success:
            success_count += 1
        total_rewards.append(reward)
    
    # 8. 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"成功率: {success_count}/{NUM_EPISODES} = {success_count/NUM_EPISODES*100:.1f}%")
    print(f"平均 episode reward: {np.mean(total_rewards):.1f}")
    
    # 保存结果
    results = {
        'success_rate': success_count / NUM_EPISODES * 100,
        'num_episodes': NUM_EPISODES,
        'success_count': success_count,
        'mean_reward': float(np.mean(total_rewards)),
        'rewards': total_rewards,
    }
    
    with open('/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/eval_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到 eval_results_v2.json")
    
    env.close()

if __name__ == '__main__':
    main()
