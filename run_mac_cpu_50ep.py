"""
实验1：Mac CPU 50 Episode 稳定成功率
对齐官方原生流程，运行 50 个 episode 统计稳定成功率
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
NUM_EPISODES = 50
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
    model.interpolate_pos_encoding = True
    
    # 修复 ViTConfig
    cfg = model.encoder.config
    if not hasattr(cfg, 'output_attentions'):
        cfg.output_attentions = getattr(cfg, '_output_attentions', False)
    if not hasattr(cfg, 'torchscript'):
        cfg.torchscript = False
    if not hasattr(cfg, 'use_return_dict'):
        cfg.use_return_dict = getattr(cfg, 'return_dict', True)
    
    # 修复 ViTSelfAttention dropout
    for name, module in model.encoder.named_modules():
        if type(module).__name__ == 'ViTSelfAttention':
            if not hasattr(module, 'dropout'):
                module.dropout = nn.Dropout(p=getattr(module, 'dropout_prob', 0.0))
    
    # GRU train mode
    for m in model.modules():
        if isinstance(m, torch.nn.GRU):
            m.train()
    
    return model

def fit_process(dataset):
    """拟合 process（action+proprio+state）"""
    process = {}
    print("拟合 StandardScaler...")
    df = dataset.to_pandas(columns=['action', 'proprio', 'state'])
    
    for col in ['action', 'proprio', 'state']:
        data = np.stack(df[col].values)
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        process[col] = scaler
        print(f"  {col}: mean={scaler.mean_[:4]}, std={scaler.scale_[:4]}")
        
        if col != 'action':
            process[f'goal_{col}'] = scaler
    
    return process

def sample_episodes(dataset, num_episodes, goal_offset, seed=42):
    """从数据集中采样评估 episode"""
    df = dataset.to_pandas(columns=['episode_idx', 'step_idx'])
    episode_lengths = df.groupby('episode_idx')['step_idx'].max() + 1
    valid_episodes = episode_lengths[episode_lengths > goal_offset + 1].index
    
    rng = np.random.default_rng(seed)
    selected_episodes = rng.choice(valid_episodes, size=num_episodes, replace=False)
    
    episodes = []
    for ep_id in selected_episodes:
        ep_data = df[df['episode_idx'] == ep_id]
        max_start = len(ep_data) - goal_offset - 1
        start_step = rng.integers(0, max_start)
        episodes.append((int(ep_id), int(start_step)))
    
    return episodes

def get_episode_data(dataset, episode_id, start_step, goal_offset):
    """获取单个 episode 的起始状态和 goal 状态"""
    idx_df = dataset.to_pandas(columns=['episode_idx', 'step_idx'])
    mask = (idx_df['episode_idx'] == episode_id) & (idx_df['step_idx'] >= start_step) & (idx_df['step_idx'] <= start_step + goal_offset)
    row_indices = np.where(mask)[0]
    
    if len(row_indices) == 0:
        return None
    
    min_row = row_indices[0]
    max_row = row_indices[-1]
    num_rows = max_row - min_row + 1
    
    df = dataset.to_pandas(
        columns=['action', 'proprio', 'state', 'pixels', 'episode_idx', 'step_idx'],
        offset=min_row,
        limit=num_rows
    )
    
    mask = (df['episode_idx'] == episode_id) & (df['step_idx'] >= start_step) & (df['step_idx'] <= start_step + goal_offset)
    ep_data = df[mask].reset_index(drop=True)
    
    if len(ep_data) == 0:
        return None
    
    init_state = {
        'action': np.array(ep_data['action'].iloc[0]),
        'proprio': np.array(ep_data['proprio'].iloc[0]),
        'state': np.array(ep_data['state'].iloc[0]),
        'pixels': ep_data['pixels'].iloc[0],
    }
    
    goal_state = {
        'goal_action': np.array(ep_data['action'].iloc[-1]),
        'goal_proprio': np.array(ep_data['proprio'].iloc[-1]),
        'goal_state': np.array(ep_data['state'].iloc[-1]),
        'goal': ep_data['pixels'].iloc[-1],
    }
    
    return init_state, goal_state

def decode_pixels(pixels_bytes):
    """解码 pixels"""
    import cv2
    img_array = np.frombuffer(pixels_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def run_episode(env, policy, init_state, goal_state, max_steps, episode_idx):
    """运行单个 episode"""
    obs, info = env.reset(seed=42)
    unwrapped = env.unwrapped
    
    unwrapped._set_state(state=init_state['state'])
    unwrapped._set_goal_state(goal_state=goal_state['goal_state'])
    
    goal_img = decode_pixels(goal_state['goal'])
    
    total_reward = 0
    success = False
    cem_times = []
    
    for step in range(max_steps):
        info['goal'] = goal_img
        
        info_dict = {}
        for k in ['pixels', 'goal', 'state', 'proprio']:
            if k not in info:
                continue
            v = info[k]
            if k in ['pixels', 'goal']:
                info_dict[k] = v[np.newaxis, np.newaxis, ...]
            else:
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    info_dict[k] = v[np.newaxis, ...]
                else:
                    info_dict[k] = v
        
        info_dict['action'] = np.zeros((1, 1, 2), dtype=np.float32)
        
        # 计时 CEM solve
        t0 = time.time()
        action = policy.get_action(info_dict)
        cem_time = time.time() - t0
        cem_times.append(cem_time)
        
        action = action[0]
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            success = True
            break
        
        if truncated:
            break
    
    return {
        'episode_idx': episode_idx,
        'success': success,
        'steps': step + 1,
        'total_reward': float(total_reward),
        'mean_cem_time': float(np.mean(cem_times)),
        'max_cem_time': float(np.max(cem_times)),
        'min_cem_time': float(np.min(cem_times)),
        'cem_times': [float(t) for t in cem_times],
        'start_state': init_state['state'].tolist(),
        'goal_state': goal_state['goal_state'].tolist(),
    }

def main():
    print("=" * 60)
    print("实验1：Mac CPU 50 Episode 稳定成功率")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n1. 加载数据集...")
    dataset = lance.dataset(DATASET_PATH)
    print(f"  数据集行数: {dataset.count_rows()}")
    
    # 2. 拟合 process
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
    
    config = PlanConfig(horizon=1, receding_horizon=1, action_block=25)
    
    transform = {'pixels': img_transform(), 'goal': img_transform()}
    
    policy = WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
    policy.set_env(env)
    print("  Policy 创建成功")
    
    # 6. 采样评估 episode
    print(f"\n6. 采样 {NUM_EPISODES} 个评估 episode...")
    episodes = sample_episodes(dataset, NUM_EPISODES, GOAL_OFFSET)
    print(f"  已采样 {len(episodes)} 个 episode")
    
    # 7. 运行评估
    print(f"\n7. 运行 {NUM_EPISODES} 个 episode...")
    results = []
    success_count = 0
    t_total_start = time.time()
    
    for i, (ep_id, start_step) in enumerate(episodes):
        t_ep_start = time.time()
        
        result = get_episode_data(dataset, ep_id, start_step, GOAL_OFFSET)
        if result is None:
            print(f"  Episode {i+1}/{NUM_EPISODES}: 无法获取数据，跳过")
            continue
        
        init_state, goal_state = result
        ep_result = run_episode(env, policy, init_state, goal_state, MAX_STEPS, i)
        
        ep_time = time.time() - t_ep_start
        results.append(ep_result)
        
        if ep_result['success']:
            success_count += 1
        
        print(f"  Episode {i+1}/{NUM_EPISODES}: success={ep_result['success']}, "
              f"steps={ep_result['steps']}, reward={ep_result['total_reward']:.1f}, "
              f"cem_time={ep_result['mean_cem_time']:.3f}s, ep_time={ep_time:.1f}s")
    
    total_time = time.time() - t_total_start
    
    # 8. 统计结果
    print("\n" + "=" * 60)
    print("实验结果统计")
    print("=" * 60)
    
    success_rate = success_count / len(results) * 100
    all_cem_times = [t for r in results for t in r['cem_times']]
    success_steps = [r['steps'] for r in results if r['success']]
    success_rewards = [r['total_reward'] for r in results if r['success']]
    all_rewards = [r['total_reward'] for r in results]
    
    stats = {
        'num_episodes': len(results),
        'success_count': success_count,
        'success_rate': float(success_rate),
        'mean_steps': float(np.mean([r['steps'] for r in results])),
        'mean_success_steps': float(np.mean(success_steps)) if success_steps else 0,
        'mean_reward': float(np.mean(all_rewards)),
        'mean_success_reward': float(np.mean(success_rewards)) if success_rewards else 0,
        'mean_cem_time': float(np.mean(all_cem_times)),
        'median_cem_time': float(np.median(all_cem_times)),
        'p95_cem_time': float(np.percentile(all_cem_times, 95)),
        'p99_cem_time': float(np.percentile(all_cem_times, 99)),
        'max_cem_time': float(np.max(all_cem_times)),
        'min_cem_time': float(np.min(all_cem_times)),
        'total_time': float(total_time),
        'config': {
            'platform': 'Mac M5 Pro CPU',
            'model': 'Fast-lewm_pusht pretrained',
            'precision': 'FP32',
            'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
            'goal_offset_steps': GOAL_OFFSET,
            'max_steps': MAX_STEPS,
        }
    }
    
    print(f"成功率: {success_count}/{len(results)} = {success_rate:.1f}%")
    print(f"平均步数: {stats['mean_steps']:.1f}")
    print(f"成功 episode 平均步数: {stats['mean_success_steps']:.1f}")
    print(f"平均 reward: {stats['mean_reward']:.1f}")
    print(f"成功 episode 平均 reward: {stats['mean_success_reward']:.1f}")
    print(f"CEM solve time: mean={stats['mean_cem_time']:.3f}s, "
          f"median={stats['median_cem_time']:.3f}s, "
          f"P95={stats['p95_cem_time']:.3f}s, "
          f"P99={stats['p99_cem_time']:.3f}s")
    print(f"总运行时间: {total_time:.1f}s")
    
    # 9. 保存结果
    output = {
        'stats': stats,
        'episodes': results,
    }
    
    output_path = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/mac_cpu_50ep_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存到 {output_path}")
    
    env.close()

if __name__ == '__main__':
    main()
