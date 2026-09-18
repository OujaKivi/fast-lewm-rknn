"""
Mac CPU 官方完整流程评估（从数据集中采样起始状态和 goal）
"""
import sys
sys.path.insert(0, '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel')

import os
import numpy as np
import torch
import lance
from PIL import Image
import gymnasium as gym
import stable_worldmodel as swm
from stable_worldmodel.wrapper import AddPixelsWrapper
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel import PlanConfig
from sklearn import preprocessing
from torchvision import transforms

# ============ 配置 ============
CKPT_DIR = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel/weights'
CKPT_NAME = 'Fast-lewm_pusht_object.ckpt'
RUN_NAME = 'Fast-lewm_pusht'  # 去掉 _object.ckpt 后缀
LANCE_PATH = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/data/datasets/datasets--galilai-group--lewm-pusht/snapshots/ea321e392348e3c65a18ab0d685f00e57be2c3e0/pusht_expert_train.lance'
GOAL_OFFSET_STEPS = 25
NUM_EPISODES = 5
MAX_STEPS = 100

print('=== 加载数据集 ===')
ds = lance.dataset(LANCE_PATH)
print(f'数据集行数: {ds.count_rows()}')

# 获取所有 episode
episode_idx = ds.to_pandas(columns=['episode_idx'])['episode_idx'].values
unique_episodes = np.unique(episode_idx)
print(f'Episode 数量: {len(unique_episodes)}')

# 拟合 action StandardScaler
print('\n=== 拟合 StandardScaler ===')
action_data = []
batch_size = 100000
for start in range(0, ds.count_rows(), batch_size):
    end = min(start + batch_size, ds.count_rows())
    df = ds.to_pandas(columns=['action'], offset=start, limit=end-start)
    actions = np.array(df['action'].tolist())
    action_data.append(actions)
action_data = np.concatenate(action_data, axis=0)
action_scaler = preprocessing.StandardScaler()
action_scaler.fit(action_data)
print(f'Action mean: {action_scaler.mean_}')
print(f'Action std: {action_scaler.scale_}')

# ============ 图像 transform ============
from torchvision.transforms import v2
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.Resize((224, 224), antialias=True),
])

# ============ 加载模型 ============
print('\n=== 加载模型 ===')
from stable_worldmodel.policy import AutoCostModel
model = AutoCostModel(RUN_NAME, cache_dir=CKPT_DIR)
model = model.eval()
model.requires_grad_(False)
model.interpolate_pos_encoding = True
# 设置 rollout consistency 参数
model.consistency_loss_weight = 0.0
model.action_num_blocks_per_step = [2, 3]

# 修复 checkpoint 中 ViTConfig 不完整的问题
import torch.nn as nn
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

print('模型类型:', type(model).__name__)

# ============ 创建 CEM Solver ============
print('\n=== 创建 CEM Solver ===')
solver = CEMSolver(
    model=model,
    num_samples=300,
    var_scale=1.0,
    n_steps=30,
    topk=30,
    device='cpu',
    seed=42,
)
print('CEM Solver 创建成功')

# ============ 创建 PlanConfig ============
config = PlanConfig(horizon=1, receding_horizon=1, action_block=25)
print(f'PlanConfig: horizon={config.horizon}, action_block={config.action_block}, receding_horizon={config.receding_horizon}')

# ============ 创建 WorldModelPolicy ============
process = {'action': action_scaler}
transform = {'pixels': img_transform, 'goal': img_transform}
policy = WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
print('WorldModelPolicy 创建成功')

# ============ 创建环境 ============
print('\n=== 创建 PushT 环境 ===')
env = gym.make('swm/PushT-v1', max_episode_steps=MAX_STEPS)
env = AddPixelsWrapper(env, pixels_shape=(224, 224))
# 包装 action_space 为 (1, 2)，CEMSolver 用 action_space.shape[1:] 计算 _action_dim
from gymnasium import spaces
env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
env.num_envs = 1
env.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
policy.set_env(env)
print('PushT 环境创建成功')

# ============ 从数据集中采样起始点 ============
print('\n=== 从数据集中采样起始点 ===')
# 随机选择几个 episode
rng = np.random.default_rng(42)
selected_episodes = rng.choice(unique_episodes, size=NUM_EPISODES, replace=False)
print(f'选择的 episodes: {selected_episodes}')

success_count = 0
total_rewards = []

for ep_idx, ep_id in enumerate(selected_episodes):
    print(f'\n=== Episode {ep_idx+1}/{NUM_EPISODES} (episode_id={ep_id}) ===')
    
    # 获取该 episode 的所有行
    ep_mask = episode_idx == ep_id
    ep_rows = np.where(ep_mask)[0]
    ep_step_idx = ds.to_pandas(columns=['step_idx'], offset=ep_rows[0], limit=len(ep_rows))['step_idx'].values
    
    # 选择起始 step（确保有足够的后续步数作为 goal）
    valid_start_steps = ep_step_idx[:-GOAL_OFFSET_STEPS]
    if len(valid_start_steps) == 0:
        print(f'  Episode {ep_id} 太短，跳过')
        continue
    start_step = rng.choice(valid_start_steps)
    start_row = ep_rows[np.where(ep_step_idx == start_step)[0][0]]
    goal_row = ep_rows[np.where(ep_step_idx == start_step + GOAL_OFFSET_STEPS)[0][0]]
    
    # 获取起始状态和 goal 状态（把 velocity 设为 0，避免 agent 快速飞出画面）
    start_state = np.array(ds.to_pandas(columns=['state'], offset=start_row, limit=1)['state'].iloc[0])
    start_state[-2:] = 0.0  # 把 velocity 设为 0
    goal_state = np.array(ds.to_pandas(columns=['state'], offset=goal_row, limit=1)['state'].iloc[0])
    goal_state[-2:] = 0.0  # 把 velocity 设为 0
    print(f'  起始 step: {start_step}, goal step: {start_step + GOAL_OFFSET_STEPS}')
    print(f'  起始 state: {start_state}')
    print(f'  goal state: {goal_state}')
    
    # 重置环境并设置起始状态
    obs, info = env.reset(seed=42)
    unwrapped = env.unwrapped
    unwrapped._set_state(start_state)
    unwrapped._set_goal_state(goal_state)
    
    # 渲染 goal 图像（设置到 goal_state 后渲染）
    unwrapped._set_state(goal_state)
    goal_img = unwrapped.render()
    # 恢复起始状态
    unwrapped._set_state(start_state)
    
    # 获取当前图像
    current_img = info['pixels']
    
    print(f'  当前 agent: {info["pos_agent"]}, block: {info["block_pose"][:2]}')
    print(f'  goal agent: [{goal_state[0]:.1f}, {goal_state[1]:.1f}], block: [{goal_state[2]:.1f}, {goal_state[3]:.1f}]')
    
    # 运行 episode
    episode_reward = 0
    for step in range(MAX_STEPS):
        # 构建 info_dict
        info_dict = {
            'pixels': current_img[None, None],  # (1, 1, H, W, C) uint8
            'goal': goal_img[None, None],
            'action': np.zeros((1, 1, 2), dtype=np.float32),  # dummy action, numpy
        }
        
        # 获取动作
        with torch.no_grad():
            action = policy.get_action(info_dict)
        
        action = np.asarray(action).reshape(-1)[:2]
        action = np.clip(action, -1.0, 1.0)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        current_img = info['pixels']
        episode_reward += reward
        
        if step % 10 == 0:
            print(f'  step {step}: reward={reward:.1f}, action=[{action[0]:.3f},{action[1]:.3f}], agent={info["pos_agent"]}, block={info["block_pose"][:2]}')
        
        if terminated or truncated:
            success = info.get('success', False)
            print(f'  Episode 结束: success={success}, terminated={terminated}, truncated={truncated}, steps={step+1}, final_reward={reward:.1f}')
            print(f'  info keys: {list(info.keys())}')
            if 'distance' in info:
                print(f'  distance: {info["distance"]}')
            if success:
                success_count += 1
            break
    
    total_rewards.append(episode_reward)
    if not (terminated or truncated):
        print(f'  Episode 超时: steps={MAX_STEPS}, final_reward={reward:.1f}')

print(f'\n=== 完成 ===')
print(f'成功率: {success_count}/{NUM_EPISODES} = {success_count/NUM_EPISODES*100:.1f}%')
print(f'平均 episode reward: {np.mean(total_rewards):.1f}')

env.close()
