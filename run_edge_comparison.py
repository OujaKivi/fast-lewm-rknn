#!/usr/bin/env python3
"""
Mac 端评估客户端：PushT 环境 + 板端规划
对比板端纯 CPU vs 异构 NPU 的端到端性能
"""
import sys
import os
import json
import time
import base64
import subprocess
import numpy as np
import torch
import lance
from PIL import Image
import io

sys.path.insert(0, 'Fast-LeWorldModel')
import stable_worldmodel as swm
from stable_worldmodel.envs.pusht.env import PushT

# 配置
BOARD_IP = "192.168.77.2"
BOARD_USER = "root"
BOARD_PASS = "123456"
BOARD_SCRIPT = "/root/Fast-LeWorldModel/rk3588_planner_server.py"
BOARD_PYTHON = "/root/miniconda3/envs/fast-lewm/bin/python"

DATASET_PATH = "/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/data/datasets/datasets--galilai-group--lewm-pusht/snapshots/ea321e392348e3c65a18ab0d685f00e57be2c3e0/pusht_expert_train.lance"

NUM_EPISODES = 5
MAX_STEPS = 100
ACTION_BLOCK = 25
ACTION_DIM = 2
GOAL_OFFSET = 25

# StandardScaler 参数（从数据集拟合）
ACTION_MEAN = np.array([-0.0078, 0.0069])
ACTION_STD = np.array([0.2085, 0.2067])


def decode_image(img_data):
    """将图像数据（numpy array 或 JPEG bytes）转为 numpy array"""
    if isinstance(img_data, bytes):
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        return np.array(img)
    elif isinstance(img_data, np.ndarray):
        return img_data
    else:
        return np.array(img_data)

def image_to_base64(img_array):
    """将 numpy 图像数组转为 base64"""
    if isinstance(img_array, bytes):
        # 已经是编码后的 bytes，直接转 base64
        return base64.b64encode(img_array).decode('utf-8')
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


class BoardPlanner:
    """板端规划客户端，通过 SSH 通信"""
    def __init__(self, mode='cpu', cem_steps=30, num_samples=300):
        self.mode = mode
        self.cem_steps = cem_steps
        self.num_samples = num_samples
        self.process = None
        self._start_server()

    def _start_server(self):
        """启动板端 server 进程"""
        cmd = [
            'ssh', '-F', os.path.expanduser('~/.ssh/config_rknn'), 'rk3588',
            f'taskset -c 4-7 {BOARD_PYTHON} {BOARD_SCRIPT} --mode {self.mode} --cem-steps {self.cem_steps} --num-samples {self.num_samples}'
        ]
        print(f"[BoardPlanner] Starting {self.mode} server on board...")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # 等待 server 就绪
        time.sleep(15)
        print(f"[BoardPlanner] {self.mode} server started")

    def plan(self, current_image, goal_image):
        """调用板端规划"""
        request = {
            'current_image': image_to_base64(current_image),
            'goal_image': image_to_base64(goal_image),
        }
        request_json = json.dumps(request) + '\n'

        t0 = time.time()
        self.process.stdin.write(request_json)
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        t_total = time.time() - t0

        if not response_line:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"Server died. stderr: {stderr[-500:]}")

        response = json.loads(response_line)
        if 'error' in response:
            raise RuntimeError(f"Server error: {response['error']}\n{response.get('traceback', '')}")

        response['time_roundtrip'] = t_total
        return response

    def close(self):
        """关闭 server"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            print(f"[BoardPlanner] {self.mode} server closed")


def run_episode(env, planner, episode_idx, dataset):
    """运行单个 episode"""
    # 从数据集采样 episode
    ep_start = episode_idx * 125
    ep_data = dataset.take([ep_start, ep_start + 124])

    # 重置环境（初始化物理引擎）
    obs, info = env.reset()

    # 设置初始状态
    # PushT state: [agent_x, agent_y, block_x, block_y, block_angle, vel_x, vel_y]
    init_state = np.array(ep_data['state'][0].as_py())
    env._set_state(init_state)

    # 设置 goal
    goal_idx = min(GOAL_OFFSET, len(ep_data) - 1)
    goal_state = np.array(ep_data['state'][goal_idx].as_py())

    # 获取 goal 图像（直接从数据集取 pixels，JPEG bytes）
    goal_image = decode_image(ep_data['pixels'][goal_idx].as_py())

    action_buffer = []
    total_reward = 0
    plan_times = []
    success = False

    for step in range(MAX_STEPS):
        # 获取当前图像
        current_image = env.render()

        # 如果 buffer 空了，调用规划器
        if len(action_buffer) == 0:
            result = planner.plan(current_image, goal_image)
            packed_action = np.array(result['action'])  # (50,)
            # reshape 成 (25, 2)
            actions = packed_action.reshape(ACTION_BLOCK, ACTION_DIM)
            # 反标准化
            actions = actions * ACTION_STD + ACTION_MEAN
            action_buffer = list(actions)
            plan_times.append({
                'step': step,
                'time_board': result['time_total'],
                'time_roundtrip': result['time_roundtrip'],
                'time_encode': result.get('time_encode', 0),
                'time_cem': result.get('time_cem', 0),
                'cost': result.get('cost', 0),
            })
            print(f"  Step {step}: plan time={result['time_total']:.1f}s, cost={result.get('cost', 0):.2f}")

        # 执行动作
        action = action_buffer.pop(0)
        step_result = env.step(action)
        if len(step_result) == 5:  # gymnasium
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # old gym
            obs, reward, done, info = step_result
        total_reward += reward

        if done:
            success = info.get('success', False)
            break

    return {
        'episode': episode_idx,
        'success': success,
        'total_reward': total_reward,
        'steps': step + 1,
        'plan_times': plan_times,
        'num_plans': len(plan_times),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cpu', choices=['cpu', 'npu'])
    parser.add_argument('--num-episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--num-samples', type=int, default=300)
    args = parser.parse_args()

    print(f"=== Edge Comparison: {args.mode} mode ===")
    print(f"Episodes: {args.num_episodes}")

    # 加载数据集
    print("Loading dataset...")
    dataset = lance.dataset(DATASET_PATH)

    # 创建环境
    env = PushT()

    # 创建板端规划器
    planner = BoardPlanner(mode=args.mode, cem_steps=args.cem_steps, num_samples=args.num_samples)

    results = []
    try:
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
            t0 = time.time()
            result = run_episode(env, planner, ep, dataset)
            t_ep = time.time() - t0
            result['wall_time'] = t_ep
            results.append(result)
            print(f"  Success: {result['success']}, Reward: {result['total_reward']:.1f}, Steps: {result['steps']}, Time: {t_ep:.1f}s")
    finally:
        planner.close()

    # 统计结果
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    all_plan_times = [t['time_board'] for r in results for t in r['plan_times']]
    avg_plan_time = np.mean(all_plan_times) if all_plan_times else 0

    summary = {
        'mode': args.mode,
        'num_episodes': len(results),
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_plan_time_board': avg_plan_time,
        'results': results,
    }

    output_file = f'edge_comparison_{args.mode}_results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Summary ({args.mode}) ===")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Avg reward: {avg_reward:.1f}")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg plan time (board): {avg_plan_time:.1f}s")
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
