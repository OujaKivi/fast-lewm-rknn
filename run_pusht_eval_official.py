#!/usr/bin/env python3
"""
Mac 端 PushT 评估脚本（官方 stable-worldmodel 环境）
- 本地运行官方 swm/PushT-v1 环境
- 通过 SSH 调用 RK3588 上的 planner server 进行规划
- 统计成功率、完成时间、规划延迟等指标

用法:
    python run_pusht_eval_official.py --episodes 10 --cem_iters 30 --num_samples 300
"""
import argparse
import json
import subprocess
import sys
import time
import base64
import io
import threading
import numpy as np
from PIL import Image

import gymnasium as gym
import stable_worldmodel.envs  # 注册环境
from stable_worldmodel.wrapper import AddPixelsWrapper


# SSH 配置（复用 ControlMaster）
SSH_CONFIG = "/Users/wangjiwei/.ssh/config_rknn"
RK3588_HOST = "rk3588"
RK3588_WORKDIR = "/root/Fast-LeWorldModel"
RK3588_PYTHON = "/root/miniconda3/envs/fast-lewm/bin/python"


class RK3588Planner:
    """通过 SSH 与 RK3588 planner server 通信。"""

    def __init__(self, cem_iters=30, num_samples=300, topk=30, horizon=5):
        self.cem_iters = cem_iters
        self.num_samples = num_samples
        self.topk = topk
        self.horizon = horizon
        self.process = None
        self._start_server()

    def _start_server(self):
        """启动 RK3588 上的 planner server。"""
        cmd = [
            "ssh", "-F", SSH_CONFIG, RK3588_HOST,
            f"cd {RK3588_WORKDIR} && taskset -c 4-7 {RK3588_PYTHON} -u rk3588_planner_server.py"
        ]
        print(f"启动 RK3588 planner server...", file=sys.stderr)
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # 用线程实时打印 server 的 stderr
        def print_stderr():
            for line in self.process.stderr:
                print(f"  [server] {line.rstrip()}", file=sys.stderr)

        self.stderr_thread = threading.Thread(target=print_stderr, daemon=True)
        self.stderr_thread.start()

        # 等待 server 初始化完成（模型加载 + RKNN 初始化约需 30 秒）
        print("等待 server 初始化（35秒）...", file=sys.stderr)
        time.sleep(35)

        # 检查进程是否还在运行
        if self.process.poll() is not None:
            raise RuntimeError(f"Server 启动失败，退出码: {self.process.returncode}")

        print("RK3588 planner server 已启动", file=sys.stderr)

    def plan(self, image_np, goal_image_np):
        """
        发送规划请求，返回动作序列。
        image_np: [224, 224, 3] uint8
        goal_image_np: [224, 224, 3] uint8
        返回: (actions [horizon, 2], plan_time_ms, final_cost)
        """
        # 用 base64 + PNG 压缩图像
        def encode_image(img):
            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        req = {
            "image": encode_image(image_np),
            "goal_image": encode_image(goal_image_np),
            "cem_iters": self.cem_iters,
            "num_samples": self.num_samples,
            "topk": self.topk,
            "horizon": self.horizon,
        }
        req_json = json.dumps(req) + "\n"

        # 发送请求
        self.process.stdin.write(req_json)
        self.process.stdin.flush()

        # 读取响应（跳过 RKNN 日志等非 JSON 行）
        resp_line = ""
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            if line.startswith("{"):
                resp_line = line
                break

        if not resp_line:
            retcode = self.process.poll()
            raise RuntimeError(f"RK3588 server 连接断开, retcode={retcode}")

        resp = json.loads(resp_line)

        if "error" in resp:
            raise RuntimeError(f"RK3588 server 错误: {resp['error']}")

        return (
            np.array(resp["actions"]),
            resp["plan_time_ms"],
            resp["final_cost"],
        )

    def close(self):
        """关闭 server 连接。"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


def run_episode(env, planner, seed=42, verbose=True):
    """运行一个 episode，返回 (success, steps, total_plan_time, avg_plan_time)。"""
    obs, info = env.reset(seed=seed)

    # 从官方环境获取图像和 goal
    image = info["pixels"]  # [224, 224, 3] uint8
    goal_image = info["goal"]  # [224, 224, 3] uint8

    if verbose:
        print(f"  初始: agent_pos={info['pos_agent']}, "
              f"block_pos={info['block_pose'][:2]}, "
              f"goal_pos={info['goal_pose'][:2]}", file=sys.stderr)

    total_plan_time = 0
    plan_count = 0

    for step in range(env.spec.max_episode_steps):
        # 调用 RK3588 规划
        t0 = time.perf_counter()
        actions, plan_time_ms, cost = planner.plan(image, goal_image)
        plan_wall_time = (time.perf_counter() - t0) * 1000

        total_plan_time += plan_time_ms
        plan_count += 1

        # 执行第一个动作（receding horizon control，位置控制）
        action = actions[0]  # [2]，范围 [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))

        # 更新图像
        image = info["pixels"]

        if verbose and step % 10 == 0:
            print(f"  step {step}: reward={reward:.1f}, "
                  f"plan_time={plan_time_ms:.0f}ms (wall={plan_wall_time:.0f}ms), "
                  f"cost={cost:.2f}, action=[{action[0]:.3f},{action[1]:.3f}]",
                  file=sys.stderr)

        if terminated or truncated:
            success = info.get("success", reward > 0)
            if verbose:
                print(f"  Episode 结束: success={success}, steps={step+1}, "
                      f"avg_plan_time={total_plan_time/plan_count:.0f}ms",
                      file=sys.stderr)
            return success, step + 1, total_plan_time, total_plan_time / plan_count

    return False, env.spec.max_episode_steps, total_plan_time, total_plan_time / plan_count


def main():
    parser = argparse.ArgumentParser(description="PushT 评估（官方环境 + RK3588 规划）")
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数")
    parser.add_argument("--cem_iters", type=int, default=30, help="CEM 迭代次数")
    parser.add_argument("--num_samples", type=int, default=300, help="CEM 采样数")
    parser.add_argument("--topk", type=int, default=30, help="CEM 精英数")
    parser.add_argument("--horizon", type=int, default=5, help="规划 horizon")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print(f"=== PushT 评估（官方 stable-worldmodel 环境）===", file=sys.stderr)
    print(f"episodes={args.episodes}, cem_iters={args.cem_iters}, "
          f"num_samples={args.num_samples}, topk={args.topk}, horizon={args.horizon}",
          file=sys.stderr)

    # 初始化官方环境
    env = gym.make('swm/PushT-v1', max_episode_steps=200)
    env = AddPixelsWrapper(env, pixels_shape=(224, 224))
    print("官方 PushT 环境初始化完成", file=sys.stderr)

    # 初始化 RK3588 planner
    planner = RK3588Planner(
        cem_iters=args.cem_iters,
        num_samples=args.num_samples,
        topk=args.topk,
        horizon=args.horizon,
    )

    # 运行评估
    results = []
    successes = 0
    total_steps = 0
    total_plan_time = 0

    try:
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep+1}/{args.episodes} ---", file=sys.stderr)
            success, steps, plan_time, avg_plan = run_episode(
                env, planner, seed=args.seed + ep, verbose=True
            )
            results.append({
                "episode": ep,
                "success": success,
                "steps": steps,
                "total_plan_time_ms": plan_time,
                "avg_plan_time_ms": avg_plan,
            })
            if success:
                successes += 1
            total_steps += steps
            total_plan_time += plan_time

    finally:
        planner.close()
        env.close()

    # 总结
    print(f"\n=== 评估总结 ===", file=sys.stderr)
    print(f"成功率: {successes}/{args.episodes} ({successes/args.episodes*100:.1f}%)", file=sys.stderr)
    print(f"平均步数: {total_steps/args.episodes:.1f}", file=sys.stderr)
    print(f"平均规划时间: {total_plan_time/total_steps:.0f}ms/step", file=sys.stderr)

    # 保存结果
    output = {
        "config": vars(args),
        "summary": {
            "success_rate": successes / args.episodes,
            "successes": successes,
            "total_episodes": args.episodes,
            "avg_steps": total_steps / args.episodes,
            "avg_plan_time_ms": total_plan_time / total_steps,
        },
        "episodes": results,
    }
    output_path = "/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/pusht_eval_official_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
