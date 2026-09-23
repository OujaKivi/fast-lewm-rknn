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

# StandardScaler fitted to the PushT training actions.
ACTION_MEAN = np.array([-0.0078125644, 0.0068606872], dtype=np.float32)
ACTION_STD = np.array([0.2084674428, 0.2067486264], dtype=np.float32)


class RK3588Planner:
    """通过 SSH 与 RK3588 planner server 通信。"""

    def __init__(self, cem_iters=30, num_samples=300, topk=30, horizon=5,
                 mode="npu", warm_start=False, adaptive_cem=False,
                 min_cem_steps=8, candidate_schedule=None,
                 elite_reuse_fraction=0.0, seed=42):
        self.cem_iters = cem_iters
        self.num_samples = num_samples
        self.topk = topk
        self.horizon = horizon
        self.mode = mode
        self.warm_start = warm_start
        self.adaptive_cem = adaptive_cem
        self.min_cem_steps = min_cem_steps
        self.candidate_schedule = candidate_schedule
        self.elite_reuse_fraction = elite_reuse_fraction
        self.seed = seed
        self.process = None
        self._start_server()

    def _start_server(self):
        """启动 RK3588 上的 planner server。"""
        server_args = [
            f"--mode {self.mode}",
            f"--cem-steps {self.cem_iters}",
            f"--num-samples {self.num_samples}",
            f"--topk {self.topk}",
            f"--min-cem-steps {self.min_cem_steps}",
            f"--elite-reuse-fraction {self.elite_reuse_fraction}",
            f"--seed {self.seed}",
        ]
        if self.candidate_schedule:
            server_args.append(f"--candidate-schedule {self.candidate_schedule}")
        if self.warm_start:
            server_args.append("--warm-start")
        if self.adaptive_cem:
            server_args.append("--adaptive-cem")
        cmd = [
            "ssh", "-F", SSH_CONFIG, RK3588_HOST,
            f"cd {RK3588_WORKDIR} && taskset -c 4-7 {RK3588_PYTHON} -u "
            f"rk3588_planner_server.py {' '.join(server_args)}"
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
        self.ready_event = threading.Event()

        def print_stderr():
            for line in self.process.stderr:
                print(f"  [server] {line.rstrip()}", file=sys.stderr)
                if "[Planner] Ready." in line:
                    self.ready_event.set()

        self.stderr_thread = threading.Thread(target=print_stderr, daemon=True)
        self.stderr_thread.start()

        print("等待 server 初始化...", file=sys.stderr)
        if not self.ready_event.wait(timeout=60):
            raise RuntimeError("Server 初始化超时")

        # 检查进程是否还在运行
        if self.process.poll() is not None:
            raise RuntimeError(f"Server 启动失败，退出码: {self.process.returncode}")

        print("RK3588 planner server 已启动", file=sys.stderr)

    def plan(self, image_np, goal_image_np, reset=False, executed_actions=25):
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
            "current_image": encode_image(image_np),
            "goal_image": encode_image(goal_image_np),
            "executed_actions": executed_actions,
            "reset": reset,
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
            np.array(resp["action"]).reshape(-1, 2),
            resp["time_total"] * 1000,
            resp["cost"],
            resp.get("cem", {}),
        )

    def close(self):
        """关闭 server 连接。"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


def run_episode(env, planner, seed=42, verbose=True, max_steps=None,
                replan_every=25):
    """运行一个 episode，返回成功率、耗时与 reward 统计。"""
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
    cem_iterations = []
    actions = None
    last_plan_time_ms = 0.0
    last_cost = float("nan")
    last_cem = {}
    episode_return = 0.0
    best_reward = float("-inf")

    episode_limit = env.spec.max_episode_steps if max_steps is None else max_steps
    for step in range(episode_limit):
        action_offset = step % replan_every
        if action_offset == 0:
            actions, last_plan_time_ms, last_cost, last_cem = planner.plan(
                image,
                goal_image,
                reset=(step == 0),
                executed_actions=replan_every,
            )
            total_plan_time += last_plan_time_ms
            plan_count += 1
            cem_iterations.append(
                last_cem.get("iterations_used", planner.cem_iters)
            )
        if action_offset >= len(actions):
            raise RuntimeError(
                f"replan_every={replan_every} exceeds plan length={len(actions)}"
            )

        # Execute the next primitive action from the packed 25-action plan.
        # CEM searches in the checkpoint's standardized action space.
        action = actions[action_offset] * ACTION_STD + ACTION_MEAN
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        episode_return += float(reward)
        best_reward = max(best_reward, float(reward))

        # 更新图像
        image = info["pixels"]

        if verbose and step % 10 == 0:
            print(f"  step {step}: reward={reward:.1f}, "
                  f"last_plan_time={last_plan_time_ms:.0f}ms, "
                  f"cost={last_cost:.2f}, "
                  f"cem_steps={last_cem.get('iterations_used', '?')}, "
                  f"warm={last_cem.get('warm_started', False)}, "
                  f"action=[{action[0]:.3f},{action[1]:.3f}]",
                  file=sys.stderr)

        if terminated or truncated:
            success = info.get("success", reward > 0)
            if verbose:
                print(f"  Episode 结束: success={success}, steps={step+1}, "
                      f"avg_plan_time={total_plan_time/plan_count:.0f}ms",
                      file=sys.stderr)
            return (bool(success), step + 1, total_plan_time,
                    total_plan_time / plan_count, cem_iterations,
                    episode_return, best_reward, float(reward))

    return (False, episode_limit, total_plan_time,
            total_plan_time / plan_count, cem_iterations,
            episode_return, best_reward, float(reward))


def main():
    parser = argparse.ArgumentParser(description="PushT 评估（官方环境 + RK3588 规划）")
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数")
    parser.add_argument("--cem_iters", type=int, default=30, help="CEM 迭代次数")
    parser.add_argument("--num_samples", type=int, default=300, help="CEM 采样数")
    parser.add_argument("--topk", type=int, default=30, help="CEM 精英数")
    parser.add_argument("--horizon", type=int, default=5, help="规划 horizon")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mode", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--adaptive_cem", action="store_true")
    parser.add_argument("--min_cem_steps", type=int, default=8)
    parser.add_argument(
        "--candidate_schedule", default=None,
        help="Per-iteration candidates, e.g. 300x10,150x10,64x10",
    )
    parser.add_argument("--elite_reuse_fraction", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--replan_every", type=int, default=25,
        help="Environment actions executed before replanning (paper config: 25)",
    )
    parser.add_argument(
        "--output", default="results/pusht_eval_official_result.json",
        help="JSON result path",
    )
    args = parser.parse_args()
    if not 1 <= args.replan_every <= 25:
        parser.error("--replan_every must be in [1, 25]")
    if args.horizon != 5:
        parser.error("this checkpoint requires --horizon 5 action blocks")

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
        mode=args.mode,
        warm_start=args.warm_start,
        adaptive_cem=args.adaptive_cem,
        min_cem_steps=args.min_cem_steps,
        candidate_schedule=args.candidate_schedule,
        elite_reuse_fraction=args.elite_reuse_fraction,
        seed=args.seed,
    )

    # 运行评估
    results = []
    successes = 0
    total_steps = 0
    total_plan_time = 0

    try:
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep+1}/{args.episodes} ---", file=sys.stderr)
            (success, steps, plan_time, avg_plan, cem_iterations,
             episode_return, best_reward, final_reward) = run_episode(
                env, planner, seed=args.seed + ep, verbose=True,
                max_steps=args.max_steps,
                replan_every=args.replan_every,
            )
            results.append({
                "episode": ep,
                "success": success,
                "steps": steps,
                "total_plan_time_ms": plan_time,
                "avg_plan_time_ms": avg_plan,
                "avg_cem_iterations": float(np.mean(cem_iterations)),
                "cem_iterations": cem_iterations,
                "episode_return": episode_return,
                "best_reward": best_reward,
                "final_reward": final_reward,
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
            "amortized_plan_time_per_env_step_ms": total_plan_time / total_steps,
            "avg_plan_time_per_replan_ms": float(np.mean([
                episode["avg_plan_time_ms"] for episode in results
            ])),
            "mean_episode_return": float(np.mean([
                episode["episode_return"] for episode in results
            ])),
            "mean_best_reward": float(np.mean([
                episode["best_reward"] for episode in results
            ])),
            "mean_final_reward": float(np.mean([
                episode["final_reward"] for episode in results
            ])),
        },
        "episodes": results,
    }
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
