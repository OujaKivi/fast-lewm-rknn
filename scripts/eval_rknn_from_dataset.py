#!/usr/bin/env python3
"""Evaluate the board planner with the official dataset-goal protocol."""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

import stable_worldmodel.envs  # noqa: F401
from stable_worldmodel.wrapper import AddPixelsWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pusht_eval_official import RK3588Planner  # noqa: E402


def select_rows(handle, count, seed=42, goal_offset=25):
    episode_idx = handle["episode_idx"][:]
    step_idx = handle["step_idx"][:]
    lengths = handle["ep_len"][:]
    valid = step_idx <= (lengths[episode_idx] - goal_offset - 1)
    valid_indices = np.flatnonzero(valid)
    generator = np.random.default_rng(seed)
    picked = generator.choice(len(valid_indices) - 1, size=count, replace=False)
    return np.sort(valid_indices[picked])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--mode", default="npu", choices=[
        "cpu", "npu", "npu-image", "npu-predictor", "npu-hybrid-action"
    ])
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-schedule", default=None)
    parser.add_argument("--elite-reuse-fraction", type=float, default=0.0)
    parser.add_argument("--planner-algorithm", choices=["cem", "icem"], default="cem")
    parser.add_argument("--icem-population-decay", type=float, default=1.25)
    parser.add_argument("--icem-graph-snap", action="store_true")
    parser.add_argument("--icem-adaptive-extension", action="store_true")
    parser.add_argument("--icem-extend-gain-threshold", type=float, default=0.15)
    parser.add_argument("--row-seed", type=int, default=42)
    args = parser.parse_args()

    handle = h5py.File(args.dataset, "r")
    action = handle["action"][:]
    action_mean = action.mean(axis=0)
    action_std = action.std(axis=0)
    rows = select_rows(handle, args.episodes, seed=args.row_seed)
    planner = RK3588Planner(
        cem_iters=args.cem_steps,
        mode=args.mode,
        seed=42,
        candidate_schedule=args.candidate_schedule,
        elite_reuse_fraction=args.elite_reuse_fraction,
        planner_algorithm=args.planner_algorithm,
        icem_population_decay=args.icem_population_decay,
        icem_graph_snap=args.icem_graph_snap,
        icem_adaptive_extension=args.icem_adaptive_extension,
        icem_extend_gain_threshold=args.icem_extend_gain_threshold,
    )
    results = []
    try:
        for episode_number, row in enumerate(rows):
            goal_row = int(row + 25)
            env = AddPixelsWrapper(
                gym.make("swm/PushT-v1", max_episode_steps=100),
                pixels_shape=(224, 224),
            )
            env.reset()
            env.unwrapped._set_state(handle["state"][row])
            env.unwrapped._set_goal_state(handle["state"][goal_row])
            current_image = handle["pixels"][row]
            goal_image = handle["pixels"][goal_row]
            success = False
            plan_times = []
            costs = []
            plan_traces = []
            for replan in range(2):
                plan, plan_ms, cost, metadata = planner.plan(
                    current_image, goal_image,
                    reset=(replan == 0), executed_actions=25,
                )
                plan_times.append(plan_ms)
                costs.append(cost)
                if args.save_traces:
                    plan_traces.append([
                        entry["best_cost"] for entry in metadata["trace"]
                    ])
                for normalized_action in plan:
                    env_action = normalized_action * action_std + action_mean
                    _, _, terminated, truncated, info = env.step(
                        env_action.astype(np.float32)
                    )
                    success = success or bool(terminated)
                    current_image = info["pixels"]
                    if terminated or truncated:
                        break
                if success:
                    break
            result = {
                "episode": episode_number,
                "dataset_row": int(row),
                "success": success,
                "plan_times_ms": plan_times,
                "costs": costs,
            }
            if args.save_traces:
                result["plan_best_cost_traces"] = plan_traces
            results.append(result)
            env.close()
            print(f"{episode_number + 1}/{len(rows)} success={success}")
    finally:
        planner.close()
        handle.close()

    output = {
        "mode": args.mode,
        "planner_algorithm": args.planner_algorithm,
        "icem_population_decay": args.icem_population_decay,
        "icem_graph_snap": args.icem_graph_snap,
        "icem_adaptive_extension": args.icem_adaptive_extension,
        "icem_extend_gain_threshold": args.icem_extend_gain_threshold,
        "row_seed": args.row_seed,
        "candidate_schedule": args.candidate_schedule,
        "elite_reuse_fraction": args.elite_reuse_fraction,
        "episodes": args.episodes,
        "cem_steps": args.cem_steps,
        "successes": sum(item["success"] for item in results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "mean_replan_ms": float(np.mean([
            value for item in results for value in item["plan_times_ms"]
        ])),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "rows": rows.tolist(),
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in (
        "mode", "successes", "episodes", "success_rate", "mean_replan_ms"
    )}, indent=2))


if __name__ == "__main__":
    main()
