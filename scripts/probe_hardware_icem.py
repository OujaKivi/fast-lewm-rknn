#!/usr/bin/env python3
"""Paired fixed-observation benchmark for hardware-aware CEM schedules."""

import argparse
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import stable_worldmodel.envs  # noqa: F401
from stable_worldmodel.wrapper import AddPixelsWrapper


SSH_CONFIG = "/Users/wangjiwei/.ssh/config_rknn"


def encode_image(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_server(flags, request, repeats, mode="npu"):
    command = [
        "ssh", "-F", SSH_CONFIG, "rk3588",
        "cd /root/Fast-LeWorldModel && taskset -c 4-7 "
        "/root/miniconda3/envs/fast-lewm/bin/python -u "
        f"rk3588_planner_server.py --mode {mode} --cem-steps 30 {flags}",
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    responses = []
    try:
        for index in range(repeats + 1):
            request["reset"] = True
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            while True:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"planner exited before responding (code={process.poll()})"
                    )
                if line.startswith("{"):
                    response = json.loads(line)
                    break
            if "error" in response:
                raise RuntimeError(response["error"])
            if index:
                responses.append(response)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        stderr = process.stderr.read()
        if process.returncode not in (None, 0, -15):
            print(stderr, file=sys.stderr)
    return responses


def summarize(name, responses, reference_action):
    actions = np.asarray([item["action"] for item in responses])
    action_delta = actions - reference_action
    profiles = [item["profiling"] for item in responses]
    return {
        "name": name,
        "repeats": len(responses),
        "mean_total_ms": float(np.mean([x["time_total"] for x in responses]) * 1000),
        "std_total_ms": float(np.std([x["time_total"] for x in responses]) * 1000),
        "mean_action_encoder_ms": float(np.mean([x["action_encoder"] for x in profiles]) * 1000),
        "mean_predictor_ms": float(np.mean([x["predictor"] for x in profiles]) * 1000),
        "mean_cost": float(np.mean([x["cost"] for x in responses])),
        "action_rmse_vs_baseline": float(np.sqrt(np.mean(action_delta ** 2))),
        "action_max_abs_vs_baseline": float(np.max(np.abs(action_delta))),
        "cem": responses[0]["cem"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["npu", "npu-hybrid-action"], default="npu")
    parser.add_argument(
        "--output", default="results/hardware_icem_fixed_observation.json"
    )
    args = parser.parse_args()

    env = AddPixelsWrapper(
        gym.make("swm/PushT-v1", max_episode_steps=200),
        pixels_shape=(224, 224),
    )
    _, info = env.reset(seed=args.seed)
    request = {
        "current_image": encode_image(info["pixels"]),
        "goal_image": encode_image(info["goal"]),
        "executed_actions": 25,
    }
    configs = {
        "fixed_300": "--num-samples 300 --topk 30",
        "tiered_300_150_64": (
            "--num-samples 300 --topk 30 "
            "--candidate-schedule 300x10,150x10,64x10"
        ),
        "tiered_with_elite_reuse": (
            "--num-samples 300 --topk 30 "
            "--candidate-schedule 300x10,150x10,64x10 "
            "--elite-reuse-fraction 0.3"
        ),
    }

    raw = {}
    for name, flags in configs.items():
        print(f"Running {name}...", file=sys.stderr)
        raw[name] = run_server(flags, request.copy(), args.repeats, args.mode)
    reference_action = np.asarray(raw["fixed_300"][0]["action"])
    result = {
        "seed": args.seed,
        "mode": args.mode,
        "note": "Fixed-observation deterministic latency/quality proxy; not task success.",
        "results": [
            summarize(name, responses, reference_action)
            for name, responses in raw.items()
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
