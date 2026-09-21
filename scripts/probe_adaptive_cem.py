#!/usr/bin/env python3
"""Probe warm-start and adaptive CEM on a fixed PushT observation."""

import argparse
import base64
import io
import json
import subprocess
import sys

import gymnasium as gym
from PIL import Image

import stable_worldmodel.envs  # noqa: F401
from stable_worldmodel.wrapper import AddPixelsWrapper


def encode_image(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-steps", action="store_true")
    args = parser.parse_args()

    env = AddPixelsWrapper(
        gym.make("swm/PushT-v1", max_episode_steps=200),
        pixels_shape=(224, 224),
    )
    _, info = env.reset(seed=args.seed)
    request = {
        "current_image": encode_image(info["pixels"]),
        "goal_image": encode_image(info["goal"]),
        "executed_actions": 1,
    }

    server_flags = "--mode npu --warm-start --cem-steps 30 --num-samples 300"
    if not args.fixed_steps:
        server_flags += " --adaptive-cem --min-cem-steps 8"
    command = [
        "ssh", "-F", "/Users/wangjiwei/.ssh/config_rknn", "rk3588",
        "cd /root/Fast-LeWorldModel && taskset -c 4-7 "
        "/root/miniconda3/envs/fast-lewm/bin/python -u "
        f"rk3588_planner_server.py {server_flags}",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        for index in range(args.repeats):
            request["reset"] = index == 0
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response_line = ""
            while True:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"planner exited before responding (code={process.poll()})"
                    )
                if line.startswith("{"):
                    response_line = line
                    break
            response = json.loads(response_line)
            print(json.dumps({
                "request": index + 1,
                "time_total_ms": round(response["time_total"] * 1000, 2),
                "cost": response["cost"],
                "cem": response["cem"],
            }))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        stderr = process.stderr.read()
        if stderr:
            print(stderr, file=sys.stderr)
        env.close()


if __name__ == "__main__":
    main()
