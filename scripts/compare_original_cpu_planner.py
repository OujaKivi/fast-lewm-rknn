#!/usr/bin/env python3
"""Compare official Fast-LeWM CEM with the deployment CPU planner."""

import base64
import argparse
import io
import json
import sys
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from PIL import Image
from torchvision.transforms import v2 as transforms
from transformers import ViTConfig, ViTModel

import stable_worldmodel.envs  # noqa: F401
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.wrapper import AddPixelsWrapper


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "Fast-LeWorldModel"
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(ROOT))

from rk3588_planner_server import FastLeWMPlanner  # noqa: E402


def encode_image(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def preprocess(array):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.Resize(size=(224, 224)),
    ])
    return transform(array).unsqueeze(0).unsqueeze(0)


def load_reconstructed_model():
    source = torch.load(
        MODEL_ROOT / "weights/Fast-lewm_pusht_object.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    encoder = ViTModel(ViTConfig(
        hidden_size=192, num_attention_heads=3, num_hidden_layers=12,
        intermediate_size=768, patch_size=14, image_size=224,
        num_channels=3,
    ), add_pooling_layer=False)
    encoder.load_state_dict(source.encoder.state_dict(), strict=True)
    source.encoder = encoder
    return source.eval().requires_grad_(False)


def run_board(current, goal, mode, steps):
    command = [
        "ssh", "-F", "/Users/wangjiwei/.ssh/config_rknn", "rk3588",
        "cd /root/Fast-LeWorldModel && taskset -c 4-7 "
        "/root/miniconda3/envs/fast-lewm/bin/python -u "
        f"rk3588_planner_server.py --mode {mode} --seed 42 --cem-steps {steps}",
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        request = {
            "current_image": encode_image(current),
            "goal_image": encode_image(goal),
            "executed_actions": 25,
            "reset": True,
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(process.stderr.read())
            if line.startswith("{"):
                response = json.loads(line)
                break
        if "error" in response:
            raise RuntimeError(response["error"])
        return response
    finally:
        process.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    env = AddPixelsWrapper(
        gym.make("swm/PushT-v1", max_episode_steps=200),
        pixels_shape=(224, 224),
    )
    _, info = env.reset(seed=42)
    current = info["pixels"]
    goal = info["goal"]

    model = load_reconstructed_model()
    solver = CEMSolver(
        model=model, batch_size=1, num_samples=300, var_scale=1.0,
        n_steps=args.steps, topk=30, device="cpu", seed=42,
    )
    config = PlanConfig(
        horizon=1, receding_horizon=1, action_block=25, warm_start=False
    )
    solver.configure(
        action_space=Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32),
        n_envs=1,
        config=config,
    )
    official = solver.solve({
        "pixels": preprocess(current),
        "goal": preprocess(goal),
        "action": torch.zeros(1, 1, 2),
    })
    official_action = official["actions"][0, 0].numpy()

    deployment = FastLeWMPlanner(mode="cpu")
    deployment.seed = 42
    deployment.n_steps = args.steps
    result = deployment.plan(
        encode_image(current), encode_image(goal), reset=True,
        executed_actions=25,
    )
    deployment_action = np.asarray(result["action"]).reshape(-1)
    delta = deployment_action - official_action
    report = {
        "official_final_elite_mean_cost": official["costs"][0],
        "deployment_final_elite_mean_cost": result["cost"],
        "action_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "action_max_abs": float(np.max(np.abs(delta))),
        "official_action_head": official_action[:6].tolist(),
        "deployment_action_head": deployment_action[:6].tolist(),
    }
    if args.board:
        for mode in ("cpu", "npu-image", "npu-predictor", "npu"):
            board = run_board(current, goal, mode, args.steps)
            board_action = np.asarray(board["action"]).reshape(-1)
            board_delta = board_action - official_action
            report[f"board_{mode}"] = {
                "final_elite_mean_cost": board["cost"],
                "action_rmse_vs_official": float(
                    np.sqrt(np.mean(board_delta ** 2))
                ),
                "action_max_abs_vs_official": float(np.max(np.abs(board_delta))),
                "time_total_ms": board["time_total"] * 1000,
                "action_head": board_action[:6].tolist(),
            }
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
