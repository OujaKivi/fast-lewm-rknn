#!/usr/bin/env python3
"""Evaluate one LIBERO episode with local or remote SmolVLA inference.

The simulator and LeRobot processors stay on the evaluation host. A remote
server receives only processed observations and returns normalized actions.
Use SSH forwarding for the TCP connection; the connection protocol is pickle.
"""

import argparse
import json
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import add_envs_task, preprocess_observation
from lerobot.utils.constants import ACTION


def load_policy(model_path, device, vlm_path):
    config = PreTrainedConfig.from_pretrained(model_path)
    config.device = device
    config.load_vlm_weights = False
    if vlm_path:
        config.vlm_model_name = vlm_path
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config).eval()
    return policy, config


def serve(args):
    policy, _ = load_policy(args.model_path, args.device, args.vlm_path)
    runtimes = []
    if args.vision_rknn_path or args.prefill_rknn_path or args.denoise_rknn_path:
        from rknnlite.api import RKNNLite

    def load_rknn(path):
        runtime = RKNNLite()
        if runtime.load_rknn(path) != 0:
            raise RuntimeError(f"failed to load RKNN graph: {path}")
        if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError(f"failed to initialize RKNN runtime: {path}")
        runtimes.append(runtime)
        return runtime

    vision_runtime = load_rknn(args.vision_rknn_path) if args.vision_rknn_path else None
    prefill_runtime = load_rknn(args.prefill_rknn_path) if args.prefill_rknn_path else None
    denoise_runtime = load_rknn(args.denoise_rknn_path) if args.denoise_rknn_path else None
    stage_times = {"prefill_s": 0.0, "denoise_s": 0.0}
    if vision_runtime is not None:

        def npu_embed(image):
            output = vision_runtime.inference(
                inputs=[image.detach().cpu().numpy()], data_format=["nchw"]
            )[0]
            return torch.from_numpy(output).to(dtype=image.dtype, device=image.device)

        policy.model.vlm_with_expert.embed_image = npu_embed
    if prefill_runtime is not None:
        original_forward = policy.model.vlm_with_expert.forward

        def npu_prefill(*forward_args, **forward_kwargs):
            if not forward_kwargs.get("fill_kv_cache", False):
                return original_forward(*forward_args, **forward_kwargs)
            prefix = forward_kwargs["inputs_embeds"][0]
            if prefix.shape != (1, args.prefix_length, 960):
                raise ValueError(f"expected [1,{args.prefix_length},960] prefill, got {prefix.shape}")
            positions = forward_kwargs["position_ids"]
            expected = torch.arange(args.prefix_length, device=positions.device)[None]
            if not torch.equal(positions, expected):
                raise ValueError("prefill graph requires all prefix tokens to be valid")
            started = time.perf_counter()
            outputs = prefill_runtime.inference(
                inputs=[prefix.detach().float().cpu().numpy()], data_format=None
            )
            stage_times["prefill_s"] += time.perf_counter() - started
            if len(outputs) != 2 * args.layers:
                raise RuntimeError(f"expected {2 * args.layers} K/V outputs, got {len(outputs)}")
            cache = {
                layer: {
                    "key_states": torch.from_numpy(outputs[2 * layer].copy()),
                    "value_states": torch.from_numpy(outputs[2 * layer + 1].copy()),
                }
                for layer in range(args.layers)
            }
            return [None, None], cache

        policy.model.vlm_with_expert.forward = npu_prefill
    if denoise_runtime is not None:

        def npu_denoise(prefix_pad_masks, past_key_values, x_t, timestep):
            if prefix_pad_masks.shape != (1, args.prefix_length) or not bool(prefix_pad_masks.all()):
                raise ValueError("denoising graph requires its fixed valid prefix length")
            if len(past_key_values) != args.layers:
                raise ValueError(f"expected {args.layers} cache layers, got {len(past_key_values)}")
            suffix, _, _ = policy.model.embed_suffix(x_t, timestep)
            inputs = [suffix.detach().float().cpu().numpy()]
            for layer in range(args.layers):
                for name in ("key_states", "value_states"):
                    value = past_key_values[layer][name].detach().float().cpu().numpy()
                    inputs.append(value.transpose(0, 2, 3, 1).copy())
            started = time.perf_counter()
            output = denoise_runtime.inference(inputs=inputs, data_format=None)[0]
            stage_times["denoise_s"] += time.perf_counter() - started
            return torch.from_numpy(output.copy())

        policy.model.denoise_step = npu_denoise
    try:
        serve_connections(args, policy, stage_times)
    finally:
        for runtime in runtimes:
            runtime.release()


def serve_connections(args, policy, stage_times):
    embed_image = policy.model.vlm_with_expert.embed_image
    vision_times = []

    def timed_embed(image):
        started = time.perf_counter()
        result = embed_image(image)
        vision_times.append(time.perf_counter() - started)
        return result

    policy.model.vlm_with_expert.embed_image = timed_embed
    with Listener((args.listen, args.port), authkey=args.authkey.encode()) as listener:
        print(f"ready {args.listen}:{args.port}", flush=True)
        while True:
            try:
                conn = listener.accept()
            except (ConnectionError, EOFError):
                continue
            with conn:
                while True:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break
                    if request["op"] == "reset":
                        torch.manual_seed(request["seed"])
                        policy.reset()
                        conn.send({"ok": True})
                    elif request["op"] == "act":
                        vision_times.clear()
                        stage_times["prefill_s"] = 0.0
                        stage_times["denoise_s"] = 0.0
                        observation = {
                            key: torch.from_numpy(value).to(args.device)
                            if isinstance(value, np.ndarray) else value
                            for key, value in request["observation"].items()
                        }
                        with torch.inference_mode():
                            action = policy.select_action(observation)
                        if args.device == "cuda":
                            torch.cuda.synchronize()
                        elif args.device == "mps":
                            torch.mps.synchronize()
                        conn.send({
                            "action": action.cpu().numpy(),
                            "vision_s": sum(vision_times),
                            "vision_calls": len(vision_times),
                            **stage_times,
                        })
                    elif request["op"] == "stop":
                        conn.send({"ok": True})
                        return
                    else:
                        raise ValueError(f"unknown operation: {request['op']}")


def evaluate(args):
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = args.device if args.host is None else "cpu"
    env_config = LiberoEnv(
        task=args.suite,
        task_ids=[args.task_id],
        observation_height=256,
        observation_width=256,
        episode_length=args.max_steps,
    )
    env = make_env(env_config, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, config)

    policy = None
    conn = None
    if args.host:
        conn = Client((args.host, args.port), authkey=args.authkey.encode())
    else:
        policy, _ = load_policy(args.model_path, args.device, args.vlm_path)

    started = time.perf_counter()
    inference_seconds = []
    preprocess_seconds = []
    postprocess_seconds = []
    environment_seconds = []
    vision_seconds = []
    prefill_seconds = []
    denoise_seconds = []
    first_action = None
    input_shapes = None
    success = False
    rewards = []
    try:
        if conn:
            conn.send({"op": "reset", "seed": args.seed})
            assert conn.recv()["ok"]
        else:
            torch.manual_seed(args.seed)
            policy.reset()
        observation, _ = env.reset(seed=[args.seed])
        max_steps = env.call("_max_episode_steps")[0]
        steps = 0
        for step in range(max_steps):
            step_started = time.perf_counter()
            batch = preprocess_observation(observation)
            batch = add_envs_task(env, batch)
            batch = env_preprocessor(batch)
            batch = preprocessor(batch)
            if not conn and args.device == "cuda":
                torch.cuda.synchronize()
            elif not conn and args.device == "mps":
                torch.mps.synchronize()
            preprocess_seconds.append(time.perf_counter() - step_started)
            if input_shapes is None:
                input_shapes = {
                    key: list(value.shape)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
            t0 = time.perf_counter()
            if conn:
                conn.send({
                    "op": "act",
                    "observation": {
                        key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    },
                })
                response = conn.recv()
                action = torch.from_numpy(response["action"])
                vision_seconds.append(response["vision_s"])
                prefill_seconds.append(response["prefill_s"])
                denoise_seconds.append(response["denoise_s"])
            else:
                with torch.inference_mode():
                    action = policy.select_action(batch)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                elif args.device == "mps":
                    torch.mps.synchronize()
            inference_seconds.append(time.perf_counter() - t0)
            postprocess_started = time.perf_counter()
            action = postprocessor(action)
            action = env_postprocessor({ACTION: action})[ACTION]
            if first_action is None:
                first_action = action[0].cpu().tolist()
            action_numpy = action.cpu().numpy()
            postprocess_seconds.append(time.perf_counter() - postprocess_started)
            environment_started = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            environment_seconds.append(time.perf_counter() - environment_started)
            rewards.append(float(reward[0]))
            steps = step + 1
            if "final_info" in info:
                success = bool(info["final_info"]["is_success"][0])
            if bool(terminated[0] or truncated[0]):
                break
    finally:
        if conn:
            conn.close()
        env.close()

    result = {
        "checkpoint": str(args.model_path),
        "suite": args.suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "device": args.device if not args.host else f"remote:{args.host}:{args.port}",
        "steps": steps,
        "success": success,
        "reward_sum": sum(rewards),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "inference_s_total": round(sum(inference_seconds), 3),
        "inference_s_mean": round(float(np.mean(inference_seconds)), 3),
        "inference_s_p95": round(float(np.percentile(inference_seconds, 95)), 3),
        "preprocess_s_total": round(sum(preprocess_seconds), 3),
        "postprocess_s_total": round(sum(postprocess_seconds), 3),
        "environment_s_total": round(sum(environment_seconds), 3),
        "first_action": first_action,
        "input_shapes": input_shapes,
    }
    if vision_seconds:
        result["vision_s_total"] = round(sum(vision_seconds), 3)
        result["prefill_s_total"] = round(sum(prefill_seconds), 3)
        result["denoise_s_total"] = round(sum(denoise_seconds), 3)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "eval"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=47651)
    parser.add_argument("--authkey", default="smolvla-local-eval")
    parser.add_argument("--vision-rknn-path")
    parser.add_argument("--prefill-rknn-path")
    parser.add_argument("--denoise-rknn-path")
    parser.add_argument("--prefix-length", type=int, default=149)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.prefix_length < 1 or args.layers < 1:
        parser.error("--prefix-length and --layers must be positive")
    if args.device != "cpu" and any(
        (args.vision_rknn_path, args.prefill_rknn_path, args.denoise_rknn_path)
    ):
        parser.error("RKNN graphs require --device cpu for the host-side steps")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.mode == "serve":
        serve(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
