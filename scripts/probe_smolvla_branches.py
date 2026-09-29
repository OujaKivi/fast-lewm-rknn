#!/usr/bin/env python3
"""Probe batched SmolVLA action-chunk samples from one LIBERO observation."""

import argparse
import json
import time

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import add_envs_task, preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

from eval_smolvla_libero import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = args.device
    env_config = LiberoEnv(
        task="libero_spatial", task_ids=[0], observation_height=256, observation_width=256
    )
    env = make_env(env_config, n_envs=1, use_async_envs=False)["libero_spatial"][0]
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_config, config)
    policy, config = load_policy(args.model_path, args.device, args.vlm_path)
    try:
        observation, _ = env.reset(seed=[args.seed])
        batch = preprocessor(env_preprocessor(add_envs_task(env, preprocess_observation(observation))))
        with torch.inference_mode():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]
            torch.cuda.synchronize()
            start = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=config.use_cache,
                fill_kv_cache=True,
            )
            torch.cuda.synchronize()
            prefix_ms = (time.perf_counter() - start) * 1000
        output = []
        for size in args.batches:
            repeated = {
                key: value.repeat((size,) + (1,) * (value.ndim - 1))
                if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            noise = torch.randn(
                size, config.chunk_size, config.max_action_dim, generator=generator
            ).to(args.device)
            masks = prefix_pad_masks.repeat(size, 1)
            cache = {
                layer: {
                    name: value.repeat((size,) + (1,) * (value.ndim - 1))
                    for name, value in entries.items()
                }
                for layer, entries in past_key_values.items()
            }

            def shared_denoise():
                x_t = noise
                dt = -1.0 / config.num_steps
                for step in range(config.num_steps):
                    timestep = torch.full(
                        (size,), 1.0 + step * dt, dtype=torch.float32, device=args.device
                    )
                    x_t = x_t + dt * policy.model.denoise_step(
                        prefix_pad_masks=masks,
                        past_key_values=cache,
                        x_t=x_t,
                        timestep=timestep,
                    )
                return x_t

            try:
                with torch.inference_mode():
                    policy.predict_action_chunk(repeated, noise=noise)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        actions = policy.predict_action_chunk(repeated, noise=noise)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                    shared_denoise()
                    torch.cuda.synchronize()
                    shared_times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        shared_actions = shared_denoise()
                        torch.cuda.synchronize()
                        shared_times.append((time.perf_counter() - start) * 1000)
                parity_max_abs = (
                    actions - shared_actions[:, :, : actions.shape[-1]]
                ).abs().max().item()
                first = actions[:, 0, :].float().cpu()
                distances = torch.cdist(first, first)
                upper = distances[torch.triu_indices(size, size, offset=1).unbind()]
                output.append({
                    "batch": size,
                    "median_ms": sorted(times)[1],
                    "ms_per_candidate": sorted(times)[1] / size,
                    "shared_prefix_ms": prefix_ms,
                    "shared_denoise_ms": sorted(shared_times)[1],
                    "shared_total_ms": prefix_ms + sorted(shared_times)[1],
                    "shared_parity_max_abs": parity_max_abs,
                    "first_action_pairwise_l2_mean": upper.mean().item() if upper.numel() else 0.0,
                    "first_action_pairwise_l2_max": upper.max().item() if upper.numel() else 0.0,
                })
            except torch.cuda.OutOfMemoryError:
                output.append({"batch": size, "error": "CUDA OOM"})
                torch.cuda.empty_cache()
        print(json.dumps(output, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
