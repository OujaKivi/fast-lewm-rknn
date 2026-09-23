#!/usr/bin/env python3
"""Inspect SmolVLA's cached denoising step before NPU partitioning."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class CachedDenoiseStep(torch.nn.Module):
    def __init__(self, model, prefix_pad_mask, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        batch_size, prefix_len = prefix_pad_mask.shape
        suffix_len = model.config.chunk_size
        prefix_attention = prefix_pad_mask[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_attention = torch.ones(
            batch_size, suffix_len, suffix_len, dtype=torch.bool
        ).tril()
        self.register_buffer(
            "attention_mask",
            torch.cat([prefix_attention, suffix_attention], dim=2),
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            prefix_pad_mask.sum(dim=1, keepdim=True) + torch.arange(suffix_len)[None, :],
            persistent=False,
        )

    def forward(self, suffix_embs, *flat_cache):
        cache = {
            index: {
                "key_states": flat_cache[2 * index],
                "value_states": flat_cache[2 * index + 1],
            }
            for index in range(self.num_layers)
        }
        outputs, _ = self.model.vlm_with_expert.forward(
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            past_key_values=cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs[1][:, -self.model.config.chunk_size :].float()
        return self.model.action_out_proj(suffix_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.layers <= 16:
        parser.error("--layers must be between 1 and 16")

    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval()
    image_key = next(iter(config.image_features))
    image = torch.linspace(0, 1, 256).repeat(1, 3, 256, 1)
    tokens = policy.model.vlm_with_expert.processor.tokenizer(
        "Pick up the object.", return_tensors="pt", padding=True
    )
    batch = {
        image_key: image,
        OBS_STATE: torch.zeros(1, config.input_features[OBS_STATE].shape[0]),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"],
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].bool(),
    }
    model = policy.model
    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        prefix, pad_mask, att_mask = model.embed_prefix(
            images, img_masks, tokens["input_ids"], tokens["attention_mask"].bool(), state
        )
        prefix_att = make_att_2d_masks(pad_mask, att_mask)
        positions = torch.cumsum(pad_mask, dim=1) - 1
        _, cache = model.vlm_with_expert.forward(
            attention_mask=prefix_att,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        torch.manual_seed(42)
        x_t = torch.randn(1, config.chunk_size, config.max_action_dim)
        timestep = torch.ones(1)
        suffix, _, _ = model.embed_suffix(x_t, timestep)
        output = model.denoise_step(pad_mask, cache, x_t, timestep)
    details = {
        "prefix_shape": list(prefix.shape),
        "prefix_valid_tokens": int(pad_mask.sum()),
        "cache_layers": len(cache),
        "cache_shapes": {
            str(index): {name: list(tensor.shape) for name, tensor in layer.items()}
            for index, layer in cache.items()
        },
        "suffix_shape": list(suffix.shape),
        "denoise_output_shape": list(output.shape),
    }
    print(json.dumps(details, indent=2))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.float()
        model.vlm_with_expert.num_vlm_layers = args.layers
        with torch.inference_mode():
            suffix, _, _ = model.embed_suffix(x_t, timestep)
        flat_cache = tuple(
            tensor.float().clone()
            for index in range(args.layers)
            for tensor in (cache[index]["key_states"], cache[index]["value_states"])
        )
        wrapper = CachedDenoiseStep(model, pad_mask.clone(), args.layers).eval()
        inputs = (suffix.clone(), *flat_cache)
        with torch.inference_mode():
            converted_output = wrapper(*inputs)
            cast_cache = {
                index: {
                    "key_states": flat_cache[2 * index],
                    "value_states": flat_cache[2 * index + 1],
                }
                for index in range(args.layers)
            }
            baseline_converted = model.denoise_step(pad_mask, cast_cache, x_t, timestep)
        rewrite_error = (converted_output - baseline_converted).abs().max().item()
        if rewrite_error > 1e-5:
            raise RuntimeError(f"Static-mask rewrite changed denoise output: {rewrite_error}")
        conversion_error = (converted_output - output).abs().max().item()
        input_names = ["suffix"] + [
            f"{name}_{index}"
            for index in range(args.layers)
            for name in ("key", "value")
        ]
        np.save(output_dir / "reference.npy", converted_output.numpy())
        np.save(output_dir / "original_reference.npy", output.numpy())
        for name, tensor in zip(input_names, inputs, strict=True):
            np.save(output_dir / f"{name}.npy", tensor.numpy())
        torch.onnx.export(
            wrapper,
            inputs,
            output_dir / "denoise_step.onnx",
            input_names=input_names,
            output_names=["velocity"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print(
            f"exported {output_dir / 'denoise_step.onnx'} "
            f"cache_cast_error={conversion_error} mask_rewrite_error={rewrite_error}"
        )


if __name__ == "__main__":
    main()
