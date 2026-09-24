#!/usr/bin/env python3
"""Export a fixed-shape SmolVLA prefill that produces the denoiser K/V cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class PrefillCache(torch.nn.Module):
    def __init__(self, model, pad_mask, att_mask, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.register_buffer("attention_mask", make_att_2d_masks(pad_mask, att_mask), persistent=False)
        self.register_buffer("position_ids", torch.cumsum(pad_mask, dim=1) - 1, persistent=False)

    def forward(self, prefix):
        _, cache = self.model.vlm_with_expert.forward(
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return tuple(
            tensor
            for layer in range(self.layers)
            for tensor in (cache[layer]["key_states"], cache[layer]["value_states"])
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--prompt", default="Pick up the object.")
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--invert-image", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.layers <= 32:
        parser.error("--layers must be between 1 and 32")

    torch.set_num_threads(4)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval()
    image_key = next(iter(config.image_features))
    side = config.image_features[image_key].shape[-1]
    axis = torch.linspace(0, 1, side)
    image = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0)
    image = torch.cat([image, (image[:1] + image[1:]) / 2], dim=0)
    if args.invert_image:
        image = 1 - image
    tokens = policy.model.vlm_with_expert.processor.tokenizer(
        args.prompt, return_tensors="pt", padding=True
    )
    image_keys = list(config.image_features) if args.all_images else [image_key]
    batch = {
        **{key: image.unsqueeze(0) for key in image_keys},
        OBS_STATE: torch.zeros(1, config.input_features[OBS_STATE].shape[0]),
        OBS_LANGUAGE_TOKENS: tokens["input_ids"],
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].bool(),
    }
    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        prefix, pad_mask, att_mask = policy.model.embed_prefix(
            images, img_masks, tokens["input_ids"], tokens["attention_mask"].bool(), state
        )
    if prefix.shape[0] != 1 or not bool(pad_mask.all()):
        raise RuntimeError(f"Unexpected prefill shape or mask: {tuple(prefix.shape)}")

    model = policy.model.float()
    model.vlm_with_expert.num_vlm_layers = args.layers
    wrapper = PrefillCache(model, pad_mask, att_mask, args.layers).eval()
    with torch.inference_mode():
        reference = wrapper(prefix.float())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "prefix.npy", prefix.float().numpy())
    output_names = [
        f"{name}_{layer}"
        for layer in range(args.layers)
        for name in ("key", "value")
    ]
    for name, tensor in zip(output_names, reference, strict=True):
        np.save(output_dir / f"{name}_reference.npy", tensor.float().numpy())
    if not args.reference_only:
        torch.onnx.export(
            wrapper,
            (prefix.float(),),
            output_dir / "prefill.onnx",
            input_names=["prefix"],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(json.dumps({
        "layers": args.layers,
        "prefix_shape": list(prefix.shape),
        "cache_shape": list(reference[0].shape),
        "cache_outputs": len(reference),
        "onnx": None if args.reference_only else str(output_dir / "prefill.onnx"),
    }, indent=2))


if __name__ == "__main__":
    main()
