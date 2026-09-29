#!/usr/bin/env python3
"""Capture three recorded LIBERO observations with checkpoint preprocessing."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=11)
    args = parser.parse_args()
    repo = "lerobot/libero_spatial_image"
    revision = "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"
    metadata = pq.read_table(hf_hub_download(repo, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset", revision=revision)).to_pylist()
    entry = next(row for row in metadata if row["episode_index"] == args.episode)
    data_path = f"data/chunk-{entry['data/chunk_index']:03d}/file-{entry['data/file_index']:03d}.parquet"
    local = hf_hub_download(repo, data_path, repo_type="dataset", revision=revision)
    table = pq.read_table(local)
    rows = table.to_pylist()
    episode = [row for row in rows if row["episode_index"] == args.episode]
    if len(episode) != entry["length"]:
        raise ValueError("Episode spans shards; capture requires a complete single-shard episode")
    tasks = pq.read_table(hf_hub_download(repo, "meta/tasks.parquet", repo_type="dataset", revision=revision)).to_pylist()
    task_names = {row["task_index"]: row["task"] for row in tasks}
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "tokenizer_processor": {"tokenizer_name": args.vlm_path}},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"repo": repo, "revision": revision, "data_path": data_path,
              "parquet_sha256": hashlib.sha256(Path(local).read_bytes()).hexdigest(),
              "scope": "Three offline recorded demonstration frames, not simulator rollout or task success evaluation",
              "camera_mapping": {"observation.images.wrist_image": "observation.images.image2"},
              "state_protocol": "Recorded eight-dimensional state passed unchanged into checkpoint normalization; not a claim of simulator state-semantic parity",
              "cases": []}
    for index in (0, len(episode) // 2, len(episode) - 1):
        row = episode[index]
        batch = {"observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
                 "task": task_names[row["task_index"]]}
        for source, target in (("observation.images.image", "observation.images.image"),
                               ("observation.images.wrist_image", "observation.images.image2")):
            image = np.asarray(Image.open(io.BytesIO(row[source]["bytes"])).convert("RGB")).copy()
            batch[target] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        processed = preprocessor(batch)
        values = {key: value.cpu().numpy() for key, value in processed.items() if isinstance(value, torch.Tensor)}
        if values["observation.language.tokens"].shape != (1, 20) or not values["observation.language.attention_mask"].all():
            raise ValueError("Recorded task must have exactly 20 valid tokens for the existing 149-token graph; no artificial padding")
        values["noise"] = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(1000 + index)).numpy()
        path = args.output_dir / f"frame_{index}.npz"
        np.savez(path, **values)
        report["cases"].append({"file": path.name, "frame_index": row["frame_index"],
                                "episode_index": row["episode_index"], "task": batch["task"],
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "shapes": {key: list(value.shape) for key, value in values.items()}})
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
