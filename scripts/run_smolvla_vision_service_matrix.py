#!/usr/bin/env python3
"""Run matched native vision plans on manifest-verified actual camera pairs."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cases(fixtures, model_path, output_dir):
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(model_path)
    keys = list(config.image_features)
    if keys != ["observation.images.image", "observation.images.image2"]:
        raise ValueError(f"Unexpected camera order: {keys}")
    if tuple(config.resize_imgs_with_padding) != (512, 512):
        raise ValueError("Existing vision graphs require checkpoint 512x512 preprocessing")
    manifest = json.loads((fixtures / "manifest.json").read_text())
    cases = []
    seen = set()
    for case in manifest["cases"]:
        name = case["file"]
        if Path(name).name != name or name in seen:
            raise ValueError("Unsafe or duplicate fixture name")
        seen.add(name)
        path = fixtures / name
        if sha256(path) != case["sha256"]:
            raise ValueError(f"Fixture hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as fixture:
            batch = {key: torch.from_numpy(fixture[key].copy()) for key in keys}
        images, masks = SmolVLAPolicy.prepare_images(SimpleNamespace(config=config), batch)
        if len(images) != 2 or not all(mask.all().item() for mask in masks):
            raise ValueError("Exactly two valid cameras required")
        paths = []
        for index, image in enumerate(images):
            values = np.ascontiguousarray(image.numpy(), dtype=np.float32)
            if values.shape != (1, 3, 512, 512) or not np.isfinite(values).all():
                raise ValueError("Invalid checkpoint-prepared image")
            destination = output_dir / f"{path.stem}_camera{index}_fp32.bin"
            values.tofile(destination)
            paths.append(destination)
        cases.append((name, paths))
    if not cases:
        raise ValueError("No fixtures")
    return manifest, cases, {
        "camera_order": keys,
        "config_sha256": sha256(Path(model_path) / "config.json"),
        "prepare_images_source_sha256": hashlib.sha256(inspect.getsource(SmolVLAPolicy.prepare_images).encode()).hexdigest(),
        "resize_imgs_with_padding": list(config.resize_imgs_with_padding),
        "timing_excludes_checkpoint_resize_normalize": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--sampler", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "session.json"
    if report_path.exists():
        parser.error("Refusing to overwrite an existing session; use a new output directory")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest, cases, preprocessing = prepare_cases(args.fixtures, args.model_path, args.output_dir)
    graphs = [args.model_dir / name for name in ("vision_unmasked.rknn", "stem_native.rknn", "tail_native.rknn",
                                                "attention_full.rknn", "attention_h4.rknn")]
    graphs += [args.model_dir / f"layer{layer:02d}_{kind}.rknn" for layer in range(12) for kind in ("qkv", "suffix")]
    report = {
        "scope": "Actual demonstration camera service matrix; no policy rollout, task quality, or edge-cloud acceleration claim",
        "status": "running", "fixtures": manifest, "preprocessing": preprocessing,
        "repeats": args.repeats, "affinity": sorted(os.sched_getaffinity(0)),
        "script_sha256": sha256(Path(__file__)), "binary_sha256": sha256(args.binary),
        "artifact_sha256": {str(path): sha256(path) for path in graphs}, "cases": {},
        "limitations": ["One episode, three recorded frames, not live cameras or success rates",
                        "All fused/packed contexts coexist, not isolated memory/energy",
                        "Split/fused rounding differences require separate action/quality validation",
                        "Resident and host-I/O modes are separate calls; their median difference is not exact I/O cost",
                        "Eight repeats do not establish stable p95/p99; stages are separate three-repeat diagnostics"],
    }
    try:
        for name, paths in cases:
            raw = args.output_dir / f"{Path(name).stem}.json"
            command = [str(args.binary.resolve()), str(args.model_dir.resolve()), str(args.repeats), str(raw.resolve()),
                       *[str(path.resolve()) for path in paths]]
            if args.sampler:
                command = [sys.executable, str(args.sampler), "--output", str(raw.with_suffix(".load.json")),
                           "--system-telemetry", "--stdout-log", str(raw.with_suffix(".log")), "--quiet", "--", *command]
            print(f"Running actual pair {name}", flush=True)
            subprocess.run(command, check=True)
            result = json.loads(raw.read_text())
            if not result["schedule_swap_restore_and_post_timing_bitwise_checks"]:
                raise ValueError("Missing state checks")
            result["prepared_camera_sha256"] = {path.name: sha256(path) for path in paths}
            result["connector_sha256"] = {path.name: sha256(path) for path in args.output_dir.glob(raw.name + ".*.bin")}
            report["cases"][name] = result
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed {len(cases)} actual camera pairs: {report_path}", flush=True)


if __name__ == "__main__":
    main()
