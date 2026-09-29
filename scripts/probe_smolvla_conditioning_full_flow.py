#!/usr/bin/env python3
"""Full dependent ten-step baseline audit for hoisted consumer-ready K/V."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def compare(actual, expected):
    pairs = list(zip(actual, expected, strict=True))
    if not pairs or any(a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32 or
                        not np.isfinite(a).all() or not np.isfinite(b).all() for a, b in pairs):
        raise ValueError("Invalid parity operands")
    return {"max_abs": max(float(np.max(np.abs(a - b))) for a, b in pairs),
            "bitwise_equal": all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in pairs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--original-rknn", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval().float()
    if config.num_steps != 10 or config.chunk_size != 50 or config.max_action_dim != 32:
        raise ValueError("Expected unchanged LIBERO flow protocol")
    library = ctypes.CDLL(str(args.library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    timing_array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.full_conditioning_error.restype = ctypes.c_char_p
    library.full_conditioning_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.full_conditioning_create.restype = ctypes.c_void_p
    library.full_conditioning_destroy.argtypes = [ctypes.c_void_p]
    library.full_conditioning_prefix.argtypes = [ctypes.c_void_p, array, ctypes.c_size_t]
    library.full_conditioning_begin.argtypes = [ctypes.c_void_p]
    library.full_conditioning_step.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_size_t, array, ctypes.c_size_t, timing_array]
    library.full_conditioning_info.argtypes = [ctypes.c_void_p]
    library.full_conditioning_info.restype = ctypes.c_char_p
    handle = library.full_conditioning_create(str(args.original_rknn).encode(), str(args.directory).encode())
    if not handle:
        raise RuntimeError(library.full_conditioning_error().decode())

    def check(code):
        if code < 0:
            raise RuntimeError(library.full_conditioning_error().decode())

    names = ("original_resident", "original_native_passthrough", "hoisted_consumer_ready")
    cache = [np.load(args.data_dir / f"{kind}_{layer}.npy") for layer in range(32) for kind in ("key", "value")]
    if any(value.shape != (1, 149, 5, 64) or value.dtype != np.float32 or not np.isfinite(value).all() for value in cache):
        raise ValueError("Expected original canonical prefix fixtures")
    prefix = np.concatenate([value.reshape(-1) for value in cache])
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(812), dtype=torch.float32)

    def set_prefix(values):
        start = time.perf_counter()
        check(library.full_conditioning_prefix(handle, values, values.size))
        return (time.perf_counter() - start) * 1000

    def step(mode, suffix):
        velocity = np.empty((1, 50, 32), np.float32)
        times = np.zeros(4, np.float64)
        check(library.full_conditioning_step(handle, mode, suffix, suffix.size, velocity, velocity.size, times))
        return velocity, times

    def trajectory(mode, capture=False):
        check(library.full_conditioning_begin(handle))
        x_t = noise.clone()
        snapshots = []
        totals = np.zeros(6, np.float64)
        start = time.perf_counter()
        for index in range(10):
            embed_start = time.perf_counter()
            suffix, _, _ = policy.model.embed_suffix(x_t, torch.tensor(1.0 - index / 10, dtype=torch.float32).expand(1))
            suffix = np.ascontiguousarray(suffix.numpy(), dtype=np.float32)
            totals[0] += (time.perf_counter() - embed_start) * 1000
            velocity, times = step(mode, suffix)
            totals[1:5] += times
            solver_start = time.perf_counter()
            x_t = x_t + (-1.0 / 10) * torch.from_numpy(velocity)
            totals[5] += (time.perf_counter() - solver_start) * 1000
            if x_t.dtype != torch.float32 or not torch.isfinite(x_t).all():
                raise ValueError("Invalid original FP32 solver state")
            if capture:
                snapshots.extend((velocity.copy(), x_t.numpy().copy()))
        return x_t.numpy().copy(), (time.perf_counter() - start) * 1000, totals, snapshots

    report = {"scope": "Complete 32-layer action network, ten dependent steps with unchanged CPU FP32 embedding/Euler, real checkpoint; synthetic fixed prefix; excludes vision/prefill/context initialization, not closed-loop task evaluation",
              "protocol": "All target heads/actions/steps/condition retained, original FP16 graph precision; padding removed before attention; one-time NPU preparation included in hoisted trajectory",
              "stage_order": ["suffix_embedding", "once_per_observation_npu_preparation", "suffix_pack_sync_and_optional_inputs_set", "run", "velocity_retrieval", "fp32_solver"],
              "repeats": args.repeats, "affinity": sorted(os.sched_getaffinity(0)), "steps": 10,
              "limitations": ["Four contexts coexist; reported per-graph memory is not isolated peak RSS",
                              "No continuous CPU frequency or thermal lock; plans rotated",
                              "Only one synthetic prefix plus an invalidation variant; no task success/energy claim",
                              "Common one-time raw prefix packing is reported separately; cached NPU preparation is charged inside every hoisted flow"],
              "plans": {}}
    try:
        info = library.full_conditioning_info(handle)
        if not info:
            raise RuntimeError(library.full_conditioning_error().decode())
        report["graphs"] = json.loads(info)
        with torch.inference_mode():
            report["common_prefix_pack_sync_ms"] = set_prefix(prefix)
            fixture = np.ascontiguousarray(np.load(args.data_dir / "suffix.npy"), dtype=np.float32)
            reference = step(0, fixture)[0]
            report["fixed_suffix_parity"] = {name: compare([step(mode, fixture)[0]], [reference]) for mode, name in enumerate(names)}
            from rknnlite.api import RKNNLite
            runtime = RKNNLite()
            try:
                check(runtime.load_rknn(str(args.original_rknn)))
                check(runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2))
                lite = runtime.inference(inputs=[fixture, *[value.transpose(0, 2, 3, 1).copy() for value in cache]], data_format=None)[0]
                report["original_native_vs_lite"] = compare([reference], [lite])
                if not report["original_native_vs_lite"]["bitwise_equal"]:
                    raise ValueError("Original native control differs from existing deployment")
            finally:
                runtime.release()
            references = [trajectory(mode, capture=True)[3] for mode in range(3)]
            report["per_step_velocity_latent_parity"] = {name: compare(values, references[0]) for name, values in zip(names, references)}
            if not report["per_step_velocity_latent_parity"][names[1]]["bitwise_equal"]:
                raise ValueError("Original graph controls differ")
            changed = np.concatenate([(value if index % 2 == 0 else -np.flip(value, axis=2).copy()).reshape(-1)
                                      for index, value in enumerate(cache)])
            set_prefix(changed)
            changed_references = [trajectory(mode, capture=True)[3] for mode in range(3)]
            report["changed_prefix_parity"] = {name: compare(values, changed_references[0]) for name, values in zip(names, changed_references)}
            report["changed_prefix_final_difference"] = compare([changed_references[0][-1]], [references[0][-1]])
            if report["changed_prefix_final_difference"]["max_abs"] == 0:
                raise ValueError("Changed prefix did not change the actions")
            set_prefix(prefix)
            report["restored_prefix_parity"] = {name: compare(trajectory(mode, capture=True)[3], references[0])
                                                for mode, name in enumerate(names)}
            for mode in range(3):
                trajectory(mode)
            totals = [[] for _ in names]
            stages = [[] for _ in names]
            print("PROBE_MEASURE_START", flush=True)
            for iteration in range(args.repeats):
                for offset in range(3):
                    mode = (iteration + offset) % 3
                    print(f"PROBE_PLAN_START {names[mode]}", flush=True)
                    final, duration, stage_times, _ = trajectory(mode)
                    print("PROBE_PLAN_END", flush=True)
                    if not compare([final], [references[mode][-1]])["bitwise_equal"]:
                        raise ValueError(f"Nondeterministic trajectory: {names[mode]}")
                    totals[mode].append(duration)
                    stages[mode].append(stage_times.tolist())
            print("PROBE_MEASURE_END", flush=True)
            for name, samples, stage_values in zip(names, totals, stages):
                report["plans"][name] = {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                                          "samples_ms": samples, "stage_median_totals_ms": np.median(stage_values, axis=0).tolist(),
                                          "stage_samples_totals_ms": stage_values}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"plans": {name: values["median_ms"] for name, values in report["plans"].items()},
                          "parity": report["per_step_velocity_latent_parity"]}, indent=2))
    finally:
        library.full_conditioning_destroy(handle)


if __name__ == "__main__":
    main()
