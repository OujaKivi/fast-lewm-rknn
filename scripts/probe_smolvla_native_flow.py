#!/usr/bin/env python3
"""Compare unchanged ten-step denoising input lifetimes with a fixed prefix."""

import argparse
import ctypes
import json
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODES = ["legacy_c_api_prepacked", "native_rewrite_prefix",
         "native_resident_default_sync", "native_resident_dirty_sync",
         "legacy_fp16_prefix", "legacy_native_prefix_passthrough"]
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


def pointer(array):
    if array.dtype != np.float32 or not array.flags.c_contiguous:
        raise ValueError("Expected contiguous FP32 array")
    return array.ctypes.data_as(FLOAT_PTR)


class NativeFlow:
    def __init__(self, library, model):
        self.library = ctypes.CDLL(str(library))
        self.library.denoise_error.restype = ctypes.c_char_p
        self.library.denoise_create.argtypes = [ctypes.c_char_p]
        self.library.denoise_create.restype = ctypes.c_void_p
        self.library.denoise_destroy.argtypes = [ctypes.c_void_p]
        self.library.denoise_prepare.argtypes = [ctypes.c_void_p, FLOAT_PTR, ctypes.c_size_t, DOUBLE_PTR]
        self.library.denoise_step.argtypes = [ctypes.c_void_p, ctypes.c_uint, FLOAT_PTR, ctypes.c_size_t,
                                            FLOAT_PTR, ctypes.c_size_t, DOUBLE_PTR]
        self.handle = self.library.denoise_create(str(model).encode())
        if not self.handle:
            raise RuntimeError(self.library.denoise_error().decode())

    def check(self, code):
        if code < 0:
            raise RuntimeError(self.library.denoise_error().decode())

    def prepare(self, prefix):
        times = np.zeros(5, dtype=np.float64)
        self.check(self.library.denoise_prepare(self.handle, pointer(prefix), prefix.size,
                                               times.ctypes.data_as(DOUBLE_PTR)))
        return times

    def step(self, mode, suffix, shape):
        output = np.empty(shape, dtype=np.float32)
        times = np.zeros(3, dtype=np.float64)
        self.check(self.library.denoise_step(self.handle, mode, pointer(suffix), suffix.size,
                                            pointer(output), output.size, times.ctypes.data_as(DOUBLE_PTR)))
        return output, times

    def close(self):
        if self.handle:
            self.library.denoise_destroy(self.handle)
            self.handle = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rknn-model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = args.vlm_path
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=config).eval().float()
    if config.num_steps != 10 or config.chunk_size != 50 or config.max_action_dim != 32:
        raise ValueError("Expected unchanged ten-step LIBERO policy geometry")
    cache = [np.load(args.data_dir / f"{kind}_{layer}.npy")
             for layer in range(32) for kind in ("key", "value")]
    if any(value.shape != (1, 149, 5, 64) or value.dtype != np.float32 or
           not np.isfinite(value).all() for value in cache):
        raise ValueError("Expected finite FP32 prefix fixtures")

    def pack_prefix(values):
        return np.concatenate([value.transpose(0, 2, 3, 1).copy().reshape(-1) for value in values])

    prefix = pack_prefix(cache)
    native = NativeFlow(args.library, args.rknn_model)
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(812), dtype=torch.float32)
    dt = -1.0 / config.num_steps
    shape = (1, 50, 32)
    preparation = []
    samples = {name: [] for name in MODES}
    stage_samples = {name: [] for name in MODES}

    def trajectory(mode, capture=False):
        x_t = noise.clone()
        snapshots = []
        totals = np.zeros(5, dtype=np.float64)
        started = time.perf_counter()
        for step in range(config.num_steps):
            embed_start = time.perf_counter()
            timestep = torch.tensor(1.0 + step * dt, dtype=torch.float32).expand(1)
            suffix, _, _ = policy.model.embed_suffix(x_t, timestep)
            suffix = np.ascontiguousarray(suffix.detach().float().numpy())
            totals[0] += (time.perf_counter() - embed_start) * 1000
            velocity, device_times = native.step(mode, suffix, shape)
            totals[1:4] += device_times
            solver_start = time.perf_counter()
            x_t = x_t + dt * torch.from_numpy(velocity)
            totals[4] += (time.perf_counter() - solver_start) * 1000
            if x_t.dtype != torch.float32 or not torch.isfinite(x_t).all():
                raise ValueError("Invalid solver state")
            if capture:
                snapshots.append((velocity.copy(), x_t.numpy().copy()))
        milliseconds = (time.perf_counter() - started) * 1000
        return x_t.numpy().copy(), milliseconds, totals, snapshots

    def compare(actual, expected):
        maximum = 0.0
        for a, b in zip(actual, expected, strict=True):
            if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
                raise ValueError("Parity comparison requires matching FP32 shapes")
            maximum = max(maximum, float(np.max(np.abs(a - b))))
            if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
                raise ValueError(f"Bitwise parity failed: max_abs={maximum}")
        return maximum

    try:
        with torch.inference_mode():
            preparation.append(native.prepare(prefix))
            fixture_suffix = np.ascontiguousarray(np.load(args.data_dir / "suffix.npy"), dtype=np.float32)
            fixture_reference, _ = native.step(0, fixture_suffix, shape)
            # The C ABI control must match the existing Python deployment API.
            from rknnlite.api import RKNNLite
            runtime = RKNNLite()
            try:
                if runtime.load_rknn(str(args.rknn_model)) != 0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                    raise RuntimeError("RKNNLite initialization failed")
                lite_reference = runtime.inference(inputs=[fixture_suffix, *[
                    value.transpose(0, 2, 3, 1).copy() for value in cache]], data_format=None)[0]
                compare([fixture_reference], [lite_reference])
            finally:
                runtime.release()
            for mode in range(len(MODES)):
                actual, _ = native.step(mode, fixture_suffix, shape)
                compare([actual], [fixture_reference])

            _, _, _, reference = trajectory(0, capture=True)
            for mode in range(1, len(MODES)):
                _, _, _, actual = trajectory(mode, capture=True)
                compare([value for pair in actual for value in pair],
                        [value for pair in reference for value in pair])

            changed = pack_prefix([value if index % 2 == 0 else -np.flip(value, axis=2).copy()
                                   for index, value in enumerate(cache)])
            native.prepare(changed)
            _, _, _, changed_reference = trajectory(0, capture=True)
            changed_final_difference = float(np.max(np.abs(changed_reference[-1][1] - reference[-1][1])))
            if changed_final_difference == 0:
                raise ValueError("Prefix invalidation check did not change the trajectory")
            for mode in range(1, len(MODES)):
                _, _, _, actual = trajectory(mode, capture=True)
                compare([value for pair in actual for value in pair],
                        [value for pair in changed_reference for value in pair])
            native.prepare(prefix)
            restored, _, _, _ = trajectory(3)
            compare([restored], [reference[-1][1]])

            for mode in range(len(MODES)):
                trajectory(mode)
            print("PROBE_MEASURE_START", flush=True)
            for iteration in range(args.repeats):
                for offset in range(len(MODES)):
                    mode = (iteration + offset) % len(MODES)
                    print(f"PROBE_PLAN_START {MODES[mode]}", flush=True)
                    result, milliseconds, totals, _ = trajectory(mode)
                    print("PROBE_PLAN_END", flush=True)
                    compare([result], [reference[-1][1]])
                    samples[MODES[mode]].append(milliseconds)
                    stage_samples[MODES[mode]].append(totals.tolist())
            print("PROBE_MEASURE_END", flush=True)
            for _ in range(10):
                preparation.append(native.prepare(prefix))

        def summarize(values):
            return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                    "samples_ms": values}

        record = {
            "scope": "Ten dependent original-graph denoising steps, synthetic exported fixed prefix; excludes vision/prefill, initialization and one-time prefix preparation; not a rollout",
            "protocol": "Same checkpoint, graph, ten timesteps/noise; CPU FP32 suffix embedding and Euler updates; graph native FP16 unchanged",
            "rknn_model": str(args.rknn_model), "torch_version": torch.__version__,
            "repeats": args.repeats, "prefix_fp32_bytes": prefix.nbytes,
            "steps": config.num_steps, "rk_lite_c_api_fixture_max_abs": 0.0,
            "per_step_velocity_and_latent_max_abs": 0.0,
            "per_step_velocity_and_latent_bitwise_equal": True,
            "prefix_switch_restore_check": True,
            "changed_prefix_final_action_max_abs_vs_original": changed_final_difference,
            "static_preparation_scope": "From prepacked logical NHWC FP32 arrays; common initial NumPy repack excluded; CPU storage copy and each context's native packing/sync reported separately",
            "static_preparation": {name: summarize([float(value[index]) for value in preparation])
                                   for index, name in enumerate(("legacy_host_copy", "native_default_pack_sync", "native_dirty_pack_sync",
                                                                "legacy_fp16_conversion", "legacy_native_host_copy"))},
            "input_sync_policy": "mode 3 disables automatic input flush only; explicitly syncs every CPU-written suffix and all prefix buffers after every new prefix; no output flush disabled",
            "stage_order": ["suffix_embedding", "input_prepare_sync", "run", "output_retrieval", "fp32_solver"],
            "plans": {name: {"total": summarize(samples[name]),
                             "stage_median_totals_ms": np.median(stage_samples[name], axis=0).tolist(),
                             "stage_samples_totals_ms": stage_samples[name]} for name in MODES},
        }
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        for name in MODES:
            print(name, record["plans"][name]["total"]["median_ms"], flush=True)
    finally:
        native.close()


if __name__ == "__main__":
    main()
