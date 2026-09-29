#!/usr/bin/env python3
"""Profile complete policy replay against deployed and resident flow controls."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from rknnlite.api import RKNNLite

from probe_smolvla_conditioning_full_flow import compare


NAMES = ("deployed_lite", "original_resident", "hoisted_consumer_ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--vision-rknn", type=Path, required=True)
    parser.add_argument("--vision-library", type=Path)
    parser.add_argument("--prefill-rknn", type=Path, required=True)
    parser.add_argument("--denoise-rknn", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--conditioning-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--capture-boundary-dir", type=Path)
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
    if (config.num_steps, config.chunk_size, config.max_action_dim) != (10, 50, 32):
        raise ValueError("Expected unchanged LIBERO flow")
    library = ctypes.CDLL(str(args.library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    timings = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.full_conditioning_error.restype = ctypes.c_char_p
    library.full_conditioning_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.full_conditioning_create.restype = ctypes.c_void_p
    library.full_conditioning_destroy.argtypes = [ctypes.c_void_p]
    library.full_conditioning_prefix.argtypes = [ctypes.c_void_p, array, ctypes.c_size_t]
    library.full_conditioning_begin.argtypes = [ctypes.c_void_p]
    library.full_conditioning_step.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_size_t, array, ctypes.c_size_t, timings]
    handle = None
    pair_handle = None
    runtimes = []

    def check(code):
        if code < 0:
            raise RuntimeError(library.full_conditioning_error().decode())

    def load(path):
        runtime = RKNNLite()
        runtimes.append(runtime)
        if runtime.load_rknn(str(path)) != 0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError(f"Cannot initialize {path}")
        return runtime

    report = {"scope": "Complete two-camera policy replay including vision, prefill, 32 action layers, ten dependent CPU FP32 Euler steps; no C17 integration or closed-loop success claim",
              "fixtures": json.loads((args.fixtures / "manifest.json").read_text()),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "affinity": sorted(os.sched_getaffinity(0)), "repeats": args.repeats,
              "vision_control": "native fused two-camera mask1/mask2 concurrent" if args.vision_library else "deployed Lite serial cameras mask7",
              "limitations": ["Seven/eight RKNN contexts coexist; no isolated memory/energy claim",
                              "Fixed valid 149-token prefix and one recorded task/episode",
                              "Checkpoint preprocessing done before measured policy; transport/simulator/postprocessing excluded",
                              "No cross-observation reuse: both images, prefill and cache preparation rerun for every call",
                              "C06 hoisting is a strong baseline, not the C17 proposed mechanism"],
              "artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  [args.vision_rknn, args.prefill_rknn, args.denoise_rknn, args.library,
                                   args.conditioning_dir / "prepare_full.rknn", args.conditioning_dir / "warm_full.rknn"]},
              "validation": {}, "cases": {}}
    current = None
    mode = 0
    snapshots = None
    packed = False
    original_sample = policy.model.sample_actions
    original_prefix = policy.model.embed_prefix
    pending_images = []
    boundary_inputs = None

    def embed_image(image):
        if args.vision_library:
            if not pending_images:
                raise ValueError("Missing paired camera embedding")
            current["vision_calls"] += 1
            return pending_images.pop(0)
        started = time.perf_counter()
        output = vision.inference(inputs=[image.detach().numpy()], data_format=["nchw"])[0]
        result = torch.from_numpy(output).to(dtype=image.dtype)
        current["vision_ms"] += 1000 * (time.perf_counter() - started)
        current["vision_calls"] += 1
        return result

    def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
        if args.vision_library:
            if len(images) != 2 or pending_images:
                raise ValueError("Expected exactly two new cameras, no stale embedding")
            started = time.perf_counter()
            inputs = np.ascontiguousarray(torch.cat(images).numpy())
            output = np.empty((2, 64, 960), np.float32)
            times = np.zeros(3, np.float64)
            if pair_library.camera_pair_run(pair_handle, 1, inputs, inputs.size, output, output.size, times) < 0:
                raise RuntimeError(pair_library.camera_pair_error().decode())
            pending_images.extend(torch.from_numpy(output[index:index + 1].copy()) for index in range(2))
            current["vision_ms"] += 1000 * (time.perf_counter() - started)
        result = original_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        if pending_images:
            raise ValueError("Incomplete paired embedding consumption")
        return result

    def prefill(*forward_args, **kwargs):
        if not kwargs.get("fill_kv_cache"):
            raise ValueError("Action steps must use the compiled complete action network")
        prefix = kwargs["inputs_embeds"][0]
        if prefix.shape != (1, 149, 960) or not torch.equal(kwargs["position_ids"], torch.arange(149)[None]):
            raise ValueError("Compiled prefill requires 149 valid tokens")
        started = time.perf_counter()
        outputs = prefill_runtime.inference(inputs=[prefix.detach().float().numpy()], data_format=None)
        if len(outputs) != 64 or any(value.shape != (1, 149, 5, 64) for value in outputs):
            raise ValueError("Unexpected prefill cache geometry")
        cache = {layer: {"key_states": torch.from_numpy(outputs[2 * layer].copy()),
                         "value_states": torch.from_numpy(outputs[2 * layer + 1].copy())} for layer in range(32)}
        if boundary_inputs is not None:
            for layer in (0, 1):
                for index, kind in enumerate(("key", "value")):
                    boundary_inputs[f"{kind}_{layer}"] = outputs[2 * layer + index].copy()
        current["prefill_ms"] += 1000 * (time.perf_counter() - started)
        return [None, None], cache

    def denoise(prefix_pad_masks, past_key_values, x_t, timestep):
        nonlocal packed
        if prefix_pad_masks.shape != (1, 149) or not prefix_pad_masks.all():
            raise ValueError("Invalid denoising prefix mask")
        started = time.perf_counter()
        suffix, _, _ = policy.model.embed_suffix(x_t, timestep)
        suffix = np.ascontiguousarray(suffix.numpy())
        if boundary_inputs is not None:
            boundary_inputs["suffix"].append(suffix.copy())
        if mode == 0:
            inputs = [suffix]
            for layer in range(32):
                for kind in ("key_states", "value_states"):
                    inputs.append(past_key_values[layer][kind].numpy().transpose(0, 2, 3, 1).copy())
            output = denoise_runtime.inference(inputs=inputs, data_format=None)[0].copy()
        else:
            if not packed:
                pack_start = time.perf_counter()
                prefix = np.concatenate([past_key_values[layer][kind].numpy().reshape(-1)
                                         for layer in range(32) for kind in ("key_states", "value_states")])
                check(library.full_conditioning_prefix(handle, prefix, prefix.size))
                check(library.full_conditioning_begin(handle))
                current["common_prefix_pack_ms"] += 1000 * (time.perf_counter() - pack_start)
                packed = True
            output = np.empty((1, 50, 32), np.float32)
            times = np.zeros(4, np.float64)
            check(library.full_conditioning_step(handle, 0 if mode == 1 else 2, suffix, suffix.size, output, output.size, times))
            current["once_preparation_ms"] += times[0]
            current["native_run_ms"] += times[2]
        current["denoise_ms"] += 1000 * (time.perf_counter() - started)
        current["denoise_steps"] += 1
        if snapshots is not None:
            snapshots.extend((x_t.numpy().copy(), output.copy()))
        return torch.from_numpy(output)

    def sample(*sample_args, **kwargs):
        output = original_sample(*sample_args, **kwargs)
        if snapshots is not None:
            snapshots.append(output.numpy().copy())
        return output

    def invoke(selected, fixture, capture=False, case_name=None):
        nonlocal mode, current, snapshots, packed, boundary_inputs
        mode, packed = selected, False
        snapshots = [] if capture else None
        boundary_inputs = {"suffix": []} if capture and selected == 0 and args.capture_boundary_dir and case_name else None
        current = {key: 0.0 for key in ("vision_ms", "prefill_ms", "denoise_ms", "common_prefix_pack_ms",
                                       "once_preparation_ms", "native_run_ms")}
        current.update(vision_calls=0, denoise_steps=0)
        policy.reset()
        started = time.perf_counter()
        action = policy.predict_action_chunk({key: value.clone() for key, value in fixture.items() if key != "noise"}, noise=fixture["noise"].clone())
        current["total_ms"] = 1000 * (time.perf_counter() - started)
        current["other_ms"] = current["total_ms"] - current["vision_ms"] - current["prefill_ms"] - current["denoise_ms"]
        if current["vision_calls"] != 2 or current["denoise_steps"] != 10 or not torch.isfinite(action).all():
            raise ValueError("Incomplete or invalid policy")
        if boundary_inputs is not None:
            args.capture_boundary_dir.mkdir(parents=True, exist_ok=True)
            path = args.capture_boundary_dir / f"{Path(case_name).stem}_inputs.npz"
            np.savez(path, **{key: np.stack(value) if key == "suffix" else value for key, value in boundary_inputs.items()})
            report.setdefault("recorded_boundary_inputs", {})[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return action.numpy().copy(), current.copy(), snapshots

    try:
        handle = library.full_conditioning_create(str(args.denoise_rknn).encode(), str(args.conditioning_dir).encode())
        if not handle:
            raise RuntimeError(library.full_conditioning_error().decode())
        if args.vision_library:
            pair_library = ctypes.CDLL(str(args.vision_library))
            pair_library.camera_pair_create.argtypes = [ctypes.c_char_p]
            pair_library.camera_pair_create.restype = ctypes.c_void_p
            pair_library.camera_pair_destroy.argtypes = [ctypes.c_void_p]
            pair_library.camera_pair_error.restype = ctypes.c_char_p
            pair_library.camera_pair_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_size_t, array, ctypes.c_size_t, timings]
            pair_handle = pair_library.camera_pair_create(str(args.vision_rknn).encode())
            if not pair_handle:
                raise RuntimeError(pair_library.camera_pair_error().decode())
            report["artifact_sha256"][str(args.vision_library)] = hashlib.sha256(args.vision_library.read_bytes()).hexdigest()
            prefill_runtime, denoise_runtime = [load(path) for path in (args.prefill_rknn, args.denoise_rknn)]
        else:
            vision, prefill_runtime, denoise_runtime = [load(path) for path in (args.vision_rknn, args.prefill_rknn, args.denoise_rknn)]
        policy.model.vlm_with_expert.embed_image = embed_image
        policy.model.embed_prefix = embed_prefix
        policy.model.vlm_with_expert.forward = prefill
        policy.model.denoise_step = denoise
        policy.model.sample_actions = sample
        fixtures = {path.name: {key: torch.from_numpy(value.copy()) for key, value in np.load(path).items()}
                    for path in sorted(args.fixtures.glob("frame_*.npz"))}
        with torch.inference_mode():
            anchors = {}
            for name, fixture in fixtures.items():
                if args.vision_library:
                    images, _ = policy.prepare_images(fixture)
                    inputs = np.ascontiguousarray(torch.cat(images).numpy())
                    results = []
                    for parallel, swapped in ((0, False), (1, False), (1, True), (1, False)):
                        source = np.ascontiguousarray(inputs[::-1]) if swapped else inputs
                        output = np.empty((2, 64, 960), np.float32)
                        times = np.zeros(3, np.float64)
                        if pair_library.camera_pair_run(pair_handle, parallel, source, source.size, output, output.size, times) < 0:
                            raise RuntimeError(pair_library.camera_pair_error().decode())
                        results.append(output[::-1].copy() if swapped else output.copy())
                    parity = compare(results, [results[0]] * len(results))
                    report.setdefault("camera_schedule_parity", {})[name] = parity
                    if not parity["bitwise_equal"]:
                        raise ValueError("Concurrent camera schedule/swap/restore changes original fused bits")
                captures = [invoke(selected, fixture, capture=True, case_name=name) for selected in range(3)]
                anchors[name] = [value[0] for value in captures]
                report["validation"][name] = {plan: compare(value[2], captures[0][2]) for plan, value in zip(NAMES, captures)}
                if not all(result["bitwise_equal"] for result in report["validation"][name].values()):
                    raise ValueError("Native baseline changes dependent velocities/latents")
            first = next(iter(fixtures))
            report["restored_observation"] = compare(invoke(2, fixtures[first], capture=True)[2],
                                                       invoke(0, fixtures[first], capture=True)[2])
            for selected in range(3):
                invoke(selected, fixtures[first])
            records = {name: {plan: [] for plan in NAMES} for name in fixtures}
            print("PROBE_MEASURE_START", flush=True)
            for iteration in range(args.repeats):
                for name, fixture in fixtures.items():
                    for offset in range(3):
                        selected = (iteration + offset) % 3
                        print(f"PROBE_PLAN_START {name}_{NAMES[selected]}", flush=True)
                        action, times, _ = invoke(selected, fixture)
                        print("PROBE_PLAN_END", flush=True)
                        if not compare([action], [anchors[name][selected]])["bitwise_equal"]:
                            raise ValueError("Nondeterministic replay")
                        records[name][NAMES[selected]].append(times)
            print("PROBE_MEASURE_END", flush=True)
            for name, plans in records.items():
                report["cases"][name] = {plan: {"samples": values,
                                                   "median_ms": {key: float(np.median([value[key] for value in values])) for key in values[0]},
                                                   "p95_total_ms": float(np.percentile([value["total_ms"] for value in values], 95))}
                                             for plan, values in plans.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"validation": report["validation"], "medians": {
            name: {plan: value["median_ms"] for plan, value in plans.items()} for name, plans in report["cases"].items()}}, indent=2))
    finally:
        for runtime in runtimes:
            runtime.release()
        if handle:
            library.full_conditioning_destroy(handle)
        if pair_handle:
            pair_library.camera_pair_destroy(pair_handle)


if __name__ == "__main__":
    main()
