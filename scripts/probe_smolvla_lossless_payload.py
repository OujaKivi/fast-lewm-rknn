#!/usr/bin/env python3
"""Count actual recorded pixels/native connector bytes using lossless codecs."""

import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
from PIL import Image
import torch
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad


def codec(value, encode, decode):
    times = []
    for _ in range(5):
        started = time.perf_counter()
        payload = encode(value)
        encoded_ms = 1000 * (time.perf_counter() - started)
        started = time.perf_counter()
        recovered = decode(payload)
        decoded_ms = 1000 * (time.perf_counter() - started)
        if not np.array_equal(recovered, value):
            raise ValueError("Codec is not lossless")
        times.append([encoded_ms, decoded_ms])
    return {"bytes": len(payload), "encode_decode_median_ms": np.median(times, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--vision-library", type=Path, required=True)
    parser.add_argument("--vision-rknn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {4, 5, 6, 7})
    torch.set_num_threads(4)
    library = ctypes.CDLL(str(args.vision_library))
    array = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    timing = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    library.camera_pair_create.argtypes = [ctypes.c_char_p]
    library.camera_pair_create.restype = ctypes.c_void_p
    library.camera_pair_run.argtypes = [ctypes.c_void_p, ctypes.c_uint, array, ctypes.c_size_t, array, ctypes.c_size_t, timing]
    library.camera_pair_destroy.argtypes = [ctypes.c_void_p]
    library.camera_pair_error.restype = ctypes.c_char_p
    handle = library.camera_pair_create(str(args.vision_rknn).encode())
    if not handle:
        raise RuntimeError(library.camera_pair_error().decode())
    report = {"scope": "Actual recorded observation/source-pixel and native NPU connector lossless payload sizes/CPU codec times; no cloud inference, network measurements, or branch-placement speed claim",
              "fixtures": json.loads((args.fixtures / "manifest.json").read_text()),
              "vision_sha256": hashlib.sha256(args.vision_rknn.read_bytes()).hexdigest(),
              "affinity": sorted(os.sched_getaffinity(0)),
              "cases": {}}

    def png(value):
        stream = io.BytesIO()
        Image.fromarray(value).save(stream, format="PNG", compress_level=6)
        return stream.getvalue()

    try:
        for path in sorted(args.fixtures.glob("frame_*.npz")):
            with np.load(path) as fixture:
                original = [fixture[name] for name in ("observation.images.image", "observation.images.image2")]
            images = torch.cat([resize_with_pad(torch.from_numpy(value), 512, 512, pad_value=0) * 2 - 1 for value in original])
            inputs = np.ascontiguousarray(images.numpy())
            outputs = np.empty((2, 64, 960), np.float32)
            times = np.zeros(3, np.float64)
            if library.camera_pair_run(handle, 1, inputs, inputs.size, outputs, outputs.size, times) < 0:
                raise RuntimeError(library.camera_pair_error().decode())
            cases = []
            for index, value in enumerate(original):
                pixels = np.rint(value[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                restored = pixels.transpose(2, 0, 1)[None].astype(np.float32) / 255
                if not np.array_equal(restored.view(np.uint32), value.view(np.uint32)):
                    raise ValueError("Raw uint8 cannot recover this observation's exact FP32 image bits")
                feature = np.ascontiguousarray(outputs[index])
                half = feature.astype(np.float16)
                half_exact = np.array_equal(half.astype(np.float32).view(np.uint32), feature.view(np.uint32))
                records = {"source_uint8_raw_bytes": pixels.nbytes, "source_fp32_raw_bytes": value.nbytes,
                           "png": codec(pixels, png, lambda payload: np.asarray(Image.open(io.BytesIO(payload))).copy()),
                           "source_zlib": codec(pixels, lambda x: zlib.compress(x.tobytes(), 6),
                                                lambda b: np.frombuffer(zlib.decompress(b), np.uint8).reshape(pixels.shape)),
                           "connector_fp32_raw_bytes": feature.nbytes,
                           "connector_fp32_zlib": codec(feature, lambda x: zlib.compress(x.tobytes(), 6),
                                                        lambda b: np.frombuffer(zlib.decompress(b), np.float32).reshape(feature.shape)),
                           "connector_fp16_roundtrip_bitwise_equal": half_exact}
                if half_exact:
                    records["connector_fp16_raw_bytes"] = half.nbytes
                    records["connector_fp16_zlib"] = codec(half, lambda x: zlib.compress(x.tobytes(), 6),
                                                            lambda b: np.frombuffer(zlib.decompress(b), np.float16).reshape(half.shape))
                cases.append(records)
            report["cases"][path.name] = cases
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["cases"], indent=2))
    finally:
        library.camera_pair_destroy(handle)


if __name__ == "__main__":
    main()
