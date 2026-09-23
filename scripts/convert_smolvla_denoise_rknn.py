#!/usr/bin/env python3
"""Probe RKNN FP16 conversion of one cached SmolVLA denoising step."""

import argparse

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime = RKNN(verbose=False)
    if runtime.config(target_platform="rk3588", optimization_level=0) != 0:
        raise RuntimeError("RKNN config failed")
    if runtime.load_onnx(model=args.onnx) != 0:
        raise RuntimeError("RKNN ONNX load failed")
    if runtime.build(do_quantization=False) != 0:
        raise RuntimeError("RKNN build failed")
    if runtime.export_rknn(args.output) != 0:
        raise RuntimeError("RKNN export failed")
    runtime.release()
    print(args.output)


if __name__ == "__main__":
    main()
