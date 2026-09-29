#!/usr/bin/env python3
"""Compile all exported native-vision stages, retaining per-model build logs."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    converter = Path(__file__).with_name("convert_smolvla_denoise_rknn.py")
    for source in sorted(args.directory.glob("*.onnx")):
        target = source.with_suffix(".rknn")
        with source.with_suffix(".build.log").open("w") as log:
            result = subprocess.run([sys.executable, str(converter), "--onnx", str(source),
                                     "--output", str(target)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Conversion failed: {source}; see {source.with_suffix('.build.log')}")
        print(target.name, flush=True)


if __name__ == "__main__":
    main()
