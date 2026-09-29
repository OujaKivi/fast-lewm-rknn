#!/usr/bin/env python3
"""Hoist actual odd-layer K/V preparation, retaining every denoise operation."""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx.utils import Extractor

from extract_smolvla_conditioning_probe import dimensions, native_interface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    original_inputs = [value.name for value in source.graph.input]
    if original_inputs != ["suffix", *[f"{kind}_{layer}" for layer in range(32) for kind in ("key", "value")]]:
        raise ValueError("Expected the original 32-layer input order")
    shapes = {value.name: dimensions(value) for value in source.graph.value_info}
    ready = [f"/Transpose_{4 * layer + offset}_output_0" for layer in range(1, 32, 2) for offset in (1, 2)]
    for index, name in enumerate(ready):
        if shapes[name] != ([1, 15, 64, 149] if index % 2 == 0 else [1, 15, 149, 64]):
            raise ValueError(f"Unexpected layer-specific K/V geometry: {name}")
    raw = [f"{kind}_{layer}" for layer in range(1, 32, 2) for kind in ("key", "value")]
    warm_inputs = [name if index == 0 or ((index - 1) // 2) % 2 == 0 else ready[((index - 1) // 4) * 2 + (index - 1) % 2]
                   for index, name in enumerate(original_inputs)]
    models = {
        "prepare_full": Extractor(source).extract_model(raw, ready),
        "warm_full": Extractor(source).extract_model(warm_inputs, [value.name for value in source.graph.output]),
    }
    report = {"scope": "Ordinary exact-dependency hoisting plus expanded consumer-ready K/V; engineering baseline, not a new algorithm",
              "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
              "ready_source_tensors": ready, "warm_source_input_order": warm_inputs, "models": {}}
    for name, model in models.items():
        native = native_interface(model, ready)
        onnx.save(native, args.output_dir / (name + ".onnx"))
        report["models"][name] = {"nodes": len(native.graph.node),
                                  "initializer_bytes": sum(len(value.raw_data) for value in native.graph.initializer),
                                  "inputs": [dimensions(value) for value in native.graph.input],
                                  "outputs": [dimensions(value) for value in native.graph.output]}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in information.items() if key not in ("inputs", "outputs")}
                      for name, information in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
