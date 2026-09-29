#!/usr/bin/env python3
"""Extract a complete packed 12-layer vision pipeline and FP32 fixtures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.utils import Extractor

from extract_smolvla_vision_probes import extract, native_interfaces, remove_verified_zero_masks, set_shape


def attention_signature(model):
    names = {value.name: f"input{index}" for index, value in enumerate(model.graph.input)}
    if model.graph.initializer:
        raise ValueError("Expected parameter-free attention")
    records = []
    for index, node in enumerate(model.graph.node):
        inputs = [names[name] for name in node.input]
        for output_index, name in enumerate(node.output):
            names[name] = f"node{index}:{output_index}"
        records.append([node.op_type, inputs, [value.SerializeToString().hex() for value in node.attribute]])
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def packed(value):
    return np.expand_dims(value.transpose(0, 2, 1), 2)


def run(model, values):
    result = ReferenceEvaluator(model).run(None, {tensor.name: value for tensor, value in zip(model.graph.input, values)})
    if not all(np.all(np.isfinite(value)) for value in result):
        raise ValueError("Non-finite FP32 reference output")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-heads", type=int, choices=[4, 6], default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = np.load(args.input).astype(np.float32)
    if image.shape != (1, 3, 512, 512) or not np.all(np.isfinite(image)):
        raise ValueError("Expected finite B1 512x512 image fixture")
    model = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    unmasked = remove_verified_zero_masks(model, 1)
    onnx.save(model, args.output_dir / "vision_original.onnx")
    full = Extractor(unmasked).extract_model([model.graph.input[0].name], [model.graph.output[0].name])
    onnx.save(full, args.output_dir / "vision_unmasked.onnx")
    first_hidden = "/vision_model/embeddings/Add_output_0"
    layer_outputs = [f"/vision_model/encoder/layers.{layer}/Add_1_output_0" for layer in range(12)]
    auditor = Extractor(model).extract_model([model.graph.input[0].name], [*layer_outputs, model.graph.output[0].name])
    onnx.save(auditor, args.output_dir / "vision_audit.onnx")
    stem = native_interfaces(extract(full, [full.graph.input[0].name], first_hidden,
                                     {full.graph.input[0].name: list(image.shape)}), [], [0])
    tail = native_interfaces(extract(full, [layer_outputs[-1]], full.graph.output[0].name,
                                     {layer_outputs[-1]: [1, 1024, 768]}), [0], [])
    onnx.save(stem, args.output_dir / "stem_native.onnx")
    onnx.save(tail, args.output_dir / "tail_native.onnx")
    signature, shared_attention = None, None
    producers, consumers = [], []
    for layer in range(12):
        prefix = f"/vision_model/encoder/layers.{layer}"
        hidden = first_hidden if layer == 0 else layer_outputs[layer - 1]
        qkv = [f"{prefix}/self_attn/{letter}_proj/Add_output_0" for letter in "qkv"]
        attention_output = f"{prefix}/self_attn/Reshape_3_output_0"
        shapes = {name: [1, 1024, 768] for name in [hidden, attention_output, *qkv]}
        producer = native_interfaces(extract(full, [hidden], qkv, shapes), [0], [0, 1, 2])
        consumer = native_interfaces(extract(full, [hidden, attention_output], layer_outputs[layer], shapes), [0, 1], [0])
        attention = extract(full, qkv, attention_output, shapes)
        current = attention_signature(attention)
        if signature is None:
            signature, shared_attention = current, attention
        elif signature != current:
            raise ValueError(f"Layer {layer} attention is not structurally identical to layer 0")
        onnx.save(producer, args.output_dir / f"layer{layer:02d}_qkv.onnx")
        onnx.save(consumer, args.output_dir / f"layer{layer:02d}_suffix.onnx")
        producers.append(producer)
        consumers.append(consumer)
    onnx.save(native_interfaces(shared_attention, [0, 1, 2], [0]), args.output_dir / "attention_full.onnx")
    width = args.partition_heads * 64
    for node in shared_attention.graph.node:
        if node.op_type != "Constant":
            continue
        value = numpy_helper.to_array(node.attribute[0].t)
        if np.array_equal(value, [1, 1024, 12, 64]):
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([1, 1024, args.partition_heads, 64], dtype=np.int64)))
        elif np.array_equal(value, [1, 1024, 768]):
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([1, 1024, width], dtype=np.int64)))
    for value in [*shared_attention.graph.input, *shared_attention.graph.output]:
        set_shape(value, [1, 1024, width])
    shared_attention.graph.ClearField("value_info")
    shared_attention = onnx.shape_inference.infer_shapes(shared_attention)
    partition = native_interfaces(shared_attention, [0, 1, 2], [0])
    onnx.save(partition, args.output_dir / f"attention_h{args.partition_heads}.onnx")

    references = ReferenceEvaluator(model).run([*layer_outputs, model.graph.output[0].name],
                                              {model.graph.input[0].name: image})
    if not all(np.all(np.isfinite(value)) for value in references):
        raise ValueError("Non-finite original reference")
    hidden = run(stem, [image])[0]
    differences = []
    for layer, (producer, consumer) in enumerate(zip(producers, consumers)):
        qkv = run(producer, [hidden])
        outputs = []
        for part in range(12 // args.partition_heads):
            outputs.append(run(partition, [value[:, part * width:(part + 1) * width] for value in qkv])[0])
        hidden = run(consumer, [hidden, np.concatenate(outputs, axis=1)])[0]
        differences.append(float(np.max(np.abs(hidden - packed(references[layer])))))
    final = run(tail, [hidden])[0]
    final_difference = float(np.max(np.abs(final - references[-1])))
    if max([*differences, final_difference]) != 0:
        raise ValueError(f"FP32 pipeline differs: layers={differences}, final={final_difference}")
    image.tofile(args.output_dir / "image_fp32.bin")
    np.stack(references[:-1]).tofile(args.output_dir / "layers_fp32.bin")
    references[-1].tofile(args.output_dir / "embedding_fp32.bin")
    report = {"source_onnx": str(args.onnx), "input_fixture": str(args.input),
              "image_shape": list(image.shape), "embedding_shape": list(final.shape), "layers": 12,
              "partition_heads": args.partition_heads, "shared_attention_signature": signature,
              "layer_fp32_max_abs_differences": differences, "embedding_fp32_max_abs_difference": final_difference,
              "scope": "Original versus packed/head-partitioned complete vision+connector, FP32 reference, one fixture"}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
