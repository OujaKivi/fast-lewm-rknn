#!/usr/bin/env python3
"""Extract real layer-0 graphs and a proven-zero attention-mask control."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.utils import Extractor


PREFIX = "/vision_model/encoder/layers.0"


def set_shape(value_info, dimensions):
    value_info.type.tensor_type.shape.ClearField("dim")
    for size in dimensions:
        value_info.type.tensor_type.shape.dim.add().dim_value = size


def extract(model, inputs, output, shapes):
    result = Extractor(model).extract_model(inputs, [output] if isinstance(output, str) else output)
    for value_info in result.graph.input:
        set_shape(value_info, shapes[value_info.name])
    onnx.checker.check_model(result)
    return result


def native_interfaces(model, input_indices, output_indices):
    """Expose channel-major 4D semantics so the SDK can select packed native I/O."""
    result = copy.deepcopy(model)
    before, after = [], []
    for is_input, indices, tensors, nodes in (
        (True, input_indices, result.graph.input, before),
        (False, output_indices, result.graph.output, after),
    ):
        for index in indices:
            tensor = tensors[index]
            batch, length, width = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
            original_name = tensor.name
            native_name = original_name + "_native_nchw"
            tag = f"__native_{'in' if is_input else 'out'}_{index}"
            axes = tag + "_axes"
            result.graph.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), axes))
            if is_input:
                nodes.extend([helper.make_node("Squeeze", [native_name, axes], [tag], name=tag + "_squeeze"),
                              helper.make_node("Transpose", [tag], [original_name], perm=[0, 2, 1], name=tag + "_transpose")])
            else:
                nodes.extend([helper.make_node("Transpose", [original_name], [tag], perm=[0, 2, 1], name=tag + "_transpose"),
                              helper.make_node("Unsqueeze", [tag, axes], [native_name], name=tag + "_unsqueeze")])
            tensor.name = native_name
            set_shape(tensor, [batch, width, 1, length])
    old_nodes = list(result.graph.node)
    result.graph.ClearField("node")
    result.graph.node.extend(before + old_nodes + after)
    result.graph.ClearField("value_info")
    result = onnx.shape_inference.infer_shapes(result)
    onnx.checker.check_model(result)
    return result


def remove_verified_zero_masks(model, batch):
    mask_model = Extractor(model).extract_model([], ["/vision_model/Where_1_output_0"])
    if mask_model.graph.input:
        raise ValueError("The exported mask depends on runtime inputs")
    mask = ReferenceEvaluator(mask_model).run(None, {})[0]
    if mask.shape != (batch, 1, 1024, 1024) or np.any(mask != 0):
        raise ValueError("Mask is not identically zero for the expected geometry")
    result = copy.deepcopy(model)
    replaced = 0
    for index, node in enumerate(result.graph.node):
        if node.op_type == "Add" and node.name.endswith("/self_attn/Add"):
            if not node.input[1].endswith("/self_attn/Slice_output_0"):
                raise ValueError(f"Unexpected mask input: {node.input[1]}")
            result.graph.node[index].CopyFrom(helper.make_node(
                "Identity", [node.input[0]], list(node.output), name=node.name + "_zero_mask_removed"))
            replaced += 1
    if replaced != 12:
        raise ValueError(f"Expected 12 mask additions, found {replaced}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-full-unmasked", action="store_true")
    parser.add_argument("--partition-heads", type=int, choices=[3, 4, 6], default=6)
    parser.add_argument("--native-interfaces", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    image_shape = [dimension.dim_value for dimension in model.graph.input[0].type.tensor_type.shape.dim]
    if image_shape[1:] != [3, 512, 512] or image_shape[0] not in (1, 2):
        raise ValueError(f"Unexpected fixed image geometry: {image_shape}")
    batch = image_shape[0]

    mask_name = "/vision_model/Where_1_output_0"
    mask_model = Extractor(model).extract_model([], [mask_name])
    if mask_model.graph.input:
        raise ValueError("The exported mask depends on runtime inputs")
    mask = ReferenceEvaluator(mask_model).run(None, {})[0]
    if mask.shape != (batch, 1, 1024, 1024) or np.any(mask != 0):
        raise ValueError(f"Mask cannot be removed: shape={mask.shape}, nonzero={np.count_nonzero(mask)}")

    unmasked = copy.deepcopy(model)
    replaced = 0
    for index, node in enumerate(unmasked.graph.node):
        if node.op_type == "Add" and node.name.endswith("/self_attn/Add"):
            if not node.input[1].endswith("/self_attn/Slice_output_0"):
                raise ValueError(f"Unexpected mask input: {node.input[1]}")
            unmasked.graph.node[index].CopyFrom(helper.make_node(
                "Identity", [node.input[0]], list(node.output), name=node.name + "_zero_mask_removed"))
            replaced += 1
    if replaced != 12:
        raise ValueError(f"Expected 12 visual attention mask additions, found {replaced}")
    if args.write_full_unmasked:
        full = Extractor(unmasked).extract_model([value.name for value in model.graph.input],
                                                [value.name for value in model.graph.output])
        onnx.checker.check_model(full)
        onnx.save(full, args.output_dir / "vision_unmasked.onnx")

    hidden = "/vision_model/embeddings/Add_output_0"
    qkv = [f"{PREFIX}/self_attn/{letter}_proj/Add_output_0" for letter in "qkv"]
    attention_output = f"{PREFIX}/self_attn/Reshape_3_output_0"
    mlp_input = f"{PREFIX}/layer_norm2/LayerNormalization_output_0"
    probes = {
        "attention": (qkv, attention_output),
        "block": ([hidden], f"{PREFIX}/Add_1_output_0"),
        "mlp": ([mlp_input], f"{PREFIX}/mlp/fc2/Add_output_0"),
        "fc1": ([mlp_input], f"{PREFIX}/mlp/fc1/Add_output_0"),
        "fc2": ([f"{PREFIX}/mlp/activation_fn/Mul_5_output_0"], f"{PREFIX}/mlp/fc2/Add_output_0"),
        "qkv": ([hidden], qkv),
        "suffix": ([hidden, attention_output], f"{PREFIX}/Add_1_output_0"),
    }
    report = {"scope": "Real exported vision layer-0 subgraphs; mask removal only for the verified fixed all-valid geometry",
              "batch": batch, "mask_shape": list(mask.shape), "mask_nonzero": int(np.count_nonzero(mask)),
              "probes": {}}
    for name, (inputs, output) in probes.items():
        shapes = {tensor: [batch, 1024, 3072 if name == "fc2" else 768] for tensor in inputs}
        for variant, source in (("original", model), ("unmasked", unmasked)):
            if name not in ("attention", "block") and variant == "unmasked":
                continue
            graph = extract(source, inputs, output, shapes)
            path = args.output_dir / f"{name}_{variant}.onnx"
            onnx.save(graph, path)
            report["probes"][path.stem] = {"nodes": len(graph.graph.node), "initializer_bytes": sum(len(value.raw_data) for value in graph.graph.initializer),
                                          "inputs": inputs, "outputs": [output] if isinstance(output, str) else output}

    partition = onnx.load(args.output_dir / "attention_unmasked.onnx")
    width = args.partition_heads * 64
    for node in partition.graph.node:
        if node.op_type != "Constant":
            continue
        value = numpy_helper.to_array(node.attribute[0].t)
        if np.array_equal(value, [batch, 1024, 12, 64]):
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([batch, 1024, args.partition_heads, 64], dtype=np.int64)))
        elif np.array_equal(value, [batch, 1024, 768]):
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([batch, 1024, width], dtype=np.int64)))
    for tensor in list(partition.graph.input) + list(partition.graph.output):
        set_shape(tensor, [batch, 1024, width])
    partition.graph.ClearField("value_info")
    partition = onnx.shape_inference.infer_shapes(partition)
    onnx.checker.check_model(partition)
    onnx.save(partition, args.output_dir / f"attention_unmasked_h{args.partition_heads}.onnx")
    native_models = {}
    if args.native_interfaces:
        producer = onnx.load(args.output_dir / "qkv_original.onnx")
        native_models = {
            "qkv_native": native_interfaces(producer, [0], [0, 1, 2]),
            "block_unmasked_native": native_interfaces(onnx.load(args.output_dir / "block_unmasked.onnx"), [0], []),
            "attention_native_full": native_interfaces(onnx.load(args.output_dir / "attention_unmasked.onnx"), [0, 1, 2], [0]),
            f"attention_native_h{args.partition_heads}": native_interfaces(partition, [0, 1, 2], [0]),
            "suffix_native": native_interfaces(onnx.load(args.output_dir / "suffix_original.onnx"), [0, 1], []),
        }
        for name, graph in native_models.items():
            onnx.save(graph, args.output_dir / f"{name}.onnx")

    generator = np.random.default_rng(0)
    for name in ("attention", "block"):
        inputs, _ = probes[name]
        values = {tensor: generator.standard_normal((batch, 1024, 768), dtype=np.float32) for tensor in inputs}
        outputs = []
        for variant in ("original", "unmasked"):
            runtime = ReferenceEvaluator(str(args.output_dir / f"{name}_{variant}.onnx"))
            outputs.append(runtime.run(None, values)[0])
        if not all(np.all(np.isfinite(output)) for output in outputs):
            raise ValueError(f"Non-finite {name} reference output")
        report[f"{name}_fp32_max_abs_difference"] = float(np.max(np.abs(outputs[0] - outputs[1])))
        if name == "attention":
            partial = ReferenceEvaluator(partition).run(None, {key: value[..., :width] for key, value in values.items()})[0]
            if not np.all(np.isfinite(partial)):
                raise ValueError("Non-finite partition reference output")
            report["partition_heads"] = args.partition_heads
            report["partition_fp32_max_abs_difference"] = float(np.max(np.abs(partial - outputs[1][..., :width])))
        elif native_models:
            producer = native_models["qkv_native"]
            packed_hidden = np.expand_dims(np.transpose(values[hidden], [0, 2, 1]), 2)
            native_qkv = ReferenceEvaluator(producer).run(None, {producer.graph.input[0].name: packed_hidden})
            attention = native_models[f"attention_native_h{args.partition_heads}"]
            parts = []
            for part in range(12 // args.partition_heads):
                feed = {tensor.name: value[:, part * width:(part + 1) * width]
                        for tensor, value in zip(attention.graph.input, native_qkv[:3])}
                parts.append(ReferenceEvaluator(attention).run(None, feed)[0])
            suffix = native_models["suffix_native"]
            feed = {suffix.graph.input[0].name: packed_hidden,
                    suffix.graph.input[1].name: np.concatenate(parts, axis=1)}
            native_output = ReferenceEvaluator(suffix).run(None, feed)[0]
            if not np.all(np.isfinite(native_output)):
                raise ValueError("Non-finite native-interface block output")
            report["native_split_block_fp32_max_abs_difference"] = float(np.max(np.abs(native_output - outputs[0])))
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
