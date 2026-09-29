#!/usr/bin/env python3
"""Actual layer-1 Q producer, attention and output-projection boundary audit."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor

from extract_smolvla_conditioning_probe import (
    COMPACT, OUTPUT, QUERY, READY, dimensions, grouped_control, native_interface,
)


HIDDEN = "/Add_3_output_0"
RESIDUAL = "/input_layernorm_1/Cast_output_0"
BLOCK_OUTPUT = "/Add_5_output_0"
GROUP_READY = ["c17_key_ready", "c17_value_ready"]


def group_ready_models(source, propagated, values, propagated_names=True):
    prepared = Extractor(source).extract_model(["key_1", "value_1"], COMPACT)
    prepared.graph.ClearField("output")
    for original, name, perm, shape in zip(COMPACT, GROUP_READY, ([0, 2, 3, 1], [0, 2, 1, 3]),
                                          ([1, 5, 64, 149], [1, 5, 149, 64])):
        prepared.graph.node.append(helper.make_node("Transpose", [original], [name], perm=perm, name=name))
        prepared.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    result = copy.deepcopy(propagated)
    names = ("prop_kv0_order", "prop_kv0", "prop_key_transposed", "prop_kv1_order", "prop_kv1") if propagated_names else (
        "k_group_order", "k_grouped", "key_transposed", "v_group_order", "v_grouped")
    removed = set(names)
    nodes = []
    for node in result.graph.node:
        if names[0] in node.output:
            nodes.append(helper.make_node("Reshape", [GROUP_READY[0], "c17_key_shape"], [names[2]], name="c17_ready_key"))
        if names[3] in node.output:
            nodes.append(helper.make_node("Reshape", [GROUP_READY[1], "c17_value_shape"], [names[4]], name="c17_ready_value"))
        if not removed.intersection(node.output):
            nodes.append(node)
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    result.graph.initializer.extend([
        numpy_helper.from_array(np.array([5, 64, 149], np.int64), "c17_key_shape"),
        numpy_helper.from_array(np.array([5, 149, 64], np.int64), "c17_value_shape")])
    result.graph.ClearField("input")
    result.graph.input.extend([values[HIDDEN], *prepared.graph.output])
    result.graph.ClearField("value_info")
    return onnx.shape_inference.infer_shapes(prepared), onnx.shape_inference.infer_shapes(result)


def propagated_layout(source, values):
    """Carry group/dimension/head channel order through Q, RoPE and O weights."""
    norm = "/Cast_11_output_0"
    prefix = Extractor(source).extract_model([HIDDEN], [norm])
    constants = Extractor(source).extract_model([], ["/Sin_2_output_0", "/Cos_2_output_0"])
    weights = {value.name: numpy_helper.to_array(value) for value in source.graph.initializer}
    order = np.arange(960).reshape(5, 3, 64).transpose(0, 2, 1).reshape(-1)
    initializers = []
    nodes = []

    def const(name, value):
        name = "prop_" + name
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def node(kind, inputs, output, **attributes):
        output = "prop_" + output
        nodes.append(helper.make_node(kind, inputs, [output], name=output, **attributes))
        return output

    hidden = node("Transpose", [norm], "hidden_channels", perm=[0, 2, 1])
    query = node("MatMul", [const("q_weight", weights["onnx::MatMul_14912"][:, order].T.copy()), hidden], "q_channels")
    query = node("Reshape", [query, const("q_shape", np.array([5, 64, 3, 50], np.int64))], "q_groups")
    halves = ["prop_first", "prop_second"]
    nodes.append(helper.make_node("Split", [query, const("split", np.array([32, 32], np.int64))], halves, axis=1, name="prop_q_halves"))
    sin = node("Transpose", ["/Sin_2_output_0"], "sin", perm=[0, 3, 2, 1])
    cos = node("Transpose", ["/Cos_2_output_0"], "cos", perm=[0, 3, 2, 1])
    first = node("Sub", [node("Mul", [halves[0], cos], "first_cos"), node("Mul", [halves[1], sin], "second_sin")], "rope_first")
    second = node("Add", [node("Mul", [halves[1], cos], "second_cos"), node("Mul", [halves[0], sin], "first_sin")], "rope_second")
    query = node("Concat", [first, second], "rope", axis=1)
    query = node("Reshape", [query, const("q_packed_shape", np.array([5, 64, 150], np.int64))], "q_packed_transposed")
    query = node("Transpose", [query], "q_packed", perm=[0, 2, 1])
    kv = []
    for ordinal, original in enumerate(COMPACT):
        value = node("Transpose", [original], f"kv{ordinal}_order", perm=[0, 2, 1, 3])
        kv.append(node("Reshape", [value, const(f"kv{ordinal}_shape", np.array([5, 149, 64], np.int64))], f"kv{ordinal}"))
    key = node("Transpose", [kv[0]], "key_transposed", perm=[0, 2, 1])
    score = node("MatMul", [query, key], "score")
    score = node("Mul", [score, const("scale", np.float32(0.125))], "scaled_score")
    probability = node("Softmax", [score], "probability", axis=-1)
    result = node("MatMul", [probability, kv[1]], "attended")
    result = node("Transpose", [result], "attended_channels", perm=[0, 2, 1])
    result = node("Reshape", [result, const("out_shape", np.array([1, 960, 50], np.int64))], "out_channels")
    result = node("Transpose", [result], "out_rows", perm=[0, 2, 1])
    result = node("MatMul", [result, const("o_weight", weights["onnx::MatMul_14993"][order].copy())], "output_projection")
    nodes.append(helper.make_node("Add", [result, RESIDUAL], [BLOCK_OUTPUT], name="prop_residual"))
    body = helper.make_model(helper.make_graph(nodes, "propagated_channel_layout",
                                              [values[norm], values[RESIDUAL], *[values[name] for name in COMPACT]],
                                              [values[BLOCK_OUTPUT]], initializers), opset_imports=list(source.opset_import))
    body.ir_version = source.ir_version
    return join([prefix, constants, body], [values[HIDDEN], *[values[name] for name in COMPACT]],
                [values[BLOCK_OUTPUT]], "actual_cross_propagated_layout")


def join(parts, inputs, outputs, name):
    result = copy.deepcopy(parts[0])
    result.graph.name = name
    nodes, initializers = [], {}
    for part in parts:
        nodes.extend(copy.deepcopy(list(part.graph.node)))
        for value in part.graph.initializer:
            if value.name in initializers and value.SerializeToString() != initializers[value.name].SerializeToString():
                raise ValueError(f"Different shared initializer: {value.name}")
            initializers[value.name] = copy.deepcopy(value)
    for field, values in (("node", nodes), ("initializer", initializers.values()),
                          ("input", inputs), ("output", outputs), ("value_info", [])):
        result.graph.ClearField(field)
        getattr(result.graph, field).extend(values)
    result = onnx.shape_inference.infer_shapes(result)
    onnx.checker.check_model(result)
    return result


def extract(args):
    source = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    extractor = Extractor(source)
    values = {value.name: value for value in list(source.graph.value_info) + list(source.graph.input)}
    required = {HIDDEN: [1, 50, 480], QUERY: [1, 50, 15, 64],
                OUTPUT: [1, 50, 960], BLOCK_OUTPUT: [1, 50, 480]}
    for name, shape in required.items():
        if dimensions(values[name]) != shape:
            raise ValueError(f"Unexpected actual boundary: {name}")
    producer = extractor.extract_model([HIDDEN], [QUERY])
    tail = extractor.extract_model([OUTPUT, RESIDUAL], [BLOCK_OUTPUT])
    group = grouped_control(source)
    grouped = join([producer, group, tail],
                   [values[HIDDEN], *[values[name] for name in COMPACT]],
                   [values[BLOCK_OUTPUT]], "actual_cross_joint_grouped")
    models = {
        "block_original": extractor.extract_model([HIDDEN, "key_1", "value_1"], [BLOCK_OUTPUT]),
        "block_ready": extractor.extract_model([HIDDEN, *READY], [BLOCK_OUTPUT]),
        "block_compact_repeat": extractor.extract_model([HIDDEN, *COMPACT], [BLOCK_OUTPUT]),
        "block_grouped_joint": grouped,
        "block_grouped_interleaved": join([producer, grouped_control(source, token_interleaved=True), tail],
                                          [values[HIDDEN], *[values[name] for name in COMPACT]],
                                          [values[BLOCK_OUTPUT]], "actual_cross_token_interleaved"),
        "block_grouped_propagated": propagated_layout(source, values),
        "query_producer": producer,
        "grouped_attention": group,
        "output_consumer": extractor.extract_model([HIDDEN, OUTPUT], [BLOCK_OUTPUT]),
        "prepare_compact": extractor.extract_model(["key_1", "value_1"], COMPACT),
        "prepare_ready": extractor.extract_model(["key_1", "value_1"], READY),
    }
    models["prepare_group_ready"], models["block_grouped_propagated_ready"] = group_ready_models(
        source, models["block_grouped_propagated"], values)
    _, models["block_grouped_joint_ready"] = group_ready_models(source, grouped, values, propagated_names=False)
    model_dir, ref_dir = args.directory / "models", args.directory / "reference"
    model_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    record = {"scope": "Actual layer-1 input RMSNorm, Q projection, full RoPE, attention, output projection and residual; no MLP or full dependent flow",
              "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
              "hidden_source": HIDDEN, "block_output_source": BLOCK_OUTPUT,
              "ready_source": READY, "compact_source": COMPACT, "group_ready_source": GROUP_READY, "models": {}}
    for name, model in models.items():
        onnx.save(model, ref_dir / (name + ".onnx"))
        grouped_ready = name in ("prepare_group_ready", "block_grouped_propagated_ready", "block_grouped_joint_ready")
        native = native_interface(model, ready_names=GROUP_READY if grouped_ready else READY,
                                  flat_widths=(480,), ready_heads=5 if grouped_ready else 15)
        onnx.save(native, model_dir / (name + ".onnx"))
        record["models"][name] = {
            "original_inputs": [value.name for value in model.graph.input],
            "original_outputs": [value.name for value in model.graph.output],
            "native_inputs": [{"name": value.name, "shape": dimensions(value)} for value in native.graph.input],
            "native_outputs": [{"name": value.name, "shape": dimensions(value)} for value in native.graph.output],
            "operations": {kind: sum(node.op_type == kind for node in model.graph.node)
                           for kind in sorted({node.op_type for node in model.graph.node})},
        }
    onnx.save(extractor.extract_model(["suffix", "key_0", "value_0"], [HIDDEN]), ref_dir / "prefix_to_hidden.onnx")
    (args.directory / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"models": list(models), "scope": record["scope"]}))


def verify(args):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    manifest = json.loads((args.directory / "manifest.json").read_text())
    sessions = {name: ort.InferenceSession(str(args.directory / "reference" / (name + ".onnx")), options)
                for name in [*manifest["models"], "prefix_to_hidden"]}
    natives = {name: ort.InferenceSession(str(args.directory / "models" / (name + ".onnx")), options)
               for name in manifest["models"]}
    suffix = np.load(args.fixtures / "suffix.npy")
    raw = {name: np.load(args.fixtures / (name + ".npy"))
           for name in ("key_0", "value_0", "key_1", "value_1")}
    random = np.random.default_rng(812)
    checks = {}
    for case in ("original", "other_suffix", "changed_prefix"):
        action = suffix + np.float32(0.1) * random.standard_normal(suffix.shape, dtype=np.float32) if case == "other_suffix" else suffix
        prefix = {name: np.ascontiguousarray(-value[:, ::-1]) if case == "changed_prefix" and name == "value_1" else value
                  for name, value in raw.items()}
        hidden = sessions["prefix_to_hidden"].run(None, {"suffix": action, **{name: prefix[name] for name in ("key_0", "value_0")}})[0]
        feeds = {HIDDEN: hidden, **prefix}
        for name, sources in (("prepare_compact", COMPACT), ("prepare_ready", READY), ("prepare_group_ready", GROUP_READY)):
            results = sessions[name].run(None, {key: feeds[key] for key in manifest["models"][name]["original_inputs"]})
            feeds.update(zip(sources, results))
        query = sessions["query_producer"].run(None, {HIDDEN: hidden})[0]
        feeds[QUERY] = query
        attended = sessions["grouped_attention"].run(None, {key: feeds[key] for key in (QUERY, *COMPACT)})[0]
        feeds[OUTPUT] = attended
        feeds[RESIDUAL] = hidden
        expected = sessions["block_original"].run(None, {key: feeds[key] for key in (HIDDEN, "key_1", "value_1")})[0]
        case_checks = {}
        for name in ("block_original", "block_ready", "block_compact_repeat", "block_grouped_joint",
                     "block_grouped_interleaved", "block_grouped_propagated", "block_grouped_propagated_ready",
                     "block_grouped_joint_ready", "output_consumer"):
            inputs = manifest["models"][name]["original_inputs"]
            result = sessions[name].run(None, {key: feeds[key] for key in inputs})[0]
            native_feeds = {}
            for value, original in zip(natives[name].get_inputs(), inputs):
                if original in READY or original in GROUP_READY:
                    names = GROUP_READY if original in GROUP_READY else READY
                    heads = 5 if original in GROUP_READY else 15
                    state = feeds[original] if original == names[0] else feeds[original].transpose(0, 1, 3, 2)
                    padded = np.zeros((heads, 64, 1, 152), np.float32)
                    padded[..., :149] = state.reshape(heads, 64, 1, 149)
                    native_feeds[value.name] = padded
                else:
                    native_feeds[value.name] = feeds[original].reshape(value.shape)
            native_result = natives[name].run(None, native_feeds)[0].reshape(expected.shape)
            if not np.isfinite(result).all() or not np.allclose(result, expected, atol=1e-5, rtol=1e-5):
                raise ValueError(f"FP32 graph parity failed: {case}/{name}")
            if not np.array_equal(native_result, result):
                raise ValueError(f"Native adapter parity failed: {case}/{name}")
            case_checks[name] = {"max_abs": float(np.max(np.abs(result - expected))),
                                 "bitwise_equal": bool(np.array_equal(result.view(np.uint32), expected.view(np.uint32)))}
        for name, value in (("hidden", hidden), ("reference", expected), ("query", query),
                            ("key", prefix["key_1"]), ("value", prefix["value_1"])):
            np.save(args.directory / f"{case}_{name}.npy", value)
        checks[case] = case_checks
    (args.directory / "fp32_parity.json").write_text(json.dumps({"backend": "ORT CPU optimizations disabled", "checks": checks}, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        if not args.fixtures:
            parser.error("--fixtures required for verification")
        verify(args)
    else:
        if not args.onnx:
            parser.error("--onnx required for extraction")
        extract(args)


if __name__ == "__main__":
    main()
