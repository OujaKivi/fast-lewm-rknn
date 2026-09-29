#!/usr/bin/env python3
"""Numerical ablations and native head-parallel controls for the C17 boundary."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor

from extract_smolvla_cross_boundary import (
    BLOCK_OUTPUT, GROUP_READY, HIDDEN, RESIDUAL, join,
)
from extract_smolvla_conditioning_probe import OUTPUT, QUERY, READY, dimensions, native_interface


def original_output_order(model, source):
    result = copy.deepcopy(model)
    weights = {value.name: numpy_helper.to_array(value) for value in source.graph.initializer}
    order = np.arange(960).reshape(5, 3, 64).transpose(0, 2, 1).reshape(-1)
    nodes = []
    for node in result.graph.node:
        if node.output[0] == "prop_output_projection":
            nodes.extend([
                helper.make_node("Gather", ["prop_out_rows", "audit_inverse_order"], ["audit_original_rows"], axis=2),
                helper.make_node("MatMul", ["audit_original_rows", "audit_original_o_weight"], ["prop_output_projection"]),
            ])
        else:
            nodes.append(node)
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    result.graph.initializer.extend([
        numpy_helper.from_array(np.argsort(order).astype(np.int64), "audit_inverse_order"),
        numpy_helper.from_array(weights["onnx::MatMul_14993"].copy(), "audit_original_o_weight"),
    ])
    return onnx.shape_inference.infer_shapes(result)


def original_query_path(model, source):
    values = {v.name: v for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    producer = Extractor(source).extract_model([HIDDEN], [QUERY])
    tail = Extractor(model).extract_model(["prop_q_packed", RESIDUAL, *GROUP_READY], [BLOCK_OUTPUT])
    bridge = helper.make_model(helper.make_graph([
        helper.make_node("Reshape", [QUERY, "audit_q_split_shape"], ["audit_q_split"]),
        helper.make_node("Transpose", ["audit_q_split"], ["audit_q_order"], perm=[0, 2, 3, 1, 4]),
        helper.make_node("Reshape", ["audit_q_order", "audit_q_packed_shape"], ["prop_q_packed"]),
    ], "original_q_then_group", [producer.graph.input[0], producer.graph.output[0]],
        [values["prop_q_packed"], values[RESIDUAL]], [
            numpy_helper.from_array(np.array([1, 50, 5, 3, 64], np.int64), "audit_q_split_shape"),
            numpy_helper.from_array(np.array([5, 150, 64], np.int64), "audit_q_packed_shape"),
        ]), opset_imports=list(source.opset_import))
    bridge.ir_version = source.ir_version
    return join([producer, bridge, tail], list(model.graph.input), list(model.graph.output), "original_q_ablation")


def head_models(source):
    producer = Extractor(source).extract_model([HIDDEN], [QUERY])
    producer.graph.node.extend([
        helper.make_node("Transpose", [QUERY], ["head_q_order"], perm=[0, 2, 3, 1]),
        helper.make_node("Reshape", ["head_q_order", "head_q_shape"], ["head_q"]),
    ])
    producer.graph.initializer.append(numpy_helper.from_array(np.array([1, 960, 1, 50], np.int64), "head_q_shape"))
    producer.graph.ClearField("output")
    producer.graph.output.append(helper.make_tensor_value_info("head_q", onnx.TensorProto.FLOAT, [1, 960, 1, 50]))
    tail = Extractor(source).extract_model([HIDDEN, OUTPUT], [BLOCK_OUTPUT])
    tail.graph.node.insert(0, helper.make_node("Reshape", ["head_a_order", "head_a_shape"], [OUTPUT]))
    tail.graph.node.insert(0, helper.make_node("Transpose", ["head_a_trim"], ["head_a_order"], perm=[0, 3, 2, 1]))
    tail.graph.node.insert(0, helper.make_node("Slice", ["head_a", "head_start", "head_end", "head_axis", "head_step"], ["head_a_trim"]))
    for label, value in (("start", [0]), ("end", [50]), ("axis", [3]), ("step", [1])):
        tail.graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), "head_" + label))
    tail.graph.initializer.append(numpy_helper.from_array(np.array([1, 50, 960], np.int64), "head_a_shape"))
    tail.graph.input[1].CopyFrom(helper.make_tensor_value_info("head_a", onnx.TensorProto.FLOAT, [1, 960, 1, 52]))
    models = {"head_producer": producer, "head_consumer": tail}
    for heads in (5, 15):
        nodes, initializers = [], []

        def const(name, value):
            initializers.append(numpy_helper.from_array(np.asarray(value), name))
            return name

        def op(kind, inputs, output, **attrs):
            nodes.append(helper.make_node(kind, inputs, [output], name=output, **attrs))
            return output

        trimmed_q = op("Slice", ["head_q", *[const("q_" + label, np.array(value, np.int64)) for label, value in
                        (("start", [0]), ("end", [50]), ("axis", [3]), ("step", [1]))]], "q_trim")
        q = op("Reshape", [trimmed_q, const("q_shape", np.array([heads, 64, 50], np.int64))], "q_channels")
        q = op("Transpose", [q], "q_rows", perm=[0, 2, 1])
        states = []
        for name in ("k", "v"):
            trimmed = op("Slice", [name, *[const(name + label, np.array(value, np.int64)) for label, value in
                         (("start", [0]), ("end", [149]), ("axis", [3]), ("step", [1]))]], name + "_trim")
            state = op("Reshape", [trimmed, const(name + "_shape", np.array([heads, 64, 149], np.int64))], name + "_channels")
            states.append(state)
        scores = op("MatMul", [q, states[0]], "score")
        scaled = op("Mul", [scores, const("scale", np.float32(.125))], "scaled")
        prob = op("Softmax", [scaled], "probability", axis=-1)
        v = op("Transpose", [states[1]], "v_rows", perm=[0, 2, 1])
        attended = op("MatMul", [prob, v], "attended")
        channels = op("Transpose", [attended], "a_channels", perm=[0, 2, 1])
        op("Reshape", [channels, const("a_shape", np.array([1, heads * 64, 1, 50], np.int64))], "head_a")
        model = helper.make_model(helper.make_graph(nodes, "native_head_attention", [
            helper.make_tensor_value_info("head_q", onnx.TensorProto.FLOAT, [1, heads * 64, 1, 52]),
            *[helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [heads, 64, 1, 152]) for n in ("k", "v")],
        ], [helper.make_tensor_value_info("head_a", onnx.TensorProto.FLOAT, [1, heads * 64, 1, 50])], initializers),
            opset_imports=list(source.opset_import))
        model.ir_version = source.ir_version
        models[f"head_attention_{heads}"] = model
    return models


def compact_head_models(propagated, restored):
    producer = Extractor(propagated).extract_model([HIDDEN], ["prop_q_packed_transposed"])
    producer.graph.node.append(helper.make_node("Reshape", ["prop_q_packed_transposed", "compact_q_shape"], ["group_q"]))
    producer.graph.initializer.append(numpy_helper.from_array(np.array([1, 320, 1, 150], np.int64), "compact_q_shape"))
    producer.graph.ClearField("output")
    producer.graph.output.append(helper.make_tensor_value_info("group_q", onnx.TensorProto.FLOAT, [1, 320, 1, 150]))
    consumer = Extractor(restored).extract_model([HIDDEN, "prop_attended_channels"], [BLOCK_OUTPUT])
    consumer.graph.node.insert(0, helper.make_node("Reshape", ["compact_a_trim", "compact_a_shape"], ["prop_attended_channels"]))
    consumer.graph.node.insert(0, helper.make_node("Slice", ["group_a", "compact_start", "compact_end", "compact_axis", "compact_step"], ["compact_a_trim"]))
    for name, value in (("a_shape", [5, 64, 150]), ("start", [0]), ("end", [150]), ("axis", [3]), ("step", [1])):
        consumer.graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), "compact_" + name))
    consumer.graph.input[1].CopyFrom(helper.make_tensor_value_info("group_a", onnx.TensorProto.FLOAT, [1, 320, 1, 152]))
    models = {"group_producer": producer, "group_consumer": consumer}
    for groups in (1, 2, 5):
        nodes, initializers = [], []

        def const(name, value):
            initializers.append(numpy_helper.from_array(np.asarray(value), name))
            return name

        def op(kind, inputs, output, **attrs):
            nodes.append(helper.make_node(kind, inputs, [output], name=output, **attrs))
            return output

        states = []
        for name, rows in (("group_q", 150), ("k", 149), ("v", 149)):
            trim = op("Slice", [name, *[const(name + "_" + label, np.array(value, np.int64)) for label, value in
                       (("start", [0]), ("end", [rows]), ("axis", [3]), ("step", [1]))]], name + "_trim")
            states.append(op("Reshape", [trim, const(name + "_shape", np.array([groups, 64, rows], np.int64))], name + "_channels"))
        q = op("Transpose", [states[0]], "q_rows", perm=[0, 2, 1])
        score = op("MatMul", [q, states[1]], "score")
        scaled = op("Mul", [score, const("scale", np.float32(.125))], "scaled")
        prob = op("Softmax", [scaled], "probability", axis=-1)
        v = op("Transpose", [states[2]], "v_rows", perm=[0, 2, 1])
        a = op("MatMul", [prob, v], "attended")
        a = op("Transpose", [a], "a_channels", perm=[0, 2, 1])
        op("Reshape", [a, const("a_shape", np.array([1, groups * 64, 1, 150], np.int64))], "group_a")
        model = helper.make_model(helper.make_graph(nodes, "compact_native_groups", [
            helper.make_tensor_value_info("group_q", onnx.TensorProto.FLOAT, [1, groups * 64, 1, 152]),
            *[helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [groups, 64, 1, 152]) for n in ("k", "v")],
        ], [helper.make_tensor_value_info("group_a", onnx.TensorProto.FLOAT, [1, groups * 64, 1, 150])], initializers),
            opset_imports=list(propagated.opset_import))
        model.ir_version = propagated.ir_version
        models[f"group_attention_{groups}"] = model
    return models


def adapt_boundary(model):
    # Only hidden uses the existing adapter; head-major native edges must remain unchanged.
    result = copy.deepcopy(model)
    for is_input, infos in ((True, result.graph.input), (False, result.graph.output)):
        for index, info in enumerate(infos):
            if info.name not in (HIDDEN, BLOCK_OUTPUT):
                continue
            original = info.name
            name = f"follow_{'input' if is_input else 'output'}_{index}"
            shape = [1, 50, 480] if is_input else [1, 50, 1, 480]
            result.graph.initializer.append(numpy_helper.from_array(np.array(shape, np.int64), name + "_shape"))
            node = helper.make_node("Reshape", [name if is_input else original, name + "_shape"], [original if is_input else name])
            if is_input:
                result.graph.node.insert(0, node)
            else:
                result.graph.node.append(node)
            info.CopyFrom(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 50, 1, 480]))
    result.graph.ClearField("value_info")
    return onnx.shape_inference.infer_shapes(result)


def original_order_group_models(opsets, ir_version):
    models = {}
    for groups in (1, 2, 5):
        nodes, initializers = [], []

        def const(name, value):
            initializers.append(numpy_helper.from_array(np.asarray(value), name))
            return name

        def op(kind, inputs, output, **attrs):
            nodes.append(helper.make_node(kind, inputs, [output], name=output, **attrs))
            return output

        states = []
        for name, rows in (("head_q", 50), ("k", 149), ("v", 149)):
            trim = op("Slice", [name, *[const(name + "_" + label, np.array(value, np.int64)) for label, value in
                       (("start", [0]), ("end", [rows]), ("axis", [3]), ("step", [1]))]], name + "_trim")
            shape = [groups, 3, 64, 50] if name == "head_q" else [groups, 64, 149]
            states.append(op("Reshape", [trim, const(name + "_shape", np.array(shape, np.int64))], name + "_channels"))
        q = op("Transpose", [states[0]], "q_head_rows", perm=[0, 1, 3, 2])
        q = op("Reshape", [q, const("q_packed_shape", np.array([groups, 150, 64], np.int64))], "q_rows")
        score = op("MatMul", [q, states[1]], "score")
        scaled = op("Mul", [score, const("scale", np.float32(.125))], "scaled")
        prob = op("Softmax", [scaled], "probability", axis=-1)
        v = op("Transpose", [states[2]], "v_rows", perm=[0, 2, 1])
        a = op("MatMul", [prob, v], "attended")
        a = op("Reshape", [a, const("a_heads_shape", np.array([groups, 3, 50, 64], np.int64))], "a_heads")
        a = op("Transpose", [a], "a_channels", perm=[0, 1, 3, 2])
        op("Reshape", [a, const("a_shape", np.array([1, groups * 192, 1, 50], np.int64))], "head_a")
        model = helper.make_model(helper.make_graph(nodes, "compact_groups_original_head_order", [
            helper.make_tensor_value_info("head_q", onnx.TensorProto.FLOAT, [1, groups * 192, 1, 52]),
            *[helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [groups, 64, 1, 152]) for n in ("k", "v")],
        ], [helper.make_tensor_value_info("head_a", onnx.TensorProto.FLOAT, [1, groups * 192, 1, 50])], initializers),
            opset_imports=opsets)
        model.ir_version = ir_version
        models[f"original_order_attention_{groups}"] = model
    return models


def extract(args):
    source_hash = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
    base_manifest = json.loads((args.base / "manifest.json").read_text())
    if source_hash != base_manifest["source_sha256"]:
        raise ValueError("Followup source differs from the measured base checkpoint graph")
    source = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
    propagated = onnx.load(args.base / "reference/block_grouped_propagated_ready.onnx")
    variants = {"restore_o": original_output_order(propagated, source),
                "original_q": original_query_path(propagated, source)}
    variants["original_q_restore_o"] = original_query_path(variants["restore_o"], source)
    head = {**head_models(source), **compact_head_models(propagated, variants["restore_o"]),
            **original_order_group_models(list(source.opset_import), source.ir_version)}
    directory = args.directory / "models"
    reference = args.directory / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    reference.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, model in {**variants, **head}.items():
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, reference / (name + ".onnx"))
        native = (native_interface(model, ready_names=GROUP_READY, flat_widths=(480,), ready_heads=5)
                  if name in variants else adapt_boundary(model))
        onnx.checker.check_model(native)
        onnx.save(native, directory / (name + ".onnx"))
        manifest[name] = {"inputs": [v.name for v in model.graph.input],
                          "native_inputs": [dimensions(v) for v in native.graph.input],
                          "native_outputs": [dimensions(v) for v in native.graph.output]}
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.directory / "sources.json").write_text(json.dumps({
        "source_sha256": source_hash,
        "base_propagated_sha256": hashlib.sha256((args.base / "reference/block_grouped_propagated_ready.onnx").read_bytes()).hexdigest(),
        "native_onnx_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.onnx"))},
        "logical_onnx_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reference.glob("*.onnx"))},
    }, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def verify(args):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    def session(model):
        return ort.InferenceSession(model.SerializeToString(), options)

    baseline = onnx.load(args.base / "reference/block_ready.onnx")
    propagated = onnx.load(args.base / "reference/block_grouped_propagated_ready.onnx")
    baseline_stages = Extractor(baseline).extract_model([v.name for v in baseline.graph.input], [QUERY, OUTPUT, BLOCK_OUTPUT])
    propagated_stages = Extractor(propagated).extract_model([v.name for v in propagated.graph.input], ["prop_rope", "prop_out_rows", BLOCK_OUTPUT])
    sessions = {"baseline_stages": session(baseline_stages), "propagated_stages": session(propagated_stages),
                "ready": session(onnx.load(args.base / "reference/prepare_ready.onnx")),
                "group_ready": session(onnx.load(args.base / "reference/prepare_group_ready.onnx"))}
    manifest = json.loads((args.directory / "manifest.json").read_text())
    models = {name: onnx.load(args.directory / "reference" / (name + ".onnx")) for name in manifest}
    sessions.update({name: session(model) for name, model in models.items()})
    natives = {name: session(onnx.load(args.directory / "models" / (name + ".onnx"))) for name in manifest}
    order = np.arange(960).reshape(5, 3, 64).transpose(0, 2, 1).reshape(-1)

    def compare(a, b):
        return {"bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
                "different_elements": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
                "max_abs": float(np.max(np.abs(a - b)))}

    report = {"scope": "Independent CPU ORT, optimizations disabled; stage localization is not a fused NPU intermediate capture", "cases": {}}
    for case in ("original", "other_suffix", "changed_prefix"):
        hidden = np.load(args.base / f"{case}_hidden.npy")
        raw = {n + "_1": np.load(args.base / f"{case}_{n}.npy") for n in ("key", "value")}
        ready = sessions["ready"].run(None, raw)
        group_ready = sessions["group_ready"].run(None, raw)
        q, attended, expected = sessions["baseline_stages"].run(None, {HIDDEN: hidden, **dict(zip(READY, ready))})
        pq, pa, po = sessions["propagated_stages"].run(None, {HIDDEN: hidden, **dict(zip(GROUP_READY, group_ready))})
        checks = {"query_after_rope": compare(pq.transpose(3, 0, 2, 1).reshape(q.shape), q),
                  "attention_output_original_order": compare(pa[..., np.argsort(order)], attended),
                  "propagated_final": compare(po, expected)}
        feeds = {HIDDEN: hidden, **dict(zip(GROUP_READY, group_ready))}
        for name in ("restore_o", "original_q", "original_q_restore_o"):
            result = sessions[name].run(None, {n: feeds[n] for n in manifest[name]["inputs"]})[0]
            native_feeds = {}
            for value, original in zip(natives[name].get_inputs(), manifest[name]["inputs"]):
                if original in GROUP_READY:
                    state = feeds[original] if original == GROUP_READY[0] else feeds[original].transpose(0, 1, 3, 2)
                    padded = np.zeros((5, 64, 1, 152), np.float32)
                    padded[..., :149] = state.reshape(5, 64, 1, 149)
                    native_feeds[value.name] = padded
                else:
                    native_feeds[value.name] = hidden.reshape(value.shape)
            native = natives[name].run(None, native_feeds)[0].reshape(expected.shape)
            if not np.array_equal(native, result) or not np.allclose(result, expected, atol=1e-5, rtol=1e-5):
                raise ValueError(f"Ablation parity: {name}/{case}")
            checks[name] = compare(result, expected)
        head_q = np.pad(sessions["head_producer"].run(None, {HIDDEN: hidden})[0], ((0, 0), (0, 0), (0, 0), (0, 2)))
        chunks = []
        for part in range(3):
            states = []
            for index, state in enumerate(ready):
                channels = state if index == 0 else state.transpose(0, 1, 3, 2)
                padded = np.zeros((15, 64, 1, 152), np.float32)
                padded[..., :149] = channels.reshape(15, 64, 1, 149)
                states.append(padded[part * 5:(part + 1) * 5].copy())
            chunks.append(sessions["head_attention_5"].run(None, {"head_q": head_q[:, part * 320:(part + 1) * 320].copy(),
                                                                   "k": states[0], "v": states[1]})[0])
        head_a = np.concatenate(chunks, axis=1)
        result = sessions["head_consumer"].run(None, {HIDDEN: hidden, "head_a": np.pad(head_a, ((0, 0), (0, 0), (0, 0), (0, 2)))})[0]
        checks["head_split_final"] = compare(result, expected)
        if not np.array_equal(result, expected):
            raise ValueError(f"CPU head split mismatch: {case}")
        group_q = np.pad(sessions["group_producer"].run(None, {HIDDEN: hidden})[0], ((0, 0), (0, 0), (0, 0), (0, 2)))
        chunks = []
        for first, count in ((0, 2), (2, 2), (4, 1)):
            states = []
            for index, state in enumerate(group_ready):
                channels = state if index == 0 else state.transpose(0, 1, 3, 2)
                padded = np.zeros((5, 64, 1, 152), np.float32)
                padded[..., :149] = channels.reshape(5, 64, 1, 149)
                states.append(padded[first:first + count].copy())
            chunks.append(sessions[f"group_attention_{count}"].run(None, {
                "group_q": group_q[:, first * 64:(first + count) * 64].copy(), "k": states[0], "v": states[1]})[0])
        group_a = np.pad(np.concatenate(chunks, axis=1), ((0, 0), (0, 0), (0, 0), (0, 2)))
        result = sessions["group_consumer"].run(None, {HIDDEN: hidden, "group_a": group_a})[0]
        checks["compact_group_split_final"] = compare(result, expected)
        if not np.array_equal(result, expected):
            raise ValueError(f"CPU compact group split mismatch: {case}")
        chunks = []
        for first, count in ((0, 2), (2, 2), (4, 1)):
            states = []
            for index, state in enumerate(group_ready):
                channels = state if index == 0 else state.transpose(0, 1, 3, 2)
                padded = np.zeros((5, 64, 1, 152), np.float32)
                padded[..., :149] = channels.reshape(5, 64, 1, 149)
                states.append(padded[first:first + count].copy())
            chunks.append(sessions[f"original_order_attention_{count}"].run(None, {
                "head_q": head_q[:, first * 192:(first + count) * 192].copy(), "k": states[0], "v": states[1]})[0])
        original_a = np.pad(np.concatenate(chunks, axis=1), ((0, 0), (0, 0), (0, 0), (0, 2)))
        result = sessions["head_consumer"].run(None, {HIDDEN: hidden, "head_a": original_a})[0]
        checks["original_order_group_split_final"] = compare(result, expected)
        if not np.array_equal(result, expected):
            raise ValueError(f"CPU original-order compact group split mismatch: {case}")
        report["cases"][case] = checks
    (args.directory / "fp32_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args)
    else:
        if not args.onnx:
            parser.error("--onnx required")
        extract(args)


if __name__ == "__main__":
    main()
