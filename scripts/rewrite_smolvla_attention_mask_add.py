#!/usr/bin/env python3
"""Replace constant boolean attention Where with an FP16-safe additive mask."""

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def constant_value(node):
    if node is None or node.op_type != "Constant":
        return None
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(attribute.t)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = onnx.load(args.input, load_external_data=True)
    producers = {output: node for node in model.graph.node for output in node.output}
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Where" or len(node.input) != 3:
            continue
        mask = constant_value(producers.get(node.input[0]))
        sentinel = constant_value(producers.get(node.input[2]))
        if (
            mask is None
            or mask.dtype != np.bool_
            or sentinel is None
            or sentinel.size != 1
            or float(sentinel.item()) > -1e30
        ):
            continue
        additive = np.where(mask, 0.0, -10000.0).astype(np.float32)
        initializer_name = f"smolvla_additive_attention_mask_{changed}"
        model.graph.initializer.append(numpy_helper.from_array(additive, initializer_name))
        original_logits = node.input[1]
        node.op_type = "Add"
        del node.input[:]
        node.input.extend([original_logits, initializer_name])
        changed += 1
    if not changed:
        raise RuntimeError("No constant attention mask Where found")
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"Rewrote {changed} attention masks")


if __name__ == "__main__":
    main()
