#!/usr/bin/env python3
"""Small graph guards, independent of private checkpoint or NPU artifacts."""

import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from export_smolvla_spatial_consumer import spatial_consumer
from export_smolvla_stage_spatial_consumer import stage_spatial_consumer
from extract_smolvla_vision_probes import native_interfaces


def toy_consumer():
    rng = np.random.default_rng(7)
    parameters = {
        "scale": np.ones(768, dtype=np.float32), "offset": np.zeros(768, dtype=np.float32),
        "fc1_weight": rng.normal(0, 0.02, (768, 2)).astype(np.float32),
        "fc2_weight": rng.normal(0, 0.02, (2, 768)).astype(np.float32),
        "fc2_bias": np.zeros(768, dtype=np.float32),
    }
    graph = helper.make_graph([
        helper.make_node("Add", ["hidden", "attention"], ["residual"], name="residual"),
        helper.make_node("LayerNormalization", ["residual", "scale", "offset"], ["normalized"], name="norm", axis=-1),
        helper.make_node("MatMul", ["normalized", "fc1_weight"], ["fc1"], name="toy/mlp/fc1/MatMul"),
        helper.make_node("MatMul", ["fc1", "fc2_weight"], ["fc2"], name="toy/mlp/fc2/MatMul"),
        helper.make_node("Add", ["fc2", "fc2_bias"], ["biased"], name="toy/mlp/fc2/Add"),
        helper.make_node("Add", ["residual", "biased"], ["output"], name="final_residual"),
    ], "toy_consumer", [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1024, 768])
                        for name in ("hidden", "attention")],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1024, 768])],
        [numpy_helper.from_array(array, name) for name, array in parameters.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return native_interfaces(model, [0, 1], [0])


class GeometryGuards(unittest.TestCase):
    def test_invalid_grid_rejected(self):
        source = toy_consumer()
        for transform in (spatial_consumer, stage_spatial_consumer):
            for height, width in ((32, 16), (0, 1024), (-1, -1024)):
                with self.assertRaises(ValueError):
                    transform(source, height, width)

    def test_stage_cut_remains_inside_one_graph(self):
        source = toy_consumer()
        before = source.SerializeToString()
        candidate = stage_spatial_consumer(source, 32, 32)
        self.assertEqual(source.SerializeToString(), before)
        self.assertEqual([value.SerializeToString() for value in source.graph.input],
                         [value.SerializeToString() for value in candidate.graph.input])
        names = [node.name for node in candidate.graph.node]
        self.assertLess(names.index("norm"), names.index("__stage_norm_to_spatial"))
        self.assertLess(names.index("__stage_norm_to_spatial"), names.index("toy/mlp/fc1/MatMul"))
        self.assertLess(names.index("__stage_mlp_to_flat"), names.index("final_residual"))
        self.assertEqual([value.SerializeToString() for value in source.graph.initializer],
                         [value.SerializeToString() for value in candidate.graph.initializer[:-2]])

    def test_whole_and_stage_shapes_preserve_small_reference(self):
        source = toy_consumer()
        rng = np.random.default_rng(13)
        arrays = [rng.normal(0, 0.2, (1, 768, 1, 1024)).astype(np.float32) for _ in source.graph.input]
        expected = ReferenceEvaluator(source).run(None, dict(zip([value.name for value in source.graph.input], arrays)))[0]
        for transform in (spatial_consumer, stage_spatial_consumer):
            candidate = transform(source, 32, 32)
            values = arrays if transform is stage_spatial_consumer else [array.reshape(1, 768, 32, 32) for array in arrays]
            output = ReferenceEvaluator(candidate).run(None, dict(zip([value.name for value in candidate.graph.input], values)))[0]
            self.assertTrue(np.array_equal(output.reshape(expected.shape).view(np.uint32), expected.view(np.uint32)))

    def test_non_channel_normalization_rejected(self):
        source = toy_consumer()
        norm = next(node for node in source.graph.node if node.op_type == "LayerNormalization")
        next(attr for attr in norm.attribute if attr.name == "axis").i = 1
        for transform in (spatial_consumer, stage_spatial_consumer):
            with self.assertRaises(ValueError):
                transform(source, 32, 32)


if __name__ == "__main__":
    unittest.main()
