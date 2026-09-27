"""Generate the tiny ONNX model used for the browser worker smoke check."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

input_tensor = helper.make_tensor_value_info("input_image", TensorProto.FLOAT, [1, 3, 2, 2])
output_tensor = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 2, 2])
node = helper.make_node("ReduceMean", ["input_image"], ["mask"], axes=[1], keepdims=1)
graph = helper.make_graph([node], "test-channel-mean", [input_tensor], [output_tensor])
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).resolve().parent.parent / "test" / "mean.onnx")
