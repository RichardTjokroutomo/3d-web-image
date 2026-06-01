import onnx
from onnx import inliner

model_path = "../models/depth_anything_v2_vits.onnx"
output_path = "../models/depth_anything_v2_vits_inlined.onnx"

model = onnx.load(model_path)

model = inliner.inline_local_functions(model)

onnx.checker.check_model(model)
onnx.save(model, output_path)