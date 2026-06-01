import onnx
from onnxconverter_common import float16

model_path = "../models/lama_regular.onnx"
output_path = "../models/lama_regular_quantized.onnx"

model_path = "../models/depth_anything_v2_vits_inlined.onnx"
output_path = "../models/depth_anything_v2_vits_quantized.onnx"

model = onnx.load(model_path)
print("model loaded!")

all_ops = sorted({node.op_type for node in model.graph.node})
print("all_ops loaded! number of ops: " + str(len(all_ops)))

#op_block_list = [op for op in all_ops if op != "Conv"]
op_block_list = []
op_block_list = ["Resize", "Add", "Sub", "Mul", "Div", "MatMul", "Gemm", "LayerNormalization", "Softmax", "ReduceMean", "Pow", "Sqrt", "Concat", "Slice"] 
#op_block_list = ["BatchNormalization", "Output",]
print("op block list created!")

#fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-10, max_finite_val=1e5, keep_io_types=True,
#                         disable_shape_infer=False, op_block_list=None, node_block_list=None)

fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-11, max_finite_val=1e4, keep_io_types=True,
                         disable_shape_infer=False, op_block_list=op_block_list, node_block_list=None)


onnx.save(fp16, output_path)

# run type inference
new_model = onnx.load(output_path)
new_model = onnx.shape_inference.infer_shapes(new_model)
del new_model.graph.value_info[:]

onnx.checker.check_model(new_model)

onnx.save(new_model, output_path)