from onnxconverter_common import auto_mixed_precision
from PIL import Image
import onnx
import numpy

input_path = "../models/lama_regular.onnx"
output_path = "../models/lama_regular_quantized.onnx"

# Assuming x is the input to the model

        
# image
pillow_img = Image.new("RGB", (512, 512))
pillow_img.paste(Image.open("../images/bird.jpg").resize((512, 512)))
input_data = numpy.float32(pillow_img) - numpy.array(
            [123.68, 116.78, 103.94], dtype=numpy.float32
        )
nhwc_data = numpy.expand_dims(input_data, axis=0)
nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard

# mask
mask_pillow_img = Image.new("1", (512, 512))
mask_pillow_img.paste(Image.open("../images/mask.png").resize((512, 512)))

mask_input_data = numpy.float32(mask_pillow_img)

mask_nchw_data = mask_input_data[None, None, :, :]

feed_dict = {'model': nchw_data, 'mask': mask_nchw_data}

model = onnx.load(input_path)
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, output_path)
