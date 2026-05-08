# 3D JS Framework
This is a JS library developed to convert an `<image>` into 3D pictures that can move based on the cursor's movements.

---

## How to use
You need to download the following models in `.onnx` format and put them under `models/` directory:
1. `depth_anything_v2_vitl.onnx`: [link](https://github.com/fabio-sim/Depth-Anything-ONNX/releases)
2. `lama_dilated`: [link](https://aihub.qualcomm.com/models/lama_dilated)

For `lama_dilated`, if you're downloading from the download link, notice that Qualcomm separates the parameters from the model itself (for reasons I don't know). Therefore, you need to merge them into 1 `.onnx` file. To do that, extract the compressed model into `models/` (ensure the extracted directory is `lama_dilated-onnx-w8a16`). Afterwards, run:

```bash
python3 utils/onnx_converter.py
```

Currently, the demo runs on an HTTP server. To run the demo:

```bash
python3 -m http.server 8000
```

And open the page on `localhost:8000`.

---

## Limitations
Below is a list of limitations as of 17/5/2026:

1. Despite the usage of the terms "framework" & "library", currently the code only works for `index.html`. This is not because of the limitations within JavaScript itself but rather the lack of development time.
2. Inference runs very slowly. This can hopefully be refactored by using WebNN and models with fewer parameters.
3. Currently, only simple dilation is being implemented. This should probably be updated in the future.
4. `OpenCV.js` cannot be imported as ES Module. Instead, we need to import it with `<script>` and wait for it to fully load before our framework can be loaded. Although it has no noticeable performance penalty, this solution is not elegant at all, and an alternate solution is needed before this project is ready for commercial use. 

---

## Future works

1. Fix dilation's impl.
2. Use sliding window to do depth estimation. This way, we don't have to suffer from resolution loss due to downsampling. 
3. Find a better way to import `OpenCV.js`.
4. Refactor the code so it can be plugged into any frontend project using `<script>`. The idea is that all target `<image>` elements must be wrapped in a div container with a predefined class or ID.
5. Refactor the code. Currently the code exports function everywhere, it may be a good idea to wrap everything into class instead.
6. Add code that uses [Javascript's gyroscope API](https://developer.mozilla.org/en-US/docs/Web/API/Gyroscope) so the 3D effect can be seen on mobile.