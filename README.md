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

To run the demo, we want to launch a local server so the browser can make `http` requests to load the files. This mimics real world scenario where the static files are stored in remote server. 

To do this, run:

```bash
python3 -m http.server 8000
```

And open the page on `localhost:8000`. 


If you want to load the files via `file` protocol, you need to disable web security. Below are the commands to disable web security on Chrome:

On Windows:
```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --disable-gpu --user-data-dir="C:\chromeTemp"
```

On Linux:
```
google-chrome --disable-web-security --user-data-dir="/temp/dev/"
```

And open `index.html` on the browser.

Afterwards, wait until the text "Models ready" is visible before you click the process button.

---

## Limitations
Below is a list of limitations as of 17/5/2026:

1. Despite the usage of the terms "framework" & "library", currently the code only works for sample `index.html` provided. This is not because of the limitations within JavaScript itself but rather the lack of development time.
2. Inference runs very slowly. This can hopefully be refactored by using WebNN and models with fewer parameters.
3. Currently, only simple dilation is being implemented. This should probably be updated in the future.
4. `OpenCV.js` cannot be imported as ES Module. Instead, we need to import it with `<script>` and wait for it to fully load before our framework can be loaded:

```html
<script type="text/javascript">
    var Module = {
    // https://emscripten.org/docs/api_reference/module.html#Module.onRuntimeInitialized
    onRuntimeInitialized() {
        lib_main(); // wrapper function to our JS library
    }
    };
</script>
<script async src="js/opencv.js" type="text/javascript"></script>
<script src="js/lib.js" type="module"></script>
```
 Although it causes no noticeable performance loss, this solution is not elegant at all, and an alternate solution is needed before this project is ready for commercial use. 

---

## Future works

1. Update dilation's impl.
2. Use sliding window to do depth estimation. This way, we don't have to suffer from resolution loss due to downsampling. 
3. Find a better way to import `OpenCV.js`.
4. Refactor the code so it can be plugged into any frontend project using `<script>`. The idea is that all target `<image>` elements must be wrapped in a div container with a predefined class or ID.
5. Refactor the code. Currently the code exports function everywhere, it may be a good idea to wrap everything into class instead.
6. Add code that uses [Javascript's gyroscope API](https://developer.mozilla.org/en-US/docs/Web/API/Gyroscope) so the 3D effect can be seen on mobile.