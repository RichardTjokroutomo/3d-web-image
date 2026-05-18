import * as ort from "./ort/ort.mjs";
ort.env.wasm.wasmPaths = new URL("./js/ort/", document.baseURI).href;

/// arguments: string
/// retval: ONNX runtime session
export async function ip_prepare_model(model_path){
    const sess = await ort.InferenceSession.create(model_path);
        return sess;
}

/// arguments: openCV instance; HTML canvas element; bool
/// retval: ONNX runtime tensor
export function ip_pre_process(cv, img_canvas, dilate_layer){
    const SZ = 512;
    // quantization params from metadata: scale=1.5259e-05, zero_point=0
    // uint16 = float_val / scale  (equivalent to float_val * 65535)
    const QSCALE = 1.5259021893143654e-05;

    // --- image ---
    const img_mat = cv.imread(img_canvas);
    let img_rgb = new cv.Mat();
    cv.cvtColor(img_mat, img_rgb, cv.COLOR_RGBA2RGB);
    let img_resized = new cv.Mat();
    cv.resize(img_rgb, img_resized, new cv.Size(SZ, SZ));
    img_mat.delete();
    img_rgb.delete();

    const HW = SZ * SZ;
    let img_flat = new Uint16Array(3 * HW);
    for (let i = 0; i < HW; i++) {
        const base = i * 3;
        const ch_base = i; // spatial index per channel
        img_flat[0 * HW + ch_base] = Math.round((img_resized.data[base]     / 255) / QSCALE);
        img_flat[1 * HW + ch_base] = Math.round((img_resized.data[base + 1] / 255) / QSCALE);
        img_flat[2 * HW + ch_base] = Math.round((img_resized.data[base + 2] / 255) / QSCALE);
    }
    img_resized.delete();

    if (dilate_layer) { // apply gaussian blur. TODO: convert RGB to grayscale first so we don't need to run the operation 3x per pixel.
        const ksize = 5;
        const ch0 = new cv.Mat(SZ, SZ, cv.CV_16U);
        const ch1 = new cv.Mat(SZ, SZ, cv.CV_16U);
        const ch2 = new cv.Mat(SZ, SZ, cv.CV_16U);
        ch0.data.set(img_flat.subarray(0, HW));
        ch1.data.set(img_flat.subarray(HW, 2 * HW));
        ch2.data.set(img_flat.subarray(2 * HW, 3 * HW));

        const b0 = new cv.Mat();
        const b1 = new cv.Mat();
        const b2 = new cv.Mat();
        cv.GaussianBlur(ch0, b0, new cv.Size(ksize, ksize), 0);
        cv.GaussianBlur(ch1, b1, new cv.Size(ksize, ksize), 0);
        cv.GaussianBlur(ch2, b2, new cv.Size(ksize, ksize), 0);

        img_flat.set(b0.data, 0);
        img_flat.set(b1.data, HW);
        img_flat.set(b2.data, 2 * HW);

        ch0.delete(); ch1.delete(); ch2.delete();
        b0.delete();  b1.delete();  b2.delete();
    }

    const img_tensor = new ort.Tensor("uint16", img_flat, [1, 3, SZ, SZ]);

    return img_tensor;
}

/// arguments: openCV instance; HTML canvas element
/// retval: ONNX runtime tensor
export function ip_preprocess_mask(cv, canvas){ // converts RGB to mask (1 channel binary value; 0 or 255)
    const SZ = 512;
    const QSCALE = 1.5259021893143654e-05;
    const W = canvas.width;
    const H = canvas.height;

    const mat = cv.imread(canvas);

    // 1. if alpha is 0, set RGB to 0; otherwise set RGB to 255
    const total = W * H;
    for (let i = 0; i < total; i++) {
        const base = i * 4;
        if (mat.data[base + 3] === 0) {
            mat.data[base] = 0;
            mat.data[base + 1] = 0;
            mat.data[base + 2] = 0;
        } else {
            mat.data[base] = 255;
            mat.data[base + 1] = 255;
            mat.data[base + 2] = 255;
        }
    }

    // 2. convert to grayscale (all channels are identical after the step above)
    let gray = new cv.Mat();
    cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    mat.delete();

    // 3. resize
    let resized = new cv.Mat();
    cv.resize(gray, resized, new cv.Size(SZ, SZ));
    gray.delete();

    // 4. uint8 to uint16
    const quantized = new Uint16Array(SZ * SZ);
    for (let i = 0; i < SZ * SZ; i++) {
        quantized[i] = Math.round((resized.data[i] / 255) / QSCALE);
    }
    resized.delete();

    return new ort.Tensor("uint16", quantized, [1, 1, SZ, SZ]);
}

/// arguments: ONNX runtime session; ONNX runtime tensor (1, 3, 512, 512); ONNX runtime tensor (1, 1, 512, 512)
/// retval: ONNX runtime tensor (1, 3, 512, 512)
export async function ip_run_inference(ort_session, image_tensor, mask_tensor){
    const feeds = { image: image_tensor, mask: mask_tensor };
    let result = await ort_session.run(feeds);
    return result;
}

/// arguments: ONNX runtime tensor (1, 3, 512, 512)
/// retval: HTML canvas element
export function ip_post_process(result){
    const outputTensors = Object.values(result);
    if (outputTensors.length === 0) {
        console.error("no output tensors in inference result");
        return null;
    }
    const tensor = outputTensors[0];
    const data = tensor.data; //uint16 arr

    const QSCALE = 1.5259021893143654e-05;
    const SZ = 512;
    const HW = SZ * SZ;

    const canvas = document.createElement("canvas");
    canvas.width = SZ;
    canvas.height = SZ;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(SZ, SZ);
    const pixels = imageData.data;

    for (let i = 0; i < HW; i++) {
        const j = i * 4;
        const r = Math.min(255, Math.max(0, Math.round(data[i] * QSCALE * 255)));
        const g = Math.min(255, Math.max(0, Math.round(data[HW + i] * QSCALE * 255)));
        const b = Math.min(255, Math.max(0, Math.round(data[2 * HW + i] * QSCALE * 255)));
        let alpha_val = 0;
        if (r + g + b > 0) {
            alpha_val = 255;
        }
        pixels[j]     = r;
        pixels[j + 1] = g;
        pixels[j + 2] = b;
        pixels[j + 3] = alpha_val;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

/// arguments: HTML image element; HTML canvas element
/// retval: [HTML canvas element, ...]
export function ip_segment_into_layers(orig_img_elem, depth_canvas){
    console.log("entering ip_segment_into_layers()!")
    const numLayers = 5;
    const W = depth_canvas.width;
    const H = depth_canvas.height;
    console.log("W: " + W);

    // draw original image onto an offscreen canvas at the depth map size
    const origCanvas = document.createElement("canvas");
    origCanvas.width = W;
    origCanvas.height = H;
    const origCtx = origCanvas.getContext("2d");
    origCtx.drawImage(orig_img_elem, 0, 0, W, H);
    const origData = origCtx.getImageData(0, 0, W, H);

    // read depth pixel data
    const depthCtx = depth_canvas.getContext("2d");
    const depthData = depthCtx.getImageData(0, 0, W, H);

    const layers = [];
    const bandSize = 256 / numLayers;

    for (let layer = 0; layer < numLayers; layer++) {
        const minDepth = Math.floor(layer * bandSize);
        const maxDepth = (layer === numLayers - 1) ? 255 : Math.floor((layer + 1) * bandSize) - 1;

        const canvas = document.createElement("canvas");
        canvas.width = W;
        canvas.height = H;
        const ctx = canvas.getContext("2d");
        const imageData = ctx.createImageData(W, H);
        const pixels = imageData.data;

        for (let i = 0; i < W * H; i++) {
            const j = i * 4;
            const depth = depthData.data[j]; // R channel of grayscale depth
            if (depth >= minDepth && depth <= maxDepth) {
                pixels[j]     = origData.data[j];
                pixels[j + 1] = origData.data[j + 1];
                pixels[j + 2] = origData.data[j + 2];
                pixels[j + 3] = 255;
            } else {
                pixels[j + 3] = 0; // transparent
            }
        }

        ctx.putImageData(imageData, 0, 0);

        layers.push(canvas);

        // download for testing purposes. TODO: remove this later.
        canvas.toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "layer_" + layer + ".png";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, "image/png");
    }

    return layers;
}

// arguments: canvas, ORT tensor [1, 1, 512, 512], ORT tensor [1, 1, 512, 512]
// retval: canvas
export function ip_image_processing(canvas, layer_i_plus_one_mask, layer_i_mask) {
    // Read pixel data from the inpainted result canvas (512x512)
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const imageData = ctx.getImageData(0, 0, W, H);
    const pixels = imageData.data;

    const mask1_data = layer_i_plus_one_mask.data;
    const mask2_data = layer_i_mask.data;

    for (let i = 0; i < W * H; i++) {
        if (mask1_data[i] === 0 && mask2_data[i] === 0) {
            const j = i * 4;
            pixels[j]     = 0;
            pixels[j + 1] = 0;
            pixels[j + 2] = 0;
            pixels[j + 3] = 0;
        }
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
}