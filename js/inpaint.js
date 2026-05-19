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
export function ip_pre_process(cv, img_canvas){
    const SZ = 512;

    // --- image ---
    const img_mat = cv.imread(img_canvas);
    let img_rgb = new cv.Mat();
    cv.cvtColor(img_mat, img_rgb, cv.COLOR_RGBA2RGB);
    let img_resized = new cv.Mat();
    cv.resize(img_rgb, img_resized, new cv.Size(SZ, SZ));
    img_mat.delete();
    img_rgb.delete();

    const HW = SZ * SZ;
    let img_flat = new Float32Array(3 * HW);
    for (let i = 0; i < HW; i++) {
        const base = i * 3;
        const ch_base = i; // spatial index per channel
        img_flat[0 * HW + ch_base] = img_resized.data[base]     / 255;
        img_flat[1 * HW + ch_base] = img_resized.data[base + 1] / 255;
        img_flat[2 * HW + ch_base] = img_resized.data[base + 2] / 255;
    }
    img_resized.delete();

    return new ort.Tensor("float32", img_flat, [1, 3, SZ, SZ]);
}

/// arguments: openCV instance; HTML canvas element
/// retval: ONNX runtime tensor
export function ip_preprocess_mask(cv, canvas, dilate_layer){ // converts RGB to mask (1 channel binary value; 0 or 255)
    const SZ = 512;
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

    if (dilate_layer) { // TODO: replace this dilation logic elsewhere.
        const kernel = cv.Mat.ones(7, 7, cv.CV_8U);
        const dilated = new cv.Mat();
        cv.dilate(resized, dilated, kernel);
        kernel.delete();

        const blurred = new cv.Mat();
        cv.GaussianBlur(dilated, blurred, new cv.Size(31, 31), 10);
        dilated.delete();

        resized.delete();
        resized = blurred;
    }

    // 5. normalize to [0, 1] float32 range
    const normalized = new Float32Array(SZ * SZ);
    for (let i = 0; i < SZ * SZ; i++) {
        normalized[i] = resized.data[i] / 255;
    }
    resized.delete();

    return new ort.Tensor("float32", normalized, [1, 1, SZ, SZ]);
}

/// arguments: ONNX runtime session; ONNX runtime tensor (1, 3, 512, 512); ONNX runtime tensor (1, 1, 512, 512)
/// retval: ONNX runtime tensor (1, 3, 512, 512)
export async function ip_run_inference(ort_session, image_tensor, mask_tensor){
    const feeds = { model: image_tensor, mask: mask_tensor };
    let result = await ort_session.run(feeds);
    return result;
}

/// arguments: ONNX runtime tensor (1, 3, 512, 512)
/// retval: HTML canvas element
export function ip_post_process(result, original_img, mask){
    const SZ = 512;
    const HW = SZ * SZ;

    // result
    const result_tensor = Object.values(result);
    if (result_tensor.length === 0) {
        console.error("result_tensor is empty!");
        return null;
    }
    const tensor = result_tensor[0];
    const result_data = tensor.data; // float32 arr in [0, 1]

    // original image
    const original_data = original_img.data; // float32 arr in [0, 1]

    // mask
    const maskData = mask.data; // float32 arr in [0, 1], single channel

    // retval
    const canvas = document.createElement("canvas");
    canvas.width = SZ;
    canvas.height = SZ;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(SZ, SZ);
    const pixels = imageData.data;

    for (let i = 0; i < HW; i++) {
        const j = i * 4;
        const res_r = Math.min(255, Math.max(0, Math.round(result_data[i] * 255)));
        const res_g = Math.min(255, Math.max(0, Math.round(result_data[HW + i] * 255)));
        const res_b = Math.min(255, Math.max(0, Math.round(result_data[2 * HW + i] * 255)));

        const orig_r = Math.min(255, Math.max(0, Math.round(original_data[i] * 255)));
        const orig_g = Math.min(255, Math.max(0, Math.round(original_data[HW + i] * 255)));
        const orig_b = Math.min(255, Math.max(0, Math.round(original_data[2 * HW + i] * 255)));

        const alpha = Math.min(1, Math.max(0, maskData[i]));

        const fg = Math.round(alpha * 255);

        // output = result * alpha + original * (1 - alpha)
        // pixels[j]     = Math.min(255, Math.round(res_r * alpha + orig_r * (1 - alpha)));
        // pixels[j + 1] = Math.min(255, Math.round(res_g * alpha + orig_g * (1 - alpha)));
        // pixels[j + 2] = Math.min(255, Math.round(res_b * alpha + orig_b * (1 - alpha)));
        // pixels[j + 3] = 255;

        pixels[j]     = Math.min(255, Math.round(res_r));
        pixels[j + 1] = Math.min(255, Math.round(res_g));
        pixels[j + 2] = Math.min(255, Math.round(res_b));
        pixels[j + 3] = 255;
    }

    // for (let i = 0; i < HW; i++) {
    //     const j = i * 4;
    //     const r = Math.min(255, Math.max(0, Math.round(result_data[i] * QSCALE * 255)));
    //     const g = Math.min(255, Math.max(0, Math.round(result_data[HW + i] * QSCALE * 255)));
    //     const b = Math.min(255, Math.max(0, Math.round(result_data[2 * HW + i] * QSCALE * 255)));
    //     let alpha_val = 0;
    //     if (r + g + b > 0) {
    //         alpha_val = 255;
    //     }
    //     pixels[j]     = r;
    //     pixels[j + 1] = g;
    //     pixels[j + 2] = b;
    //     pixels[j + 3] = alpha_val;
    // }

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

// arguments: canvas, ORT tensor [1, 1, 512, 512] (feathered crop mask)
// retval: canvas
export function ip_image_processing(canvas, crop_mask) {
    // read pixel data from the inpainted result canvas (512x512)
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const imageData = ctx.getImageData(0, 0, W, H);
    const pixels = imageData.data;

    const mask_data = crop_mask.data; // float32 in [0, 1]

    for (let i = 0; i < W * H; i++) {
        const keep = mask_data[i]; // smooth transition at crop boundary
        const j = i * 4;
        pixels[j]     = Math.round(pixels[j]     * keep);
        pixels[j + 1] = Math.round(pixels[j + 1] * keep);
        pixels[j + 2] = Math.round(pixels[j + 2] * keep);
        pixels[j + 3] = Math.round(pixels[j + 3] * keep);
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

// arguments: openCV instance; ORT tensor [1, 1, 512, 512]; ORT tensor [1, 1, 512, 512]
// retval: ORT tensor [1, 1, 512, 512]
export function ip_create_crop_mask(cv, mask_a, mask_b) {
    const SZ = 512;

    // combine both binary masks (1 if either has content)
    const combined = new Uint8Array(SZ * SZ);
    for (let i = 0; i < SZ * SZ; i++) {
        combined[i] = (mask_a.data[i] > 0 || mask_b.data[i] > 0) ? 255 : 0;
    }

    const mat = new cv.Mat(SZ, SZ, cv.CV_8UC1);
    mat.data.set(combined);

    // dilate to push the crop boundary outward
    const kernel = cv.Mat.ones(7, 7, cv.CV_8U);
    const dilated = new cv.Mat();
    cv.dilate(mat, dilated, kernel);
    kernel.delete();
    mat.delete();

    // blur for smooth alpha transition at outer edge
    const blurred = new cv.Mat();
    cv.GaussianBlur(dilated, blurred, new cv.Size(1, 1), 10);
    dilated.delete();

    const feathered = new Float32Array(SZ * SZ);
    for (let i = 0; i < SZ * SZ; i++) {
        feathered[i] = blurred.data[i] / 255;
    }
    blurred.delete();

    return new ort.Tensor("float32", feathered, [1, 1, SZ, SZ]);
}