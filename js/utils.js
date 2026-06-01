
// arguments: HTML canvas element; string 
// retval: HTML image element
export function util_create_image_from_canvas_element(canvas_elem, mime_type = "image/png"){
    const img = new Image();
    img.src = canvas_elem.toDataURL(mime_type);
    return img;
}

// arguments: HTML image element
// retval: HTML canvas element
export function util_create_canvas_from_image_element(img_elem){
    const canvas = document.createElement("canvas");
    canvas.width = img_elem.width;
    canvas.height = img_elem.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img_elem, 0, 0, img_elem.width, img_elem.height);

    return canvas;
}

// arguments: HTML canvas element; int; int
// retval: HTML canvas element
export function util_resize_canvas(input_canvas, width, height){
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(input_canvas, 0, 0, width, height)

    return canvas;
}

/// arguments: ORT tensor
/// retval: cv.Mat
export function util_ort_tensor_to_cv(tensor){
    const data = tensor.data;
    const dims = tensor.dims;

    // Parse spatial dimensions and number of channels from tensor shape
    let H, W, C;
    if (dims.length === 4) {
        // NCHW: [N, C, H, W]
        C = dims[1];
        H = dims[2];
        W = dims[3];
    } else if (dims.length === 3) {
        // CHW: [C, H, W]
        C = dims[0];
        H = dims[1];
        W = dims[2];
    } else if (dims.length === 2) {
        // HW: [H, W]
        H = dims[0];
        W = dims[1];
        C = 1;
    } else {
        throw new Error("unsupported tensor dims length: " + dims.length);
    }

    // Determine OpenCV Mat type based on data type and channel count
    const isFloat = data instanceof Float32Array;
    let cvType;
    if (isFloat) {
        cvType = C === 3 ? cv.CV_32FC3 : C === 4 ? cv.CV_32FC4 : cv.CV_32F;
    } else {
        cvType = C === 3 ? cv.CV_8UC3 : C === 4 ? cv.CV_8UC4 : cv.CV_8U;
    }

    const mat = new cv.Mat(H, W, cvType);
    const total = H * W; // pixels per channel

    if (C === 1) {
        // Single channel: direct flat copy
        const dst = isFloat ? mat.data32F : mat.data;
        for (let i = 0; i < total; i++) {
            dst[i] = data[i];
        }
    } else {
        // Multi-channel: convert from NCHW to HWC layout
        // NCHW: channel data stored in contiguous blocks: [ch0][ch1][ch2]...
        // HWC:  channels interleaved per pixel: [p0_c0, p0_c1, p0_c2, p1_c0, ...]
        const dst = isFloat ? mat.data32F : mat.data;
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                for (let c = 0; c < C; c++) {
                    dst[(h * W + w) * C + c] = data[c * total + h * W + w];
                }
            }
        }
    }

    return mat;
}
