import * as ort from "/js/ort/ort.mjs";

ort.env.wasm.wasmPaths = "/js/ort/";

/// arguments: string
/// retval: ONNX runtime session
export async function da_prepare_model(model_path){
    const sess = await ort.InferenceSession.create(model_path);
    return sess;
}

/// arguments: ONNX runtime session; OpenCV matrix (TODO: change this to ORT tensor)
/// retval: ONNX runtime tensor
export async function da_run_inference(ort_session, input_mat){
    // cv.blobFromImage returns a 4D Mat in NCHW format: [1, 3, H, W]
    const dims = input_mat.data32F; // float32 data from the blob
    const data = new Float32Array(dims.length);
    data.set(dims);
    const tensor = new ort.Tensor("float32", data, [1, 3, 518, 518]);
    const feeds = {l_x_: tensor};
    let result = await ort_session.run(feeds);
    input_mat.delete(); // clean up the cv.Mat
    return result;
}

/// arguments: openCV object; HTML image element
/// retval: openCV matrix TODO: this should return ORT tensor instead 
export async function da_pre_process(cv, input_src){
    const img_src = cv.imread(input_src);
    let img_dst = new cv.Mat();

    cv.cvtColor(img_src, img_dst, cv.COLOR_RGBA2RGB);
    let d_img = cv.blobFromImage(
        img_dst,
        1/255,
        new cv.Size(518, 518),
        new cv.Scalar(0.485, 0,456, 0.406), // FIXME: this is the mean value from imagenet. is this acceptable?
        false,
    );
    img_dst.delete();
    return d_img;
}


/// arguments: openCV object; ONNX runtime tensor; int; int
/// retval: HTML canvas element
export function da_post_process(cv, result, orig_width, orig_height){
    // Get the first output tensor (we don't know the exact output name)
    const outputTensors = Object.values(result);
    if (outputTensors.length === 0) {
        console.error("no output tensors in inference result");
        return null;
    }
    const tensor = outputTensors[0];
    const data = tensor.data; // Float32Array

    // Determine spatial dimensions from shape, squeezing batch/channel dims
    const shape = tensor.dims;
    let H, W;
    if (shape.length === 4) {
        // NCHW: [1, 1, H, W]
        H = shape[2];
        W = shape[3];
    } else if (shape.length === 3) {
        // [1, H, W] or [C, H, W]
        H = shape[1];
        W = shape[2];
    } else {
        // flat [H*W]
        H = Math.round(Math.sqrt(data.length));
        W = Math.round(data.length / H);
    }

    // Convert float32 data to cv.Mat (single-channel)
    let depth_mat = cv.matFromArray(H, W, cv.CV_32F, Array.from(data));

    // Normalize to 0-255 and convert to 8UC1
    let norm_mat = new cv.Mat();
    cv.normalize(depth_mat, norm_mat, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    depth_mat.delete();

    // Resize to original image dimensions
    let resized_mat = new cv.Mat();
    cv.resize(norm_mat, resized_mat, new cv.Size(orig_width, orig_height), 0, 0, cv.INTER_LINEAR);
    norm_mat.delete();

    // Convert grayscale to RGBA for canvas display
    let rgba_mat = new cv.Mat();
    cv.cvtColor(resized_mat, rgba_mat, cv.COLOR_GRAY2RGBA);
    resized_mat.delete();

    // Create canvas and display the depth map
    const canvas = document.createElement("canvas");
    canvas.width = orig_width;
    canvas.height = orig_height;
    cv.imshow(canvas, rgba_mat);
    rgba_mat.delete();

    return canvas;
}
