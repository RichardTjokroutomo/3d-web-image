import { da_prepare_model, da_run_inference, da_pre_process, da_post_process } from "./depth_segmentation.js";
import { ip_segment_into_layers, ip_prepare_model, ip_pre_process, ip_preprocess_mask, ip_run_inference, ip_post_process, ip_image_processing, ip_create_crop_mask } from "./inpaint.js";
import { util_create_canvas_from_image_element, util_resize_canvas, util_create_image_from_canvas_element } from "./utils.js";
import { setup_parallax_effect } from "./desktop_cursor.js";

/// wrapper to depth segmentation
/// arguments: ONNX runtime session; HTML image element
/// retval: HTML canvas element
async function obtain_depth_canvas(ort_sess, img_elem){
    const processed_input_image = await da_pre_process(cv, img_elem);
    const TIME_DA_0 = Date.now();
    const depth_layer = await da_run_inference(ort_sess, processed_input_image);
    const TIME_DA_1 = Date.now();
    const depth_canvas = da_post_process(cv, depth_layer, img_elem.naturalWidth, img_elem.naturalHeight);

    console.log(`time taken to run inference on depth anything model: ${TIME_DA_1 - TIME_DA_0} ms`);
    return depth_canvas;
}

/// wrapper to layer inpainting
/// arguments: ONNX runtime session; HTML canvas element; HTML canvas element; HTML canvas element
/// retval: HTML canvas element
async function inpaint_layers(ort_sess, img_elem_canvas, layer_i_canvas, layer_i_plus_one_canvas){
    // inpaint using the original image, masked by the next-farther layer
    const prep_img = ip_pre_process(cv, img_elem_canvas); // ORT tensor
    const layer_i_mask = ip_preprocess_mask(cv, layer_i_canvas, false); // ORT tensor, binary
    const layer_i_plus_one_mask = ip_preprocess_mask(cv, layer_i_plus_one_canvas, false); // ORT tensor, binary for model input
    const blend_mask = ip_preprocess_mask(cv, layer_i_plus_one_canvas, true); // ORT tensor, feathered for alpha blending
    const TIME_INPAINT_0 = Date.now();
    const result = await ip_run_inference(ort_sess, prep_img, layer_i_plus_one_mask); // ORT tensor
    const TIME_INPAINT_1 = Date.now();
    const canvas = ip_post_process(result, prep_img, blend_mask); // canvas (feathered blend)

    // create feathered combined mask for smooth crop boundary
    const crop_mask = ip_create_crop_mask(cv, layer_i_plus_one_mask, layer_i_mask); // ORT tensor
    const cropped_canvas = ip_image_processing(canvas, crop_mask); // canvas

    return [cropped_canvas, TIME_INPAINT_1 - TIME_INPAINT_0];
}

/// wrapper to generate 3d image
/// arguments: HTML image element; ONNX runtime session; ONNX runtime session
/// retval: [canvas, ...] (currently the length is fixed to 5)
async function generate_inpainted_layers(img_elem, da_ort_sess, lama_ort_sess){
    // 1. obtain depth image
    const depth_canvas = await obtain_depth_canvas(da_ort_sess, img_elem);
    
    // 2. segment into 5 layers (TODO: make this a user argument)
    const layers = ip_segment_into_layers(img_elem, depth_canvas);

    // 3. inpaint each layer
    const img_elem_canvas = util_create_canvas_from_image_element(img_elem);

    const inpainted_layers = []; // [canvas]
    let time_ellapsed = 0;
    for (let i = layers.length - 1; i >= 0; i--) {
        if (i === layers.length - 1) {
            inpainted_layers.push(util_resize_canvas(layers[i], 512, 512)); // farthest layer passes through unchanged
        } else {
            const last_inpainted_layer = inpainted_layers[inpainted_layers.length - 1];
            const cropped_canvas = await inpaint_layers(lama_ort_sess, img_elem_canvas, layers[i], last_inpainted_layer);
            time_ellapsed += cropped_canvas[1];
            inpainted_layers.push(cropped_canvas[0]);
        }
    }

    console.log(`total time taken to run inpainting model 5 times: ${time_ellapsed} ms`);
    return inpainted_layers;
}

/// the main function
/// ========================================================================================================
export async function lib_main(){
    const load_time_0 = Date.now();

    // prepare models
    const da_ort_sess = await da_prepare_model("./models/depth_anything_v2_vits.onnx");
    const lama_ort_sess = await ip_prepare_model("./models/lama_regular.onnx");

    const load_time_1 = Date.now();
    console.log(`time taken to prepare models: ${load_time_1 - load_time_0} ms`);

    document.getElementById('status').innerHTML = 'Models ready. ';

    // get elements
    const img_elem = document.getElementById("base_img");
    const button_elem = document.getElementById("process");
    const div_elem = document.getElementById("result");
    
    // TODO: this shouldn't be placed here. 
    button_elem.addEventListener("click", async (event) => {
        console.log("begin processing!");
        const TIME_BUTTON_0 = Date.now();
        if (!img_elem.src || img_elem.naturalWidth === 0) {
            alert("Please upload an image first.");
            return;
        }
        document.getElementById('status').innerHTML = 'Processing...';
        const inpainted_layers = await generate_inpainted_layers(img_elem, da_ort_sess, lama_ort_sess);

        const TIME_BUTTON_1 = Date.now();

        let layer_elements = [];
        for (let i = 0; i < inpainted_layers.length; i++){
            let layer_img = util_create_image_from_canvas_element(inpainted_layers[i]);
            layer_img.style.zIndex = -i;
            layer_img.style.position = "absolute";
            const imgWidth = inpainted_layers[i].width;
            const imgHeight = inpainted_layers[i].height;
            layer_img.style.left = `${(div_elem.clientWidth - imgWidth) / 2}px`;
            layer_img.style.top = `${(div_elem.clientHeight - imgHeight) / 2}px`;
            layer_img.style.transform = "scale(1.1)";
            div_elem.appendChild(layer_img);
            layer_elements.push(layer_img);
        }

        setup_parallax_effect(div_elem, layer_elements);

        const TIME_BUTTON_2 = Date.now();

        console.log(`total time taken to run inference: ${TIME_BUTTON_1 - TIME_BUTTON_0} ms`);
        console.log(`total time taken to create the elements and add cursor effect: ${TIME_BUTTON_2 - TIME_BUTTON_1} ms`);

        // TODO: remove these later.
        for (let j = 0; j < inpainted_layers.length; j++) {
            const layer_num = inpainted_layers.length - 1 - j; // reverse so layer_0 = closest
            const canvas = inpainted_layers[j];
            canvas.toBlob(function(blob) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "mask_" + layer_num + ".png";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, "image/png");
        }
    });

}

/// ========================================================================================================

window.lib_main = lib_main; // attach lib_main to window object
