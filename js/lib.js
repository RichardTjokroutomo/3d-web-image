import { da_prepare_model, da_run_inference, da_pre_process, da_post_process } from "./depth_segmentation.js";
import { ip_segment_into_layers, ip_prepare_model, ip_pre_process, ip_preprocess_mask, ip_run_inference, ip_post_process, ip_image_processing } from "./inpaint.js";
import { util_create_canvas_from_image_element, util_resize_canvas, util_create_image_from_canvas_element } from "./utils.js";
import { setup_parallax_effect } from "./desktop_cursor.js";

console.log("function initialized");

/// wrapper to depth segmentation
/// arguments: ONNX runtime session; HTML image element
/// retval: HTML canvas element
async function obtain_depth_canvas(ort_sess, img_elem){
    const processed_input_image = await da_pre_process(cv, img_elem);
    const depth_layer = await da_run_inference(ort_sess, processed_input_image);
    const depth_canvas = da_post_process(cv, depth_layer, img_elem.naturalWidth, img_elem.naturalHeight);

    return depth_canvas;
}

/// wrapper to layer inpainting
/// arguments: ONNX runtime session; HTML canvas element; HTML canvas element; HTML canvas element
/// retval: HTML canvas element
async function inpaint_layers(ort_sess, img_elem_canvas, layer_i_canvas, layer_i_plus_one_canvas){
    // inpaint using the original image, masked by the next-farther layer
    const prep_img = ip_pre_process(cv, img_elem_canvas); // ORT tensor
    const layer_i_mask = ip_preprocess_mask(cv, layer_i_canvas, false); // ORT tensor
    const layer_i_plus_one_mask = ip_preprocess_mask(cv, layer_i_plus_one_canvas, false); // ORT tensor
    const mask = ip_preprocess_mask(cv, layer_i_plus_one_canvas, true); // ORT tensor
            
    const result = await ip_run_inference(ort_sess, prep_img, mask); // ORT tensor
    const canvas = ip_post_process(result); // canvas

    const cropped_canvas = ip_image_processing(canvas, layer_i_plus_one_mask, layer_i_mask); // canvas

    return cropped_canvas;
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
    for (let i = layers.length - 1; i >= 0; i--) {
        if (i === layers.length - 1) {
            inpainted_layers.push(util_resize_canvas(layers[i], 512, 512)); // farthest layer passes through unchanged
        } else {
            const last_inpainted_layer = inpainted_layers[inpainted_layers.length - 1];
            const cropped_canvas = await inpaint_layers(lama_ort_sess, img_elem_canvas, layers[i], last_inpainted_layer);
            inpainted_layers.push(cropped_canvas);
        }
    }
    console.log("inpainting complete, created " + inpainted_layers.length + " layers");

    return inpainted_layers;
}

/// the main function
/// ========================================================================================================
export async function lib_main(){
    console.log("begin execution!");

    // prepare models
    const da_ort_sess = await da_prepare_model("./models/depth_anything_v2_vitl.onnx");
    const lama_ort_sess = await ip_prepare_model("./models/lama_merged.onnx");

    document.getElementById('status').innerHTML = 'Models ready. ';
    console.log("models ready");

    const img_elem = document.getElementById("base_img");
    const button_elem = document.getElementById("process");
    const div_elem = document.getElementById("result");
    
    // TODO: this shouldn't be placed here. 
    button_elem.addEventListener("click", async (event) => {
        if (!img_elem.src || img_elem.naturalWidth === 0) {
            alert("Please upload an image first.");
            return;
        }
        console.log("processing image!");
        document.getElementById('status').innerHTML = 'Processing...';
        const inpainted_layers = await generate_inpainted_layers(img_elem, da_ort_sess, lama_ort_sess);

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

        // TODO: remove these later.
        console.log("downloading inpainted layers");
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
