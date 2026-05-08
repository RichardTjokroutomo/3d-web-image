
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
