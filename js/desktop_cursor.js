
/// attribute: HTML div element
/// retval: None
export function setup_parallax_effect(div_elem, layer_elements){
    div_elem.addEventListener('mousemove', (event) => {
        layer_elements.forEach(img => {
            if (img) img.style.transition = 'none';
        });
        updateParallax(div_elem, layer_elements, event.clientX, event.clientY);
    });

    div_elem.addEventListener('mouseleave', () => {
        layer_elements.forEach(img => {
            if (img) {
                img.style.transition = 'transform 0.5s cubic-bezier(0.25, 0.8, 0.25, 1)';
                img.style.transform = 'translateX(0) translateY(0) scale(0.9)';
            }
        });
        setTimeout(() => {
            layer_elements.forEach(img => {
                if (img) img.style.transition = '';
            });
        }, 600);
    });
}

/// attributes: HTML div element; int; int
/// retval: None
function updateParallax(div_elem, layer_elements, cursor_x, cursor_y) {
    if (!div_elem) return;

    const div_elem_rect = div_elem.getBoundingClientRect();
    const x_center = div_elem_rect.left + div_elem_rect.width / 2;
    const distance_from_center_x = cursor_x - x_center;

    const y_center= div_elem_rect.top + div_elem_rect.height / 2;
    const distance_from_center_y = cursor_y - y_center;

    // closer layers (lower index) move less, farther layers (higher index) move more
    //const parallaxFactors = [0.30, 0.22, 0.15, 0.08, 0.04];
    //const parallaxFactors = [0.004, 0.008, 0.015, 0.022, 0.03];
    // const parallaxFactors = [0.07, 0.075, 0.08, 0.09, 0.1];
    // const parallaxFactors = [0.35, 0.375, 0.4, 0.45, 0.5];
    const parallaxFactors = [0.07, 0.075, 0.08, 0.09, 0.11];

    layer_elements.forEach((img, index) => {
        if (img) {
            const translation_x = distance_from_center_x * parallaxFactors[index];
            const translation_y = distance_from_center_y* parallaxFactors[index];
            img.style.transform = `translateX(${translation_x}px) translateY(${translation_y}px) scale(0.9)`;
        }
    });
}