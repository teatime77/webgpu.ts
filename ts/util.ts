namespace webgputs {

export function initContext(g_device : GPUDevice, canvas: HTMLCanvasElement, alpha_mode : GPUCanvasAlphaMode) : [GPUCanvasContext, GPUTextureFormat] {
    const context = canvas.getContext('webgpu') as GPUCanvasContext;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    const devicePixelRatio = window.devicePixelRatio || 1;

    canvas.width  = canvas.clientWidth  * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;

    context.configure({
        device: g_device,
        format: presentationFormat,
        alphaMode: 'opaque',
    });

    return [context, presentationFormat];
}



}