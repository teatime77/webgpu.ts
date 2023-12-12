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

export function makeUniformBufferAndBindGroup(g_device : GPUDevice, pipeline : GPURenderPipeline, uniformBufferSize : number) : [GPUBuffer, GPUBindGroup] {

    const uniformBuffer = g_device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformBindGroup = g_device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                },
            },
        ],
    });

    return [uniformBuffer, uniformBindGroup];
}

}