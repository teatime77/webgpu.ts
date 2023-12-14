namespace webgputs {

export function range(n: number) : number[]{
    return [...Array(n).keys()];
}

export let g_device : GPUDevice;
export let g_presentationFormat : GPUTextureFormat;

export async function fetchText(fileURL: string) {
    const response = await fetch(fileURL);
    const text = await response!.text();

    return text;
}

export async function fetchModule(shader_name: string) : Promise<GPUShaderModule> {
    const text = await fetchText(`../wgsl/${shader_name}.wgsl`);

    return g_device.createShaderModule({ code: text });
}

export class Pipeline {
    async initPipeline(){
    }
}

function makeVertexBufferLayouts(is_instance : boolean) : GPUVertexBufferLayout[] {
    const cubeVertexSize = 4 * 8; // Byte size of one vertex.
    const cubePositionOffset = 4 * 0;
    const cubeColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.

    const vertex_buffer_layouts : GPUVertexBufferLayout[] = [
        {
            // 配列の要素間の距離をバイト単位で指定します。
            arrayStride: cubeVertexSize,

            // バッファを頂点ごとに参照することを意味します。
            stepMode: 'vertex',

            // 頂点バッファの属性を指定します。
            attributes: [
                {
                    // position
                    shaderLocation: 0, // @location(0) in vertex shader
                    offset: cubePositionOffset,
                    format: 'float32x4',
                },
                {
                    // color
                    shaderLocation: 1, // @location(1) in vertex shader
                    offset: cubeColorOffset,
                    format: 'float32x4',
                },
            ],
        }            
    ];

    if(is_instance){
        vertex_buffer_layouts.push({
            arrayStride: 4 * 2,

            // バッファをインスタンスごとに参照することを意味します。
            stepMode: 'instance',

            attributes: [
                {
                    shaderLocation: 2,
                    offset: 0,
                    format: 'float32x2'
                }
            ]
        });
    }

    return vertex_buffer_layouts;
}

export async function makePipeline(vert_name : string, frag_name : string, topology : GPUPrimitiveTopology, is_instance : boolean){
    const vert_module = await fetchModule(vert_name);
    const frag_module = await fetchModule(frag_name);

    const vertex_buffer_layouts = makeVertexBufferLayouts(is_instance);

    const pipeline_descriptor : GPURenderPipelineDescriptor = {
        layout: 'auto',
        vertex: {
            module: vert_module,
            entryPoint: 'main',
            buffers: vertex_buffer_layouts,
        },
        fragment: {
            module: frag_module,
            entryPoint: 'main',
            targets: [
                // 0
                { // @location(0) in fragment shader
                    format: g_presentationFormat,
                },
            ],
        },
        primitive: {
            topology: topology,
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
    };

    return g_device.createRenderPipeline(pipeline_descriptor);
}

export async function asyncBodyOnLoad(){
    console.log("body is loaded\n");
    
    if (!navigator.gpu) {
        throw Error('WebGPU not supported.');
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        throw Error('Couldn\'t request WebGPU adapter.');
    }

    g_device = await adapter!.requestDevice();
    console.log("device is ready\n");

    g_presentationFormat = navigator.gpu.getPreferredCanvasFormat();
}

let g_context : GPUCanvasContext | null = null;

export function initContext(canvas: HTMLCanvasElement, alpha_mode : GPUCanvasAlphaMode) : GPUCanvasContext {
    if(g_context != null){
        const tex = g_context.getCurrentTexture();
        tex.destroy();
        g_context.unconfigure();
    }

    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    g_context = context;

    const devicePixelRatio = window.devicePixelRatio || 1;

    canvas.width  = canvas.clientWidth  * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;

    context.configure({
        device: g_device,
        format: g_presentationFormat,
        alphaMode: 'opaque',
    });

    return context;
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