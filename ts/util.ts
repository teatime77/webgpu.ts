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

export function getColor(name : string){
    const rgb : number[] = [];

    for(const c of ["r", "g", "b"]){
        const input = document.getElementById(`${name}-${c}`) as HTMLInputElement;
        rgb.push( parseInt(input.value) / 100.0 );
    }

    return glMatrix.vec4.fromValues(rgb[0], rgb[1], rgb[2], 1.0);
}

export async function fetchModule(shader_name: string) : Promise<GPUShaderModule> {
    const text = await fetchText(`../wgsl/${shader_name}.wgsl`);

    return g_device.createShaderModule({ code: text });
}

export function makeVertexBufferLayouts(is_instance : boolean) : GPUVertexBufferLayout[] {
    const cubeVertexSize = 2 * vec3_size; // Byte size of one vertex.
    const cubePositionOffset = 0;
    const cubeNormOffset = vec3_size; // Byte offset of cube vertex norm attribute.

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
                    format: 'float32x3',
                },
                {
                    // norm
                    shaderLocation: 1, // @location(1) in vertex shader
                    offset: cubeNormOffset,
                    format: 'float32x3',
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

}