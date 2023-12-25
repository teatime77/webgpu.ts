namespace webgputs {

export function range(n: number) : number[]{
    return [...Array(n).keys()];
}

export let g_device : GPUDevice;
export let g_presentationFormat : GPUTextureFormat;

export function error(msg : string){
    console.log(`error [${msg}]`)
    throw new Error(msg);
}

export function msg(s : string){
    console.log(s);
}

export function sum(v : number[]) : number {
    return v.reduce((acc, val) => acc + val, 0);
}

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

export async function fetchModule(shader_name: string) : Promise<Module> {
    const text = await fetchText(`../wgsl/${shader_name}.wgsl`);
    const module = new Module(text);

    return module;
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

    await parseAll();
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