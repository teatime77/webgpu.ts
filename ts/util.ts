namespace webgpu_ts {

export function range(n: number) : number[]{
    return [...Array(n).keys()];
}

export let g_device : GPUDevice;
export let g_presentationFormat : GPUTextureFormat;

export class MyError extends Error {
    constructor(text : string = ""){
        super();
    }
}

export function assert(b : boolean, msg : string = ""){
    if(!b){
        throw new MyError(msg);
    }
}    

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

export async function wait(millisec : number) : Promise<null> {
    return new Promise((resolve) => {
        setTimeout(()=>{
            resolve(null);
        }, millisec);
    });
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
    bitonic_sort_test();

    editor = new Editor();
    
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

export const particleDim = 8;

function addVertex(v : number[], x : number, y : number , z: number){
    let c = 4.0;

    v.push(c * x);
    v.push(c * y);
    v.push(c * z);
    v.push(0);

    c = 1 / Math.sqrt(x * x + y * y + z * z);
    v.push(c * x);
    v.push(c * y);
    v.push(c * z);
    v.push(0);

}
export function makeInitialInstanceArrayNEW() : Float32Array {
    const v : number[] = [];

    addVertex(v,  1,0,0);
    addVertex(v, -1,0,0);

    addVertex(v, 0, 1,0);
    addVertex(v, 0,-1,0);

    addVertex(v, 0,0, 1);
    addVertex(v, 0,0,-1);

    const th = Math.PI / 3.0;
    const s = Math.sin(th);
    const c = Math.cos(th);

    addVertex(v,-c,-s,0);
    addVertex(v, c, s,0);

    addVertex(v, 0,-c,-s);
    addVertex(v, 0, c, s);

    addVertex(v,-s,0,-c);
    addVertex(v, s,0, c);

    
    return new Float32Array(v);
}

export function makeShaderModule( text : string) : GPUShaderModule {
    g_device.pushErrorScope("validation");

    const module = g_device.createShaderModule({ code: text });

    g_device.popErrorScope().then((error) => {
        if (error) {
          console.error(`shader error:${error.message}\n${text}`);
          throw new Error(`shader error:${error.message}`);
        }
    });

    return module;
}

}