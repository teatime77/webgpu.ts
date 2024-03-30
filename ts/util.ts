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

export const particleDim = 8;

export function makeInitialInstanceArray() : Float32Array {
    const theta_cnt = 10;
    const phi_cnt   = 32;
    const numParticles = theta_cnt * phi_cnt;
    const initial_instance_array = new Float32Array(numParticles * particleDim);

    const c = 3.0;
    let base = 0;
    for(let theta_i = 0; theta_i < theta_cnt; theta_i++){
        const theta = Math.PI * theta_i / theta_cnt;
        const z = c * Math.cos(theta);
        const r = c * Math.sin(theta);

        for(let phi_i = 0; phi_i < phi_cnt; phi_i++){
            const phi = 2 * Math.PI * phi_i / phi_cnt;

            const x = r * Math.cos(phi);
            const y = r * Math.sin(phi);

            initial_instance_array[base + 0] = x; // 0.0;// 2 * (Math.random() - 0.5);
            initial_instance_array[base + 1] = y; // 0.0;// 2 * (Math.random() - 0.5);
            initial_instance_array[base + 2] = z; // 3.0;// 2 * (Math.random() - 0.5);
            initial_instance_array[base + 3] = 0.0;

            const speed = 0.0;// 5.0;
            initial_instance_array[base + 4] = speed * (Math.random() - 0.5);
            initial_instance_array[base + 5] = speed * (Math.random() - 0.5);
            initial_instance_array[base + 6] = speed * (Math.random() - 0.5);
            initial_instance_array[base + 7] = 0.0;

            base += particleDim;
        }
    }
    console.assert(base == initial_instance_array.length);

    return initial_instance_array;
}


}