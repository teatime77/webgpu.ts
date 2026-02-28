declare var glMatrix: any;

import { $div, fetchText } from "@i18n";
import { testGalois } from "@algebra"
import { initEditor } from "./editor.js";
import { Module, parseAll } from "./parser.js";
import { asyncBodyOnLoadPackage, asyncBodyOnLoadTestAll, asyncBodyOnLoadMaxwell_1D, stopAnimation } from "./instance.js";
import { asyncBodyOnLoadDemo } from "./demo.js";
import { asyncBodyOnLoadCom } from "./compute.js";
import { asyncBodyOnLoadTex } from "./texture.js";
import { initFnc } from "./fnc.js";

export let g_device : GPUDevice;
export let g_presentationFormat : GPUTextureFormat;

export function error(msg : string){
    console.log(`error [${msg}]`)
    throw new Error(msg);
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
    const text = await fetchText(`./wgsl/${shader_name}.wgsl`);
    const module = new Module(text);

    return module;
}


export async function asyncBodyOnLoad(){
    console.log("body is loaded\n");
    // bitonic_sort_test();
    // testGalois();

    initEditor();
    
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

// app/src/main.ts

window.addEventListener('load', async() => {
    console.log('画像も含めてすべてのロードが完了しました');
    await asyncBodyOnLoad();
    // await initFnc();
    console.log('初期化完了');
});

let divButtons = $div("div-buttons");
function makeButton( text : string) : HTMLButtonElement {
    const button = document.createElement("button");
    button.innerText = text;
    divButtons.appendChild(button);

    return button;
}

makeButton("Function").addEventListener("click", async()=>{ await asyncBodyOnLoadPackage("function") });
makeButton("test all").addEventListener("click", async()=>{ await asyncBodyOnLoadTestAll() });
makeButton("Demo").addEventListener("click", async()=>{ await asyncBodyOnLoadDemo() });
makeButton("Compute").addEventListener("click", async()=>{ await asyncBodyOnLoadCom() });
makeButton("Arrow").addEventListener("click", async()=>{ await asyncBodyOnLoadPackage("arrow") });
makeButton("電場").addEventListener("click", async()=>{ await asyncBodyOnLoadPackage("electric-field") });
makeButton("Point").addEventListener("click", async()=>{ await asyncBodyOnLoadPackage("point") });
makeButton("Line").addEventListener("click", async()=>{ await asyncBodyOnLoadPackage("line") });
makeButton("Maxwell 1D").addEventListener("click", async()=>{ await asyncBodyOnLoadMaxwell_1D() });
makeButton("Texture").addEventListener("click", async()=>{ await asyncBodyOnLoadTex() });
makeButton("Stop").addEventListener("click", ()=>{ stopAnimation() });
