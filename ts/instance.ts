import { msg, fetchText, sleep, assert, MyError, zip } from "@i18n";
import { asyncBodyOnLoadCom, ComputePipeline } from "./compute.js";
import { editor } from "./editor.js";
import { Package, makeMesh } from "./package.js";
import { BufferUsage, ParamsParser, parseParams, ShaderType, Struct } from "./parser.js";
import { CalcRenderPipeline, RenderPipeline } from "./primitive.js";
import { initUI3D, ui3D } from "./ui.js";
import { initContext, g_device, number123 } from "./util.js";
import { asyncBodyOnLoadDemo } from "./demo.js";
import { asyncBodyOnLoadTex } from "./texture.js";

declare var glMatrix: any;

export let requestId : number = 0;

export function setRequestId(request_Id : number){
    requestId = request_Id;
}

let validFrame : boolean = false;

class Run {
    context!: GPUCanvasContext;
    meshes: RenderPipeline[] = [];
    depthTexture!: GPUTexture;
    comps : ComputePipeline[] = [];
    tick : number = 0;

    async init(inst : ComputePipeline[], meshes: RenderPipeline[]){
        this.meshes = meshes.splice(0);
        this.comps = inst.splice(0);

        const canvas = document.getElementById('world') as HTMLCanvasElement;
            
        this.context = initContext(canvas, 'opaque');

        initUI3D(canvas);
        editor.camera.camDistance = -12
    
        for(const comp of this.comps){

            await comp.initCompute();
        }

        for(let mesh of this.meshes){
            // create a render pipeline
            await mesh.makeRenderPipeline();

            if(!(mesh instanceof CalcRenderPipeline)){
                mesh.makeVertexBuffer();
            }

            mesh.makeUniformBuffer();
            mesh.makeBindGroup();
        }

        this.depthTexture = g_device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    async frame() {
        if(!validFrame){
            return;
        }

        const commandEncoder = g_device.createCommandEncoder();

        const textureView = this.context.getCurrentTexture().createView();

        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };

        ui3D.setEnv();

        const bindGroupIdx = this.tick % 2;
        for(const comp of this.comps){
            comp.writeUniformBuffer(ui3D.env, 0);

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(comp.pipeline);
            passEncoder.setBindGroup(0, comp.bindGroups[bindGroupIdx]);
            if(comp.workgroupCounts != null){

                const v = comp.workgroupCounts;
                switch(v.length){
                case 1: passEncoder.dispatchWorkgroups(v[0]); break;
                case 2: passEncoder.dispatchWorkgroups(v[0], v[1]); break;
                case 3: passEncoder.dispatchWorkgroups(v[0], v[1], v[2]); break;
                }
                
            }
            else{

                passEncoder.dispatchWorkgroups(comp.instanceCount);
            }
            passEncoder.end();
        }
        {
            ui3D.setTransformationMatrixAndLighting();

            this.meshes.forEach(mesh => mesh.writeUniform());

            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

            this.meshes.forEach(mesh => mesh.render(this.tick, bindGroupIdx, passEncoder));

            passEncoder.end();
        }

        g_device.queue.submit([commandEncoder.finish()]);
        ++this.tick;

        requestId = requestAnimationFrame(this.frame.bind(this));
    }
}


export function stopAnimation(){
    // validFrame = false;
    if(requestId != 0){
        cancelAnimationFrame(requestId);
        requestId = 0;
    }
}

async function startAnimation(comps : ComputePipeline[], meshes: RenderPipeline[]){
    stopAnimation();
    validFrame = false;

    const run = new Run();
    await run.init(comps, meshes);
    validFrame = true;
    requestId = requestAnimationFrame(run.frame.bind(run));
}

export async function asyncBodyOnLoadPackage(package_name : string){
    msg(`Load Package:${package_name}`);
    const test_text = await fetchText(`./package/${package_name}.json`);
    const packages = JSON.parse(test_text) as Package[];

    for(const pkg of packages){
        const comps : ComputePipeline[] = [];
        let  meshes : RenderPipeline[] = [];

        if(pkg.computes != undefined){
            for(const info of pkg.computes){
                let parser : ParamsParser | undefined;

                if(info.params != undefined){
                    parser = parseParams(info.params);
                }

                const comp = new ComputePipeline(info.compName);
                await comp.makeComputePipeline();
                if(comp.compModule == undefined || comp.compModule.vars == undefined){
                    throw new MyError();
                }

                const storages = comp.compModule.vars.filter(x => x.mod.usage == BufferUsage.storage_read || x.mod.usage == BufferUsage.storage_read_write);
                assert(storages.length != 0);
                assert(storages.every(x => x.type instanceof ShaderType && x.type.elementType instanceof Struct));
                const elementTypeSizes = storages.map(x => (x.type as ShaderType).elementType.size());
                elementTypeSizes.every(x => x == elementTypeSizes[0]);
                const instance_size  = elementTypeSizes[0] / 4;

                let instanceCount : number;
                if(info.globalGrid != undefined){
                    if(typeof info.globalGrid == "number"){
                        info.globalGrid = [info.globalGrid];
                        instanceCount = info.globalGrid[0];
                    }
                    else if(info.globalGrid.length == 2){
                        throw new MyError();
                    }
                    else if(info.globalGrid.length == 3){
                        instanceCount = info.globalGrid[0] * info.globalGrid[1] * info.globalGrid[2];
                    }
                    else{
                        throw new MyError();
                    }
                }
                else{
                    if(parser == undefined){
                        throw new MyError();
                    }
                    instanceCount = parser.get("@instance_count");
                }
                const array_length = instanceCount * instance_size;
                comp.makeInstanceArray(array_length)

                comps.push(comp);

                const workgroupSize = comp.compModule.fns[0].mod.workgroup_size;
                console.log("workgroup-size", workgroupSize);

                if(workgroupSize != undefined && info.globalGrid != undefined && Array.isArray(info.globalGrid)){
                    assert(workgroupSize.length == info.globalGrid!.length);
                    const workgroupCounts = Array.from( zip(info.globalGrid, workgroupSize).map(([i,a,b])=> a / b) );
                    assert(zip(workgroupCounts, workgroupSize).every(([i,a,b]) => a * b == (info.globalGrid as number[])[i]));
                    comp.workgroupCounts = workgroupCounts as number123;
                    msg(`workgroup-Counts ${comp.workgroupCounts}`);
                }
                else if(parser != undefined && parser.vars.has("@workgroup_counts")){

                    comp.workgroupCounts = parser.vars.get("@workgroup_counts") as [number,number,number];
                }
            
                const comp_meshes = info.shapes.map(shape => makeMesh(shape)).flat();
                comp_meshes.forEach(x => x.compute = comp);

                meshes = meshes.concat(comp_meshes);
            }
        }

        if(pkg.shapes != undefined){
            const pkg_meshes = pkg.shapes.map(shape => makeMesh(shape)).flat();
            meshes = meshes.concat(pkg_meshes);
        }
        
        await startAnimation(comps, meshes);

        await sleep(3000);
    }
}

export async function asyncBodyOnLoadTestAll(){
    await asyncBodyOnLoadDemo();
    await asyncBodyOnLoadCom();
    await asyncBodyOnLoadTex();
    await asyncBodyOnLoadPackage("test");
}
