var glMatrix: any;

namespace webgputs {

export let requestId : number = 0;

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

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));
    
        for(const comp of this.comps){

            await comp.initCompute();
        }

        for(let mesh of this.meshes){
            // create a render pipeline
            await mesh.makeRenderPipeline();

            mesh.makeUniformBufferAndBindGroup();

            mesh.makeVertexBuffer();
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

        for(const comp of this.comps){
            comp.writeUniformBuffer(ui3D.env, 0);

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(comp.pipeline);
            passEncoder.setBindGroup(0, comp.bindGroups[this.tick % 2]);
            if(comp.workgroupCounts != null){

                passEncoder.dispatchWorkgroups(... comp.workgroupCounts);
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

            this.meshes.forEach(mesh => mesh.render(this.tick, passEncoder));

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
    const test_text = await fetchText(`../package/${package_name}.json`);
    const test = JSON.parse(test_text) as Package[];

    for(const pkg of test){
        const comps : ComputePipeline[] = [];
        let  meshes : RenderPipeline[] = [];

        for(const info of pkg.computes!){
            const parser = parseParams(info.params);
            const array_length = parser.get("@instance_count") * parser.get("@instance_size");

            const comp = new ComputePipeline(info.compName, info.varNames, array_length);
            comps.push(comp);

            if(parser.vars.has("@workgroup_counts")){

                comp.workgroupCounts = parser.vars.get("@workgroup_counts") as [number,number,number];
            }
        
            const comp_meshes = info.shapes.map(x => makeMesh(x)).flat();
            comp_meshes.forEach(x => x.compute = comp);

            meshes = meshes.concat(comp_meshes);
        }

        if(pkg.shapes != undefined){
            const pkg_meshes = pkg.shapes.map(x => makeMesh(x)).flat();
            meshes = meshes.concat(pkg_meshes);
        }
        
        await startAnimation(comps, meshes);

        await wait(3000);
    }
}

export async function asyncBodyOnLoadTestAll(){
    await asyncBodyOnLoadPackage("test");
}

export async function asyncBodyOnLoadArrow(){
    await asyncBodyOnLoadPackage("arrow");
}

export async function asyncBodyOnLoadMaxwell_1D(){
    await asyncBodyOnLoadPackage("maxwell");
}


}