var glMatrix: any;

namespace webgputs {

export let requestId : number = 0;

let validFrame : boolean = false;

class Run {
    context!: GPUCanvasContext;
    meshes: RenderPipeline[] = [];
    depthTexture!: GPUTexture;
    comp : ComputePipeline | null = null;
    tick : number = 0;

    async init(inst : Instance | null, meshes: RenderPipeline[]){
        this.meshes = meshes.splice(0);

        const canvas = document.getElementById('world') as HTMLCanvasElement;
            
        this.context = initContext(canvas, 'opaque');

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));
    
        if(inst != null){

            this.comp = new ComputePipeline("updateSprites");
            await this.comp.initCompute(inst!);
        
            this.meshes.filter(x => !(x instanceof Line)).forEach(x => x.compute = this.comp);
        }

        for(let mesh of this.meshes){
            // create a render pipeline
            await mesh.makeRenderPipeline();

            mesh.makeUniformBufferAndBindGroup();

            mesh.makeVertexBuffer();

            if(mesh.instance != null){
                mesh.instance.makeInstanceBuffer();
            }
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

        if(this.comp != null){
            this.comp.writeUniformBuffer(ui3D.env, 0);

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.comp!.pipeline);
            passEncoder.setBindGroup(0, this.comp!.bindGroups[this.tick % 2]);
            passEncoder.dispatchWorkgroups(this.comp!.instanceCount!);
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

async function startAnimation(inst : Instance | null, meshes: RenderPipeline[]){
    stopAnimation();
    validFrame = false;

    meshes.forEach(x => x.instance = inst);
    const run = new Run();
    await run.init(inst, meshes);
    validFrame = true;
    requestId = requestAnimationFrame(run.frame.bind(run));
}

export async function asyncBodyOnLoadIns(meshes: RenderPipeline[]) {
    const inst = makeInstance([ "meshPos", "meshVec" ], makeInitialInstanceArray());

    startAnimation(inst, meshes);
}

export async function asyncBodyOnLoadBoi() {
    const inst = makeInstance([ "meshPos", "meshVec" ], makeInitialInstanceArray())!;

    const mesh = new RenderPipeline();
    if(isInstance()){

        mesh.vertName    = "sprite";
    }
    else{

        mesh.vertName    = "shape-vert";
    }
    mesh.vertexArray = makeConeSub3(true);
    mesh.vertexCount = mesh.vertexArray.length / 6;

    const meshes: RenderPipeline[] = [ mesh ];

    startAnimation(inst, meshes);
}

export async function asyncBodyOnLoadArrow(){
    asyncBodyOnLoadIns(makeArrow());
}

export function makeInstance(var_names : string[], instance_array : Float32Array) : Instance | null {
    if(isInstance()){
        return new Instance(var_names, instance_array);
    }
    else{
        return null;
    }
}

export async function asyncBodyOnLoadMulti(){
    asyncBodyOnLoadIns([ (new Cone()).red(), (new Cube()).green(), (new Tube()).blue(), new GeodesicPolyhedron() ]);
}

export async function asyncBodyOnLoadDisc(){
    asyncBodyOnLoadIns([(new Disc()).red()]);
}

export async function asyncBodyOnLoadCone(){
    asyncBodyOnLoadIns([(new Cone()).red()]);
}

export async function asyncBodyOnLoadCube(){
    asyncBodyOnLoadIns([(new Cube()).green()]);
}

export async function asyncBodyOnLoadGeodesic(){
    asyncBodyOnLoadIns([(new GeodesicPolyhedron()).blue()]);
}

export async function asyncBodyOnLoadTube(){
    asyncBodyOnLoadIns([new Tube()]);
}


}