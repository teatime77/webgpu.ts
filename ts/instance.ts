var glMatrix: any;

namespace webgputs {

let validFrame : boolean = false;

export async function updateVertexUniformBuffer(meshes : RenderPipeline[]){
    // @uniform
    const [pvw, worldMatrix] = ui3D.getTransformationMatrix();

    let normalMatrix = glMatrix.mat3.create();
    glMatrix.mat3.normalFromMat4(normalMatrix, worldMatrix);

    const ambientColor      = getColor("ambient");
    const directionalColor  = getColor("directional");
    const lightingDirection = glMatrix.vec3.create();
    glMatrix.vec3.normalize( lightingDirection, glMatrix.vec3.fromValues(0.25, 0.25, 1) );

    for(const mesh of meshes){
        if(mesh.isInstance){
            await mesh.instance!.update();
        }

        let offset = 0;

        offset = mesh.writeUniformBuffer(pvw, offset);
        offset = mesh.writeUniformBuffer(normalMatrix, offset);

        // vec4 align is 16
        // https://www.w3.org/TR/WGSL/#alignment-and-size
        offset += 12;

        offset = mesh.writeUniformBuffer(ambientColor     , offset);
        offset = mesh.writeUniformBuffer(directionalColor , offset);
        offset = mesh.writeUniformBuffer(lightingDirection, offset);
    }
}

class Run {
    context!: GPUCanvasContext;
    meshes: RenderPipeline[] = [];
    depthTexture!: GPUTexture;
    useCompute : boolean = false;
    comp : ComputePipeline | undefined;
    tick : number = 0;

    async init(meshes: RenderPipeline[]){
        this.meshes = meshes.splice(0);

        const is_instance = this.meshes.some(x => x.isInstance);

        const canvas = document.getElementById('world') as HTMLCanvasElement;
    
        let vert_name : string;
        if(is_instance){
            vert_name = "instance-vert";

            this.useCompute = true;

            this.comp = new ComputePipeline();
            await this.comp.makePipeline("updateSprites");
            const compute_uniform_array = makeComputeUniformArray();
            this.comp.makeUniformBuffer(compute_uniform_array);
        
            const initial_instance_array = makeInitialInstanceArray();
            this.comp.instanceCount = initial_instance_array.length / particleDim;
            this.comp.makeUpdateBuffers(initial_instance_array);
        
            this.meshes.forEach(x => x.compute = this.comp);
        }
        else{
            vert_name = "shape-vert";
        }
            
        const context = initContext(canvas, 'opaque');
        this.context = context;

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));

        for(let mesh of this.meshes){
            // create a render pipeline
            await mesh.makePipeline(vert_name, 'depth-frag', mesh.topology);

            mesh.makeUniformBufferAndBindGroup();

            mesh.makeVertexBuffer(mesh.cube_vertex_count, mesh.cubeVertexArray);

            if(mesh.isInstance){
                mesh.instance!.makeInstanceBuffer();
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

        await updateVertexUniformBuffer(this.meshes);

        if(this.useCompute){

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.comp!.pipeline);
            passEncoder.setBindGroup(0, this.comp!.bindGroups[this.tick % 2]);
            passEncoder.dispatchWorkgroups(this.comp!.instanceCount!);
            passEncoder.end();
        }
        {
            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

            this.meshes.forEach(mesh => mesh.render(this.tick, passEncoder));

            passEncoder.end();
        }

        g_device.queue.submit([commandEncoder.finish()]);
        ++this.tick;

        requestAnimationFrame(this.frame.bind(this));
    }
}

export async function asyncBodyOnLoadIns(meshes: RenderPipeline[]) {
    validFrame = false;
    const run = new Run();
    await run.init(meshes);
    validFrame = true;
    requestAnimationFrame(run.frame.bind(run));
}

function inst() : Instance | null {
    const is_instance = (document.getElementById("is-instance") as HTMLInputElement).checked;

    if(is_instance){
        return new Instance();
    }
    else{
        return null;
    }
}

export async function asyncBodyOnLoadMulti(){
    asyncBodyOnLoadIns([ new Cone(inst()), new Cube(inst()), new Tube(inst()), new GeodesicPolyhedron(inst()) ]);
}

export async function asyncBodyOnLoadCone(){
    asyncBodyOnLoadIns([new Cone(inst())]);
}

export async function asyncBodyOnLoadCube(){
    asyncBodyOnLoadIns([new Cube(inst())]);
}

export async function asyncBodyOnLoadGeodesic(){
    asyncBodyOnLoadIns([new GeodesicPolyhedron(inst())]);
}

export async function asyncBodyOnLoadTube(){
    asyncBodyOnLoadIns([new Tube(inst())]);
}


}