var glMatrix: any;

namespace webgputs {

export let requestId : number = 0;

let validFrame : boolean = false;

function mat4fromMat3(m3 : Float32Array){
    const m4 = glMatrix.mat4.create();

    m4[ 0] = m3[0];
    m4[ 1] = m3[1];
    m4[ 2] = m3[2];
    m4[ 3] = 0;
    
    m4[ 4] = m3[3];
    m4[ 5] = m3[4];
    m4[ 6] = m3[5];
    m4[ 7] = 0;
    
    m4[ 8] = m3[6];
    m4[ 9] = m3[7];
    m4[10] = m3[8];
    m4[11] = 0;
    
    // Set the last row and column to identity values
    m4[15] = 1;

    return m4;
}

export function updateVertexUniformBuffer(meshes : RenderPipeline[]){
    // @uniform
    const [pvw, worldMatrix] = ui3D.getTransformationMatrix();

    let normalMatrix = glMatrix.mat3.create();
    glMatrix.mat3.normalFromMat4(normalMatrix, worldMatrix);
    normalMatrix = mat4fromMat3(normalMatrix);

    const ambientColor      = getColor("ambient");
    const directionalColor  = getColor("directional");
    const lightingDirection = glMatrix.vec3.create();
    glMatrix.vec3.normalize( lightingDirection, glMatrix.vec3.fromValues(0.25, 0.25, 1) );

    for(const mesh of meshes){
        let offset = 0;

        offset = mesh.writeUniformBuffer(pvw, offset);
        offset = mesh.writeUniformBuffer(normalMatrix, offset);

        // vec4 align is 16
        // https://www.w3.org/TR/WGSL/#alignment-and-size
        // offset += 12;

        offset = mesh.writeUniformBuffer(mesh.materialColor, offset);
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

    async init(inst : Instance | null, info : ComputeInfo, meshes: RenderPipeline[]){
        this.meshes = meshes.splice(0);

        const is_instance = this.meshes.some(x => x.isInstance);

        const canvas = document.getElementById('world') as HTMLCanvasElement;
    
        if(is_instance){

            this.useCompute = true;

            this.comp = new ComputePipeline();
            await this.comp.initCompute(inst!, info);
        
            this.meshes.forEach(x => x.compute = this.comp);
        }
            
        const context = initContext(canvas, 'opaque');
        this.context = context;

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));

        for(let mesh of this.meshes){
            // create a render pipeline
            await mesh.makePipeline(info.vertName, info.fragName, mesh.topology);

            mesh.makeUniformBufferAndBindGroup();

            mesh.makeVertexBuffer();

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

        updateVertexUniformBuffer(this.meshes);

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

        requestId = requestAnimationFrame(this.frame.bind(this));
    }
}

export class ComputeInfo {
    compName : string;
    vertName : string;
    fragName : string
    uniformArray : Float32Array;

    constructor(comp_name : string, vert_name : string, frag_name : string, uniform_array : Float32Array){
        this.compName = comp_name;
        this.vertName   = vert_name;
        this.fragName   = frag_name;
        this.uniformArray = uniform_array;
    }
}

export function stopAnimation(){
    // validFrame = false;
    if(requestId != 0){
        cancelAnimationFrame(requestId);
        requestId = 0;
    }
}

async function startAnimation(inst : Instance | null, info : ComputeInfo, meshes: RenderPipeline[]){
    stopAnimation();
    validFrame = false;

    meshes.forEach(x => x.instance = inst);
    const run = new Run();
    await run.init(inst, info, meshes);
    validFrame = true;
    requestId = requestAnimationFrame(run.frame.bind(run));
}

export async function asyncBodyOnLoadIns(meshes: RenderPipeline[]) {
    const inst = makeInstance([ "a_particlePos", "a_particleVel" ], makeInitialInstanceArray());

    const vert_name = (inst == null ? "shape-vert": "instance-vert");
    const info = new ComputeInfo("updateSprites", vert_name, "depth-frag", makeComputeUniformArray());

    startAnimation(inst, info, meshes);
}

export async function asyncBodyOnLoadBoi() {
    const inst = makeInstance([ "a_particlePos", "a_particleVel" ], makeInitialInstanceArray())!;

    const vert_name = (inst == null ? "shape-vert": "sprite");
    const info = new ComputeInfo("updateSprites", vert_name, "depth-frag", makeComputeUniformArray());

    const mesh = new RenderPipeline();
    mesh.vertexArray = makeConeSub3(true);
    mesh.vertexCount = mesh.vertexArray.length / 6;

    const meshes: RenderPipeline[] = [ mesh ];

    startAnimation(inst, info, meshes);
}

export function makeInstance(var_names : string[], instance_array : Float32Array) : Instance | null {
    const is_instance = (document.getElementById("is-instance") as HTMLInputElement).checked;

    if(is_instance){
        return new Instance(var_names, instance_array);
    }
    else{
        return null;
    }
}

export async function asyncBodyOnLoadMulti(){
    asyncBodyOnLoadIns([ (new Cone()).red(), (new Cube()).green(), (new Tube()).blue(), new GeodesicPolyhedron() ]);
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