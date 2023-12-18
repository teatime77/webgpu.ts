var glMatrix: any;

namespace webgputs {

let validFrame : boolean = false;

class Run {
    context!: GPUCanvasContext;
    meshes: Mesh[] = [];
    depthTexture!: GPUTexture;

    async init(meshes: Mesh[]){
        this.meshes = meshes.splice(0);

        const is_instance = (document.getElementById("is-instance") as HTMLInputElement).checked;

        const canvas = document.getElementById('world') as HTMLCanvasElement;
    
        let vert_name : string;
        if(is_instance){
            vert_name = "instance-vert";
        }
        else{
            vert_name = "shape-vert";
        }
            
        const context = initContext(canvas, 'opaque');
        this.context = context;

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));

        for(let mesh of this.meshes){
            // create a render pipeline
            await mesh.makePipeline(vert_name, 'depth-frag', mesh.topology, is_instance);

            mesh.makeUniformBuffer();

            mesh.makeVertexBuffer(mesh.cube_vertex_count, mesh.cubeVertexArray);

            if(mesh.isInstance){
                mesh.makeInstanceBuffer();
            }
        }

        this.depthTexture = g_device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

    }

    frame(): void {
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

        // @uniform
        const [pvw, worldMatrix] = ui3D.getTransformationMatrix();

        let normalMatrix = glMatrix.mat3.create();
        glMatrix.mat3.normalFromMat4(normalMatrix, worldMatrix);
        console.assert(normalMatrix.byteLength == mat3x3_size);

        const ambientColor      = getColor("ambient");
        const directionalColor  = getColor("directional");
        const lightingDirection = glMatrix.vec3.create();
        glMatrix.vec3.normalize( lightingDirection, glMatrix.vec3.fromValues(0.25, 0.25, 1) );

        for(const mesh of this.meshes){
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

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        this.meshes.forEach(mesh => mesh.render(passEncoder));

        passEncoder.end();

        g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(this.frame.bind(this));
    }
}

export async function asyncBodyOnLoadIns(meshes: Mesh[]) {
    validFrame = false;
    const run = new Run();
    await run.init(meshes);
    validFrame = true;
    requestAnimationFrame(run.frame.bind(run));
}

export async function asyncBodyOnLoadMulti(){
    // asyncBodyOnLoadIns([new Mesh(... makeCone()) , new Mesh(... makeCube()), new Mesh(... makeGeodesicPolyhedron()), new Mesh(... makeTube())]);
}

export async function asyncBodyOnLoadCone(){
    // asyncBodyOnLoadIns([new Mesh(... makeCone())]);
}

export async function asyncBodyOnLoadCube(){
    // asyncBodyOnLoadIns([new Mesh(... makeCube())]);
}

export async function asyncBodyOnLoadGeodesic(){
    // asyncBodyOnLoadIns([new Mesh(... makeGeodesicPolyhedron())]);
}

export async function asyncBodyOnLoadTube(){
    asyncBodyOnLoadIns([new Tube()]);
}


}