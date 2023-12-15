var glMatrix: any;

namespace webgputs {

let validFrame : boolean = false;

class Run {
    context!: GPUCanvasContext;
    pipelines: Pipeline[] = [];
    depthTexture!: GPUTexture;

    
    async init(polygons: Polygon[]){
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

        for(let polygon of polygons){
            // create a render pipeline
            const pipeline = await makePipeline(vert_name, 'depth-frag', polygon.topology, is_instance);

            pipeline.makeBuffer(polygon.cube_vertex_count, polygon.cubeVertexArray);

            this.pipelines.push(pipeline);
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

        const pvw = ui3D.getTransformationMatrix();

        for(let pipeline of this.pipelines){
            g_device.queue.writeBuffer(
                pipeline.uniformBuffer, 4 * 16 * 2, 
                pvw.buffer, pvw.byteOffset, pvw.byteLength
            );
        }

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        for(let pipeline of this.pipelines){

            pipeline.render(passEncoder);
        }

        passEncoder.end();

        g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(this.frame.bind(this));
    }
}

export async function asyncBodyOnLoadIns(polygons: Polygon[]) {
    validFrame = false;
    const run = new Run();
    await run.init(polygons);
    validFrame = true;
    requestAnimationFrame(run.frame.bind(run));
}

export async function asyncBodyOnLoadCone(){
    asyncBodyOnLoadIns([new Polygon(... makeCone()) , new Polygon(... makeCube()), new Polygon(... makeGeodesicPolyhedron()), new Polygon(... makeTube())]);
}

export async function asyncBodyOnLoadSphere(){
    asyncBodyOnLoadIns([new Polygon(... makeSphere())]);
}

export async function asyncBodyOnLoadCube(){
    asyncBodyOnLoadIns([new Polygon(... makeCube())]);
}

export async function asyncBodyOnLoadGeodesic(){
    asyncBodyOnLoadIns([new Polygon(... makeGeodesicPolyhedron())]);
}

export async function asyncBodyOnLoadTube(){
    asyncBodyOnLoadIns([new Polygon(... makeTube())]);
}


}