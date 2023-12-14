var glMatrix: any;

namespace webgputs {

let validFrame : boolean = false;

class Run {
    cubeVertexCount! : number;
    context!: GPUCanvasContext;
    pipeline!: GPURenderPipeline;
    verticesBuffer!: GPUBuffer;
    uniformBindGroup!: GPUBindGroup;
    uniformBuffer!: GPUBuffer;
    depthTexture!: GPUTexture;
    instancePositions : Float32Array | undefined;
    instancesBuffer: GPUBuffer | undefined;
    isInstance! : boolean;

    
    async init(cube_vertex_count : number , cubeVertexArray : Float32Array, topology : GPUPrimitiveTopology){
        const is_instance = document.getElementById("is-instance") as HTMLInputElement;
        this.isInstance = is_instance.checked;

        const canvas = document.getElementById('world') as HTMLCanvasElement;
        const cubeVertexSize = 4 * 8; // Byte size of one vertex.
        const cubePositionOffset = 4 * 0;
        const cubeColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
    
        this.cubeVertexCount = cube_vertex_count;

        let vert_shader : string;
        if(this.isInstance){
            vert_shader = "instance-vert";
        
            this.instancePositions = new Float32Array([
                // x, y
                -5, -5,
                -5, 0,
                -5, 5,
                0, -5,
                0, 0,
                0, 5,
                5, -5,
                5, 0,
                5, 5
            ]);
        }
        else{

            vert_shader = "shape-vert";
        }
    
        const vertWGSL = await fetchText(`../wgsl/${vert_shader}.wgsl`);
    
        const fragWGSL = await fetchText('../wgsl/depth-frag.wgsl');
        
        const [context, presentationFormat] = initContext(canvas, 'opaque');
        this.context = context;

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));

        const vertex_buffer_layouts : GPUVertexBufferLayout[] = [
            {
                // 配列の要素間の距離をバイト単位で指定します。
                arrayStride: cubeVertexSize,

                // バッファを頂点ごとに参照することを意味します。
                stepMode: 'vertex',

                // 頂点バッファの属性を指定します。
                attributes: [
                    {
                        // position
                        shaderLocation: 0, // @location(0) in vertex shader
                        offset: cubePositionOffset,
                        format: 'float32x4',
                    },
                    {
                        // color
                        shaderLocation: 1, // @location(1) in vertex shader
                        offset: cubeColorOffset,
                        format: 'float32x4',
                    },
                ],
            }            
        ];

        if(this.isInstance){
            vertex_buffer_layouts.push({
                arrayStride: 4 * 2,

                // バッファをインスタンスごとに参照することを意味します。
                stepMode: 'instance',

                attributes: [
                    {
                        shaderLocation: 2,
                        offset: 0,
                        format: 'float32x2'
                    }
                ]
            });
        }

        const pipeline_descriptor : GPURenderPipelineDescriptor = {
            layout: 'auto',
            vertex: {
                module: g_device.createShaderModule({
                    code: vertWGSL,
                }),
                entryPoint: 'main',
                buffers: vertex_buffer_layouts,
            },
            fragment: {
                module: g_device.createShaderModule({
                    code: fragWGSL,
                }),
                entryPoint: 'main',
                targets: [
                    // 0
                    { // @location(0) in fragment shader
                        format: presentationFormat,
                    },
                ],
            },
            primitive: {
                topology: topology,
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        };

        // create a render pipeline
        this.pipeline = g_device.createRenderPipeline(pipeline_descriptor);

        const uniformBufferSize = 4 * 16 * 3; // 4x4 matrix * 3

        const [uniformBuffer, uniformBindGroup] = makeUniformBufferAndBindGroup(g_device, this.pipeline, uniformBufferSize);
        this.uniformBuffer    = uniformBuffer;
        this.uniformBindGroup = uniformBindGroup;

        // Create a vertex buffer from the quad data.
        this.verticesBuffer = g_device.createBuffer({
            size: cubeVertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.verticesBuffer.getMappedRange()).set(cubeVertexArray);
        this.verticesBuffer.unmap();

        if(this.isInstance){

            // Create a instances buffer
            this.instancesBuffer = g_device.createBuffer({
                size: this.instancePositions!.byteLength,
                usage: GPUBufferUsage.VERTEX,
                mappedAtCreation: true,
            });
            new Float32Array(this.instancesBuffer.getMappedRange()).set(this.instancePositions!);
            this.instancesBuffer.unmap();
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

        g_device.queue.writeBuffer(
            this.uniformBuffer, 4 * 16 * 2, 
            pvw.buffer, pvw.byteOffset, pvw.byteLength
        );

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        if(this.isInstance){

            passEncoder.setVertexBuffer(1, this.instancesBuffer!);
            passEncoder.draw(this.cubeVertexCount, Math.floor(this.instancePositions!.length / 2));
        }
        else{

            passEncoder.draw(this.cubeVertexCount);
        }
        passEncoder.end();

        g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(this.frame.bind(this));
    }
}

export async function asyncBodyOnLoadIns(cube_vertex_count : number , cubeVertexArray : Float32Array, topology : GPUPrimitiveTopology) {
    validFrame = false;
    const run = new Run();
    await run.init(cube_vertex_count , cubeVertexArray, topology);
    validFrame = true;
    requestAnimationFrame(run.frame.bind(run));
}

export async function asyncBodyOnLoadCone(){
    asyncBodyOnLoadIns(... makeCone());
}

export async function asyncBodyOnLoadSphere(){
    asyncBodyOnLoadIns(... makeSphere());
}

export async function asyncBodyOnLoadCube(){
    asyncBodyOnLoadIns(... makeCube());
}

export async function asyncBodyOnLoadGeodesic(){
    asyncBodyOnLoadIns(... makeGeodesicPolyhedron());
}

export async function asyncBodyOnLoadTube(){
    asyncBodyOnLoadIns(... makeTube());
}


}