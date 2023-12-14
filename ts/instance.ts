var glMatrix: any;

namespace webgputs {

class Run {
    cubeVertexCount! : number;
    g_device!: GPUDevice;
    context!: GPUCanvasContext;
    pipeline!: GPURenderPipeline;
    verticesBuffer!: GPUBuffer;
    uniformBindGroup!: GPUBindGroup;
    uniformBuffer!: GPUBuffer;
    depthTexture!: GPUTexture;
    instancePositions! : Float32Array;
    instancesBuffer!: GPUBuffer;
    
    async init(canvas: HTMLCanvasElement){

        const cubeVertexSize = 4 * 8; // Byte size of one vertex.
        const cubePositionOffset = 4 * 0;
        const cubeColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
    
        const [cube_vertex_count , cubeVertexArray, topology ] = makeCube();
        this.cubeVertexCount = cube_vertex_count;
    
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
    
        const vertWGSL = await fetchText('../wgsl/instance-vert.wgsl');
    
        const fragWGSL = await fetchText('../wgsl/depth-frag.wgsl');
    
        const g_adapter = await navigator.gpu.requestAdapter();
        this.g_device = await g_adapter!.requestDevice();
    
        const [context, presentationFormat] = initContext(this.g_device, canvas, 'opaque');
        this.context = context;

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -12));

        // create a render pipeline
        this.pipeline = this.g_device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.g_device.createShaderModule({
                    code: vertWGSL,
                }),
                entryPoint: 'main',
                buffers: [
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
                    },
                    {
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
                    }
                ],
            },
            fragment: {
                module: this.g_device.createShaderModule({
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
        });

        const uniformBufferSize = 4 * 16 * 3; // 4x4 matrix * 3

        const [uniformBuffer, uniformBindGroup] = makeUniformBufferAndBindGroup(this.g_device, this.pipeline, uniformBufferSize);
        this.uniformBuffer    = uniformBuffer;
        this.uniformBindGroup = uniformBindGroup;

        // Create a vertex buffer from the quad data.
        this.verticesBuffer = this.g_device.createBuffer({
            size: cubeVertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.verticesBuffer.getMappedRange()).set(cubeVertexArray);
        this.verticesBuffer.unmap();

        // Create a instances buffer
        this.instancesBuffer = this.g_device.createBuffer({
            size: this.instancePositions.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.instancesBuffer.getMappedRange()).set(this.instancePositions);
        this.instancesBuffer.unmap();


        this.depthTexture = this.g_device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

    }

    frame(): void {
        const commandEncoder = this.g_device.createCommandEncoder();
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

        this.g_device.queue.writeBuffer(
            this.uniformBuffer, 4 * 16 * 2, 
            pvw.buffer, pvw.byteOffset, pvw.byteLength
        );

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setVertexBuffer(1, this.instancesBuffer);
        passEncoder.draw(this.cubeVertexCount, Math.floor(this.instancePositions.length / 2));
        passEncoder.end();

        this.g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(this.frame.bind(this));
    }


}

export async function asyncBodyOnLoadIns() {
    const run = new Run();
    await run.init(document.getElementById('world') as HTMLCanvasElement);
    requestAnimationFrame(run.frame.bind(run));

}

}