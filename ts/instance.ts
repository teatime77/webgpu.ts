var glMatrix: any;

namespace webgputs {

export async function asyncBodyOnLoadIns() {

    const cubeVertexSize = 4 * 8; // Byte size of one vertex.
    const cubePositionOffset = 4 * 0;
    const cubeColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
    const cubeVertexCount = 36;

    const cubeVertexArray = new Float32Array([
        // float4 position, float4 color
        1, -1, 1, 1, 1, 0, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,
        1, -1, -1, 1, 1, 0, 0, 1,
        1, -1, 1, 1, 1, 0, 1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,

        1, 1, 1, 1, 1, 1, 1, 1,
        1, -1, 1, 1, 1, 0, 1, 1,
        1, -1, -1, 1, 1, 0, 0, 1,
        1, 1, -1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, -1, -1, 1, 1, 0, 0, 1,

        -1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, -1, 1, 1, 1, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
        -1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, -1, 1, 1, 1, 0, 1,

        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, 1, 1, 1, 0, 1, 1, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,

        1, 1, 1, 1, 1, 1, 1, 1,
        -1, 1, 1, 1, 0, 1, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        1, -1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,

        1, -1, -1, 1, 1, 0, 0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
        1, 1, -1, 1, 1, 1, 0, 1,
        1, -1, -1, 1, 1, 0, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
    ]);

    const instancePositions = new Float32Array([
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


    const quadIndexArray = new Uint16Array([0, 1, 2, 0, 2, 3]);

    const vertWGSL = await fetchText('../wgsl/instance-vert.wgsl');

    const fragWGSL = await fetchText('../wgsl/depth-frag.wgsl');

    const g_adapter = await navigator.gpu.requestAdapter();
    const g_device = await g_adapter!.requestDevice();

    async function init(canvas: HTMLCanvasElement): Promise<{ context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture, instancesBuffer: GPUBuffer }> {
        const [context, presentationFormat] = initContext(g_device, canvas, 'opaque');

        // create a render pipeline
        const pipeline = g_device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: g_device.createShaderModule({
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
                topology: 'triangle-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });

        const uniformBufferSize = 4 * 16 * 3; // 4x4 matrix * 3
        const uniformBuffer = g_device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });


        const uniformBindGroup = g_device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuffer,
                    },
                },
            ],
        });

        // Create a vertex buffer from the quad data.
        const verticesBuffer = g_device.createBuffer({
            size: cubeVertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(verticesBuffer.getMappedRange()).set(cubeVertexArray);
        verticesBuffer.unmap();

        // Create a instances buffer
        const instancesBuffer = g_device.createBuffer({
            size: instancePositions.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(instancesBuffer.getMappedRange()).set(instancePositions);
        instancesBuffer.unmap();


        const depthTexture = g_device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        return { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, instancesBuffer };
    }

    function frame(
        { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, instancesBuffer }:
            { context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture, instancesBuffer: GPUBuffer }
    ): void {
        const commandEncoder = g_device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

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
                view: depthTexture.createView(),

                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };

        queueTransformationMatrix(g_device, uniformBuffer, glMatrix.vec3.fromValues(0, 0, -12));


        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, verticesBuffer);
        passEncoder.setVertexBuffer(1, instancesBuffer);
        passEncoder.draw(cubeVertexCount, Math.floor(instancePositions.length / 2));
        passEncoder.end();

        g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, instancesBuffer }));
    }

    const { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, instancesBuffer } = await init(document.getElementById('world') as HTMLCanvasElement);
    requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, instancesBuffer }));

}

}