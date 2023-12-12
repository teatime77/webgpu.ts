var glMatrix: any;

namespace webgputs {

export async function asyncBodyOnLoadDep() {

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

    const vertWGSL = await fetchText('../wgsl/shape-vert.wgsl');

    const fragWGSL = await fetchText('../wgsl/depth-frag.wgsl');

    const g_adapter = await navigator.gpu.requestAdapter();
    const g_device = await g_adapter!.requestDevice();

    async function init(canvas: HTMLCanvasElement): Promise<{ context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture }> {
        const [context, presentationFormat] = initContext(g_device, canvas, 'opaque');

        initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -4));

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

        const [uniformBuffer, uniformBindGroup] = makeUniformBufferAndBindGroup(g_device, pipeline, uniformBufferSize);

        // Create a vertex buffer from the quad data.
        const verticesBuffer = g_device.createBuffer({
            size: cubeVertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(verticesBuffer.getMappedRange()).set(cubeVertexArray);
        verticesBuffer.unmap();

        const depthTexture = g_device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        return { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture };
    }

    function frame(
        { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture }:
            { context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture }
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

        const pvw = ui3D.getTransformationMatrix();

        g_device.queue.writeBuffer(
            uniformBuffer, 4 * 16 * 2, 
            pvw.buffer, pvw.byteOffset, pvw.byteLength
        );

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, verticesBuffer);
        passEncoder.draw(cubeVertexCount);
        passEncoder.end();

        g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture }));
    }

    const { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture } = await init(document.getElementById('world') as HTMLCanvasElement);
    requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture }));

}

}