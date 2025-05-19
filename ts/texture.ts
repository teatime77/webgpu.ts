var glMatrix: any;

namespace webgpu_ts {

export async function asyncBodyOnLoadTex() {

    const cubeVertexSize = 4 * 10; // Byte size of one vertex.
    const cubePositionOffset = 4 * 0;
    const cubeColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
    const cubeUVOffset = 4 * 8; // Byte offset of cube vertex color attribute.
    const cubeVertexCount = 36;

    const cubeVertexArray = new Float32Array([
        // float4 position, float4 color float2 uv
        1, -1, 1, 1, 1, 0, 1, 1, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1, 0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1, 0, 0,
        1, -1, -1, 1, 1, 0, 0, 1, 1, 0,
        1, -1, 1, 1, 1, 0, 1, 1, 1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1, 0, 0,

        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, -1, 1, 1, 1, 0, 1, 1, 0, 1,
        1, -1, -1, 1, 1, 0, 0, 1, 0, 0,
        1, 1, -1, 1, 1, 1, 0, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, -1, -1, 1, 1, 0, 0, 1, 0, 0,

        -1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 1, -1, 1, 1, 1, 0, 1, 0, 0,
        -1, 1, -1, 1, 0, 1, 0, 1, 1, 0,
        -1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
        1, 1, -1, 1, 1, 1, 0, 1, 0, 0,

        -1, -1, 1, 1, 0, 0, 1, 1, 1, 1,
        -1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1, 0, 0,
        -1, -1, -1, 1, 0, 0, 0, 1, 1, 0,
        -1, -1, 1, 1, 0, 0, 1, 1, 1, 1,
        -1, 1, -1, 1, 0, 1, 0, 1, 0, 0,

        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        -1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        -1, -1, 1, 1, 0, 0, 1, 1, 0, 0,
        -1, -1, 1, 1, 0, 0, 1, 1, 0, 0,
        1, -1, 1, 1, 1, 0, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        1, -1, -1, 1, 1, 0, 0, 1, 1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1, 0, 0,
        1, 1, -1, 1, 1, 1, 0, 1, 1, 0,
        1, -1, -1, 1, 1, 0, 0, 1, 1, 1,
        -1, 1, -1, 1, 0, 1, 0, 1, 0, 0,
    ]);

    const vertWGSL = await fetchText('../wgsl/texture-vert.wgsl');

    const fragWGSL = await fetchText('../wgsl/texture-frag.wgsl');

    async function init(canvas: HTMLCanvasElement): Promise<{ context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture, texture: GPUTexture }> {

        const context = initContext(canvas, 'opaque');

        initUI3D(canvas);
        editor.camera.camDistance = -4;

        // create a render pipeline
        const pipeline = g_device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: makeShaderModule(vertWGSL),
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
                            {
                                // uv
                                shaderLocation: 2, // @location(2) in vertex shader
                                offset: cubeUVOffset,
                                format: 'float32x2',
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: makeShaderModule(fragWGSL),
                entryPoint: 'main',
                targets: [
                    // 0
                    { // @location(0) in fragment shader
                        format: g_presentationFormat,
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

        let texture: GPUTexture;
        {
            const img = document.createElement('img');
            img.crossOrigin = 'Anonymous';
            img.src = 'https://storage.googleapis.com/emadurandal-3d-public.appspot.com/images/pexels-james-wheeler-1552212.jpg';
            await img.decode();
            const imageBitmap = await createImageBitmap(img);

            texture = g_device.createTexture({
                size: [imageBitmap.width, imageBitmap.height, 1],
                format: 'rgba8unorm',
                usage:
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });
            g_device.queue.copyExternalImageToTexture(
                { source: imageBitmap },
                { texture: texture },
                [imageBitmap.width, imageBitmap.height]
            );
        }

        const sampler = g_device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
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
                {
                    binding: 1,
                    resource: texture.createView(),
                },
                {
                    binding: 2,
                    resource: sampler,
                },
            ],
        });


        return { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture };
    }

    function frame(
        { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture }:
            { context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture, texture: GPUTexture }
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

        ui3D.getTransformationMatrix();

        let worldMatrix = glMatrix.mat4.create();
        let pvw = glMatrix.mat4.create();

        glMatrix.mat4.mul(pvw, ui3D.ProjViewMatrix, worldMatrix);

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
        requestId = requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture }));
    }

    const { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture } = await init(document.getElementById('world') as HTMLCanvasElement);
    requestId = requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture }));

}

}
