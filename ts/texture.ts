var glMatrix: any;

namespace webgputs {

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


    const quadIndexArray = new Uint16Array([0, 1, 2, 0, 2, 3]);

    const vertWGSL = `
    struct Uniforms {
      projectionMatrix : mat4x4<f32>,
      viewMatrix : mat4x4<f32>,
      worldMatrix : mat4x4<f32>,
    }
    @binding(0) @group(0) var<uniform> uniforms : Uniforms;
    
    struct VertexOutput {
      @builtin(position) Position : vec4<f32>,
      @location(0) fragUV : vec2<f32>,
    }
    
    @vertex
    fn main(
      @location(0) position: vec4<f32>,
      @location(1) color: vec4<f32>,
      @location(2) uv: vec2<f32>  
    ) -> VertexOutput {
    
        var output : VertexOutput;
        output.Position = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.worldMatrix * position;
      output.fragUV = uv;
      
      return output;
    }
    `;

    const fragWGSL = `
    @group(0) @binding(1) var myTexture: texture_2d<f32>;
    @group(0) @binding(2) var mySampler: sampler;
    
    @fragment
    fn main(
      @location(0) fragUV: vec2<f32>,
    ) -> @location(0) vec4<f32> {
      return textureSample(myTexture, mySampler, fragUV);
    }
    `;

    const g_adapter = await navigator.gpu.requestAdapter();
    const g_device = await g_adapter!.requestDevice();

    async function init(canvas: HTMLCanvasElement): Promise<{ context: GPUCanvasContext, pipeline: GPURenderPipeline, verticesBuffer: GPUBuffer, uniformBindGroup: GPUBindGroup, uniformBuffer: GPUBuffer, depthTexture: GPUTexture, texture: GPUTexture }> {

        const context = canvas.getContext('webgpu') as GPUCanvasContext;

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        const devicePixelRatio = window.devicePixelRatio || 1;
        const presentationSize = [
            canvas.clientWidth * devicePixelRatio,
            canvas.clientHeight * devicePixelRatio,
        ];
        canvas.width = presentationSize[0];
        canvas.height = presentationSize[1];

        context.configure({
            device: g_device,
            format: presentationFormat,
            alphaMode: 'opaque',
        });

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

    function getTransformationMatrix(uniformBuffer: GPUBuffer) {
        const projectionMatrix = glMatrix.mat4.create();
        glMatrix.mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, 1, 1, 100.0);
        g_device.queue.writeBuffer(
            uniformBuffer,
            4 * 16 * 0,
            projectionMatrix.buffer,
            projectionMatrix.byteOffset,
            projectionMatrix.byteLength
        );


        const viewMatrix = glMatrix.mat4.create();
        glMatrix.mat4.translate(viewMatrix, viewMatrix, glMatrix.vec3.fromValues(0, 0, -4));
        g_device.queue.writeBuffer(
            uniformBuffer,
            4 * 16 * 1,
            viewMatrix.buffer,
            viewMatrix.byteOffset,
            viewMatrix.byteLength
        );

        const worldMatrix = glMatrix.mat4.create();
        const now = Date.now() / 1000;
        glMatrix.mat4.rotate(
            worldMatrix,
            worldMatrix,
            1,
            glMatrix.vec3.fromValues(Math.sin(now), Math.cos(now), 0)
        );
        g_device.queue.writeBuffer(
            uniformBuffer,
            4 * 16 * 2,
            worldMatrix.buffer,
            worldMatrix.byteOffset,
            worldMatrix.byteLength
        );
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

        getTransformationMatrix(uniformBuffer);


        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, verticesBuffer);
        passEncoder.draw(cubeVertexCount);
        passEncoder.end();

        g_device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture }));
    }

    const { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture } = await init(document.getElementById('world') as HTMLCanvasElement);
    requestAnimationFrame(frame.bind(frame, { context, pipeline, verticesBuffer, uniformBindGroup, uniformBuffer, depthTexture, texture }));

}

}
