namespace webgputs {

const particleDim = 8;

export async function fetchText(fileURL: string) {
    const response = await fetch(fileURL);
    const text = await response!.text();

    return text;
}

export async function asyncBodyOnLoadBoi() {
    const spriteWGSL = await fetchText('../wgsl/sprite.wgsl');
    const updateSpritesWGSL = await fetchText('../wgsl/updateSprites.wgsl');

    const canvas = document.getElementById('world') as HTMLCanvasElement;

    initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -5));

    const adapter = await navigator.gpu.requestAdapter();
    console.assert(adapter != null, 'requestAdapter returned null');
    const g_device = await adapter!.requestDevice();

    const [context, presentationFormat] = initContext(g_device, canvas, 'premultiplied');
    
    const spriteShaderModule = g_device.createShaderModule({ code: spriteWGSL });
    const renderPipeline = g_device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: spriteShaderModule,
            entryPoint: 'vert_main',
            buffers: [
                {
                    // instanced particles buffer
                    arrayStride: 4 * particleDim,
                    stepMode: 'instance',
                    attributes: [
                        {
                            // instance position
                            shaderLocation: 0,
                            offset: 0,
                            format: 'float32x4',
                        },
                        {
                            // instance velocity
                            shaderLocation: 1,
                            offset: 4 * 4,
                            format: 'float32x4',
                        },
                    ],
                },
                {
                    // vertex buffer
                    arrayStride: 4 * (3 + 3),
                    stepMode: 'vertex',
                    attributes: [
                        {
                            // vertex positions
                            shaderLocation: 2,
                            offset: 0,
                            format: 'float32x3',
                        }
                        ,
                        {
                            // vertex norms
                            shaderLocation: 3,
                            offset: 4 * 3,
                            format: 'float32x3',
                        }
                    ],
                },
            ],
        },
        fragment: {
            module: spriteShaderModule,
            entryPoint: 'frag_main',
            targets: [
                {
                    format: presentationFormat,
                },
            ],
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    const computePipeline = g_device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: g_device.createShaderModule({
                code: updateSpritesWGSL,
            }),
            entryPoint: 'main',
        },
    });

    const renderPassDescriptor = {
        colorAttachments: [
            {
                view: undefined as any, // Assigned later
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear' as const,
                storeOp: 'store' as const,
            },
        ],
    };

    // prettier-ignore
    const vertexBufferDataOld = new Float32Array([
        -0.01, -0.02, 0.0, 
         0.01, -0.02, 0.0,
         0.0 ,  0.02, 0.0
    ]);

    const vertexBufferData = makeConeSub3(true); // makeSphere3(); // makeCube3();

    const spriteVertexBuffer = g_device.createBuffer({
        size: vertexBufferData.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
    spriteVertexBuffer.unmap();

    const simParams = {
        deltaT: 0.04,
        rule1Distance: 0.1,
        rule2Distance: 0.025,
        rule3Distance: 0.025,
        rule1Scale: 0.02,
        rule2Scale: 0.05,
        rule3Scale: 0.005,
    };

    const simParamBufferSize = 7 * Float32Array.BYTES_PER_ELEMENT;
    const simParamBuffer = g_device.createBuffer({
        size: simParamBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // function updateSimParams() {
    // }
    // updateSimParams();
    g_device.queue.writeBuffer(
        simParamBuffer,
        0,
        new Float32Array([
            simParams.deltaT,
            simParams.rule1Distance,
            simParams.rule2Distance,
            simParams.rule3Distance,
            simParams.rule1Scale,
            simParams.rule2Scale,
            simParams.rule3Scale,
        ])
    );
    console.log("write sim param");

    const numParticles = 320;
    const initialParticleData = new Float32Array(numParticles * particleDim);
    let base = 0;
    for (let i = 0; i < numParticles; ++i) {
        initialParticleData[base + 0] = 0.0;// 2 * (Math.random() - 0.5);
        initialParticleData[base + 1] = 0.0;// 2 * (Math.random() - 0.5);
        initialParticleData[base + 2] = 3.0;// 2 * (Math.random() - 0.5);
        initialParticleData[base + 3] = 0.0;

        const speed = 5.0;
        initialParticleData[base + 4] = speed * (Math.random() - 0.5);
        initialParticleData[base + 5] = speed * (Math.random() - 0.5);
        initialParticleData[base + 6] = speed * (Math.random() - 0.5);
        initialParticleData[base + 7] = 0.0;

        base += particleDim;
    }

    const particleBuffers: GPUBuffer[] = new Array(2);
    for (let i = 0; i < 2; ++i) {
        particleBuffers[i] = g_device.createBuffer({
            size: initialParticleData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Float32Array(particleBuffers[i].getMappedRange()).set(
            initialParticleData
        );
        particleBuffers[i].unmap();
    }

    const computeBindGroups: GPUBindGroup[] = new Array(2);
    for (let i = 0; i < 2; ++i) {
        computeBindGroups[i] = g_device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: simParamBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: particleBuffers[i],
                        offset: 0,
                        size: initialParticleData.byteLength,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: particleBuffers[(i + 1) % 2],
                        offset: 0,
                        size: initialParticleData.byteLength,
                    },
                },
            ],
        });
    }

    const [uniformBuffer, uniformBindGroup] = makeUniformBufferAndBindGroup(g_device, renderPipeline, 4 * (4 * 4 + 4));

    const lightDir = makeLightDir();

    let t = 0;
    function frame() {

        const pvw = ui3D.getTransformationMatrix();
        g_device.queue.writeBuffer(
            uniformBuffer,
            0,
            pvw.buffer,
            pvw.byteOffset,
            pvw.byteLength
        );

        console.assert(pvw.byteLength == 4 * (4 * 4));
        g_device.queue.writeBuffer(
            uniformBuffer,
            pvw.byteLength,
            lightDir.buffer,
            lightDir.byteOffset,
            lightDir.byteLength
        );

        renderPassDescriptor.colorAttachments[0].view = context
            .getCurrentTexture()
            .createView();

        const commandEncoder = g_device.createCommandEncoder();
        {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, computeBindGroups[t % 2]);
            passEncoder.dispatchWorkgroups(numParticles);
            passEncoder.end();
        }
        {
            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(renderPipeline);

            passEncoder.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
            passEncoder.setVertexBuffer(1, spriteVertexBuffer);

            passEncoder.setBindGroup(0, uniformBindGroup);

            passEncoder.draw(vertexBufferData.length / (3 + 3), numParticles, 0, 0);
            passEncoder.end();
        }
        g_device.queue.submit([commandEncoder.finish()]);

        ++t;
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
};

}