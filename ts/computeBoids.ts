namespace webgputs {

const particleDim = 8;

export async function asyncBodyOnLoadBoi() {
    const canvas = document.getElementById('world') as HTMLCanvasElement;

    initUI3D(canvas, glMatrix.vec3.fromValues(0, 0, -5));

    const context = initContext(canvas, 'premultiplied');
    
    const render_module = await fetchModule("sprite");

    const vertex_buffer_layouts = render_module.makeVertexBufferLayouts(["a_particlePos", "a_particleVel"]);

    const renderPipeline = g_device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: render_module.module,
            entryPoint: 'vert_main',
            buffers: vertex_buffer_layouts
        },
        fragment: {
            module: render_module.module,
            entryPoint: 'frag_main',
            targets: [
                {
                    format: g_presentationFormat,
                },
            ],
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    const comp = new ComputePipeline();

    await comp.makePipeline("updateSprites");

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

    const vertexBufferData = makeConeSub3(true);

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

    const simParamData = new Float32Array([
        simParams.deltaT,
        simParams.rule1Distance,
        simParams.rule2Distance,
        simParams.rule3Distance,
        simParams.rule1Scale,
        simParams.rule2Scale,
        simParams.rule3Scale,
    ]);
        
    const numParticles = 320;
    const initial_update_Data = new Float32Array(numParticles * particleDim);
    let base = 0;
    for (let i = 0; i < numParticles; ++i) {
        initial_update_Data[base + 0] = 0.0;// 2 * (Math.random() - 0.5);
        initial_update_Data[base + 1] = 0.0;// 2 * (Math.random() - 0.5);
        initial_update_Data[base + 2] = 3.0;// 2 * (Math.random() - 0.5);
        initial_update_Data[base + 3] = 0.0;

        const speed = 5.0;
        initial_update_Data[base + 4] = speed * (Math.random() - 0.5);
        initial_update_Data[base + 5] = speed * (Math.random() - 0.5);
        initial_update_Data[base + 6] = speed * (Math.random() - 0.5);
        initial_update_Data[base + 7] = 0.0;

        base += particleDim;
    }

    comp.makeUpdateBuffers(simParamData, initial_update_Data);

    const mesh = new RenderPipeline(null);
    mesh.pipeline = renderPipeline;
    mesh.vertModule = render_module;
    mesh.makeUniformBufferAndBindGroup();

    let t = 0;
    async function frame() {
        await updateVertexUniformBuffer([mesh]);

        renderPassDescriptor.colorAttachments[0].view = context
            .getCurrentTexture()
            .createView();

        const commandEncoder = g_device.createCommandEncoder();
        {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(comp.pipeline);
            passEncoder.setBindGroup(0, comp.bindGroups[t % 2]);
            passEncoder.dispatchWorkgroups(numParticles);
            passEncoder.end();
        }
        {
            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(renderPipeline);

            passEncoder.setVertexBuffer(0, comp.updateBuffers[(t + 1) % 2]);
            passEncoder.setVertexBuffer(1, spriteVertexBuffer);

            passEncoder.setBindGroup(0, mesh.uniformBindGroup);

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