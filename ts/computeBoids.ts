namespace webgputs {

export const particleDim = 8;

export function makeInitialInstanceArray() : Float32Array {
    const numParticles = 320;
    const initial_instance_array = new Float32Array(numParticles * particleDim);
    let base = 0;
    for (let i = 0; i < numParticles; ++i) {
        initial_instance_array[base + 0] = 0.0;// 2 * (Math.random() - 0.5);
        initial_instance_array[base + 1] = 0.0;// 2 * (Math.random() - 0.5);
        initial_instance_array[base + 2] = 3.0;// 2 * (Math.random() - 0.5);
        initial_instance_array[base + 3] = 0.0;

        const speed = 5.0;
        initial_instance_array[base + 4] = speed * (Math.random() - 0.5);
        initial_instance_array[base + 5] = speed * (Math.random() - 0.5);
        initial_instance_array[base + 6] = speed * (Math.random() - 0.5);
        initial_instance_array[base + 7] = 0.0;

        base += particleDim;
    }

    return initial_instance_array;
}

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
        

    const mesh = new RenderPipeline();
    mesh.pipeline = renderPipeline;
    mesh.vertModule = render_module;
    mesh.makeUniformBufferAndBindGroup();

    //-------------------------------------------------- Compute Pipeline
    const comp = new ComputePipeline();
    await comp.makePipeline("updateSprites");
    const compute_uniform_array = makeComputeUniformArray();
    comp.makeUniformBuffer(compute_uniform_array);

    const initial_instance_array = makeInitialInstanceArray();
    comp.instanceCount = initial_instance_array.length / particleDim;
    comp.makeUpdateBuffers(initial_instance_array);


    let tick = 0;

    //-------------------------------------------------- frame
    async function frame() {
        await updateVertexUniformBuffer([mesh]);

        renderPassDescriptor.colorAttachments[0].view = context
            .getCurrentTexture()
            .createView();

        const commandEncoder = g_device.createCommandEncoder();
        {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(comp.pipeline);
            passEncoder.setBindGroup(0, comp.bindGroups[tick % 2]);
            passEncoder.dispatchWorkgroups(comp.instanceCount);
            passEncoder.end();
        }
        {
            const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            passEncoder.setPipeline(renderPipeline);

            passEncoder.setVertexBuffer(0, comp.updateBuffers[(tick + 1) % 2]);
            passEncoder.setVertexBuffer(1, spriteVertexBuffer);

            passEncoder.setBindGroup(0, mesh.uniformBindGroup);

            passEncoder.draw(vertexBufferData.length / (3 + 3), comp.instanceCount, 0, 0);
            passEncoder.end();
        }
        g_device.queue.submit([commandEncoder.finish()]);

        ++tick;
        requestId = requestAnimationFrame(frame);
    }
    requestId = requestAnimationFrame(frame);
};

}