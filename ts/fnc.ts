// --- Corrected Matrix Math Helpers ---

function createPerspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovy / 2);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    // FIXED: WebGPU maps depth from 0 to 1, not -1 to 1 like WebGL
    out[10] = far / (near - far);         
    out[11] = -1;
    out[14] = (far * near) / (near - far); 
    return out;
}

function createLookAt(eye: number[], center: number[], up: number[]): Float32Array {
    let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    let len = 1 / Math.hypot(z0, z1, z2);
    z0 *= len; z1 *= len; z2 *= len;

    let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
    len = 1 / Math.hypot(x0, x1, x2);
    x0 *= len; x1 *= len; x2 *= len;

    let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;

    const out = new Float32Array(16);
    out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
    out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
    out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
    out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
    out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
    out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
    out[15] = 1;
    return out;
}

function multiplyMatrices(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    // FIXED: Proper column-major matrix multiplication
    for (let c = 0; c < 4; c++) {       // Column of B
        for (let r = 0; r < 4; r++) {   // Row of A
            out[c * 4 + r] =
                a[0 * 4 + r] * b[c * 4 + 0] +
                a[1 * 4 + r] * b[c * 4 + 1] +
                a[2 * 4 + r] * b[c * 4 + 2] +
                a[3 * 4 + r] * b[c * 4 + 3];
        }
    }
    return out;
}

// --- WebGPU Application ---
export async function initFnc() {
    const canvas = document.getElementById('world') as HTMLCanvasElement;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU not supported");
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    
    context.configure({ device, format });

    // Fetch shaders (assuming shaders.wgsl is in the same folder)
    const response = await fetch('./wgsl/fnc.wgsl');
    const wgslCode = await response.text();
    const shaderModule = device.createShaderModule({ code: wgslCode });

    // 1. Grid Configuration
    const gridSize = 100;
    const extent = 15.0;
    const numVertices = (gridSize + 1) * (gridSize + 1);

    // 2. Generate Indices (CPU)
    const indices = [];
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const row1 = i * (gridSize + 1);
            const row2 = (i + 1) * (gridSize + 1);
            indices.push(row1 + j, row1 + j + 1, row2 + j + 1);
            indices.push(row1 + j, row2 + j + 1, row2 + j);
        }
    }
    const indexData = new Uint16Array(indices);
    const indexBuffer = device.createBuffer({
        size: indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint16Array(indexBuffer.getMappedRange()).set(indexData);
    indexBuffer.unmap();

    // 3. Vertex Buffer (Written by Compute, Read by Render)
    const vertexBuffer = device.createBuffer({
        size: numVertices * 6 * Float32Array.BYTES_PER_ELEMENT, // x, y, z, nx, ny, nz
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    });

    // 4. Compute Uniforms
    const computeUniforms = new Float32Array([gridSize, extent, 0.0, 0.0]); // padding at the end
    const computeUniformBuffer = device.createBuffer({
        size: computeUniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(computeUniformBuffer, 0, computeUniforms);

    // 5. Render Uniforms (MVP Matrix)
    const renderUniformBuffer = device.createBuffer({
        size: 16 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 6. Pipelines
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'compute_main' },
    });

    const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [{
                arrayStride: 24, // 6 floats * 4 bytes
                attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x3' },  // Position
                    { shaderLocation: 1, offset: 12, format: 'float32x3' }  // Normal
                ]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' },
    });

    // 7. Bind Groups
    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: vertexBuffer } },
            { binding: 1, resource: { buffer: computeUniformBuffer } },
        ],
    });

    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: renderUniformBuffer } }],
    });

    // 8. Depth Texture
    const depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // 9. Render Loop
    const startTime = performance.now();
    
    // Matrix setup
    const projMatrix = createPerspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100.0);
    const viewMatrix = createLookAt([20, 20, 20], [0, 0, 0], [0, 0, 1]); // Looking down from an angle
    const mvpMatrix = multiplyMatrices(projMatrix, viewMatrix);
    device.queue.writeBuffer(renderUniformBuffer, 0, mvpMatrix as any);

    function frame() {
        const currentTime = (performance.now() - startTime) / 1000.0;
        
        // Update compute time (offset 8 bytes because gridSize and extent take up 8 bytes)
        device.queue.writeBuffer(computeUniformBuffer, 8, new Float32Array([currentTime]));

        const commandEncoder = device.createCommandEncoder();

        // --- COMPUTE PASS ---
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        const workgroups = Math.ceil((gridSize + 1) / 8);
        computePass.dispatchWorkgroups(workgroups, workgroups);
        computePass.end();

        // --- RENDER PASS ---
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        });

        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        renderPass.drawIndexed(indices.length);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

// initFnc().catch(console.error);