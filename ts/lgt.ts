import { fetchText, msg } from "@i18n";
import { stopAnimation } from "./instance";
import { ComputePipeline } from "./compute";

/**
 * This function sets up and runs a 2D U(1) Lattice Gauge Theory simulation.
 * It's a self-contained example to show the steps involved:
 * 1. Create GPU buffers for parameters, link variables, and RNG state.
 * 2. Create compute pipelines for initialization and updating the lattice.
 * 3. Run the initialization shader once.
 * 4. Start an animation loop to continuously run the update shader.
 * NOTE: This assumes a global `device: GPUDevice` object is available,
 * which is typical in WebGPU applications and seems to be the case in your framework.
 */
export async function runLGT(device: GPUDevice): Promise<() => void> {
    msg("Starting 2D U(1) Lattice Gauge Theory simulation.");
    stopAnimation(); // Stop any previous animation

    const L = 32; // Must match const in lgt_u1.wgsl
    const L_squared = L * L;
    const WORKGROUP_SIZE = 8; // Must match @workgroup_size in lgt_u1.wgsl

    // --- Create UI for controlling beta ---
    let betaSlider: HTMLInputElement;
    let betaValueSpan: HTMLSpanElement;
    let initialBeta = 5.5;
    let vizMode: 'plaquette' | 'vortex' = 'vortex';

    const controlsContainerId = 'lgt-controls-container';
    let controlsContainer = document.getElementById(controlsContainerId);

    if (!controlsContainer) {
        controlsContainer = document.createElement('div');
        controlsContainer.id = controlsContainerId;
        controlsContainer.style.margin = '10px';

        const betaLabel = document.createElement('label');
        betaLabel.innerText = 'Beta (β): ';
        betaLabel.htmlFor = 'beta-slider';

        betaSlider = document.createElement('input');
        betaSlider.id = 'beta-slider';
        betaSlider.type = 'range';
        betaSlider.min = '0.1';
        betaSlider.max = '10.0';
        betaSlider.step = '0.1';
        betaSlider.value = String(initialBeta);

        betaValueSpan = document.createElement('span');
        betaValueSpan.id = 'beta-value';
        betaValueSpan.innerText = betaSlider.value;

        controlsContainer.appendChild(betaLabel);
        controlsContainer.appendChild(betaSlider);
        controlsContainer.appendChild(betaValueSpan);

        // --- Create UI for visualization mode ---
        const vizControlsContainer = document.createElement('div');
        vizControlsContainer.id = 'lgt-viz-controls';
        vizControlsContainer.style.margin = '10px';
        vizControlsContainer.innerHTML = `
            <span style="margin-right: 10px;">Visualization:</span>
            <input type="radio" id="viz-vortex" name="viz-mode" value="vortex" checked>
            <label for="viz-vortex" style="margin-right: 10px;">Vortices</label>
            <input type="radio" id="viz-plaquette" name="viz-mode" value="plaquette">
            <label for="viz-plaquette">Plaquette Energy</label>
        `;

        controlsContainer.appendChild(vizControlsContainer);

        document.getElementById('span-buttons')?.insertAdjacentElement('afterend', controlsContainer);
    } else {
        betaSlider = document.getElementById('beta-slider') as HTMLInputElement;
        betaValueSpan = document.getElementById('beta-value') as HTMLSpanElement;
        initialBeta = parseFloat(betaSlider.value);
        const vortexRadio = document.getElementById('viz-vortex') as HTMLInputElement;
        vizMode = vortexRadio.checked ? 'vortex' : 'plaquette';
    }

    if (!device) {
        msg("GPUDevice not found. LGT simulation cannot start.");
        return () => {};
    }

    // 1. Fetch shader code
    const shaderCode = await fetchText('./wgsl/lgt_u1.wgsl');
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // 2. Create Buffers
    // Uniforms: { beta: f32, update_subset: u32 }
    // The 'beta' parameter controls the "order" of the system.
    // beta > ~1.0 (weak coupling): System should be ordered (mostly red).
    // beta < ~1.0 (strong coupling): System should be disordered (random red/blue static).
    //
    // TRY CHANGING THIS VALUE!
    const simParams = new Float32Array([initialBeta, 0]); 
    const paramsBuffer = device.createBuffer({
        size: simParams.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, simParams);

    // Add event listener to update beta in real-time
    betaSlider.oninput = () => {
        const newBeta = parseFloat(betaSlider.value);
        betaValueSpan.innerText = newBeta.toFixed(1);
        // Write the new beta value to the uniform buffer
        device.queue.writeBuffer(
            paramsBuffer,
            0, // Offset of beta (it's the first value)
            new Float32Array([newBeta])
        );
    };

    // Link variables: vec2<f32> per link
    const linksBuffer = device.createBuffer({
        size: 2 * L_squared * 2 * 4, // 2 directions * L*L sites * vec2<f32> (8 bytes)
        usage: GPUBufferUsage.STORAGE,
    });

    // RNG state: u32 per site, initialized on CPU
    const rngState = new Uint32Array(L_squared);
    for (let i = 0; i < L_squared; i++) rngState[i] = Math.random() * 0xFFFFFFFF;
    const rngBuffer = device.createBuffer({
        size: rngState.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(rngBuffer.getMappedRange()).set(rngState);
    rngBuffer.unmap();

    // Plaquette results buffer (for measurement)
    const vizResultBuffer = device.createBuffer({
        size: L_squared * 4, // L*L sites * f32 (can be bitcast from i32)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // 3. Create Pipelines & Bind Group
    // Explicitly define the bind group layout instead of using 'auto'.
    // This is more robust and avoids potential issues with 'auto' layout support.
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { // @binding(0) var<uniform> params
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
            },
            { // @binding(1) var<storage, read_write> links
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            },
            { // @binding(2) var<storage, read_write> rng_state
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            },
            { // @binding(3) var<storage, read_write> viz_results
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            }
        ]
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    const initPipeline = await device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'init_hot' },
    });
    const updatePipeline = await device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'metropolis_update' },
    });
    const measureVortexPipeline = await device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'measure_vortices' },
    });
    const measurePlaquettePipeline = await device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'measure_plaquette' },
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: linksBuffer } },
            { binding: 2, resource: { buffer: rngBuffer } },
            { binding: 3, resource: { buffer: vizResultBuffer } },
        ],
    });

    // 4. Run Initialization
    let commandEncoder = device.createCommandEncoder();
    let passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(initPipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    msg("LGT: Lattice initialized.");

    // --- Create Render Pipeline for Visualization ---
    const renderShaderCode = await fetchText('./wgsl/lgt_render.wgsl');
    const renderShaderModule = device.createShaderModule({ code: renderShaderCode });

    // We need the canvas context to know the format of the texture we're rendering to.
    const canvas = document.querySelector("canvas")!;
    const context = canvas.getContext("webgpu")!;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: presentationFormat,
        alphaMode: "premultiplied"
    });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { // @binding(0) var<storage, read> viz_results
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'read-only-storage' } // Best practice for read-only in render
            },
            { // @binding(1) var<uniform> render_params
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' }
            }
        ]
    });

    // Create a uniform buffer for render parameters (viz_mode)
    const renderParams = new Uint32Array([vizMode === 'vortex' ? 1 : 0]); // 0=plaquette, 1=vortex
    const renderParamsBuffer = device.createBuffer({
        size: renderParams.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(renderParamsBuffer, 0, renderParams);

    // Add event listeners for the radio buttons
    document.querySelectorAll<HTMLInputElement>('input[name="viz-mode"]').forEach(radio => {
        radio.onchange = () => {
            vizMode = radio.value as 'plaquette' | 'vortex';
            const vizModeValue = vizMode === 'vortex' ? 1 : 0;
            device.queue.writeBuffer(renderParamsBuffer, 0, new Uint32Array([vizModeValue]));
        };
    });

    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: vizResultBuffer } },
            { binding: 1, resource: { buffer: renderParamsBuffer } }
        ]
    });

    const renderPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupLayout]
    });

    const renderPipeline = await device.createRenderPipelineAsync({
        layout: renderPipelineLayout,
        vertex: {
            module: renderShaderModule,
            entryPoint: 'vs_main',
        },
        fragment: {
            module: renderShaderModule,
            entryPoint: 'fs_main',
            targets: [{ format: presentationFormat }],
        },
        primitive: { topology: 'triangle-list' },
    });

    // 5. Start a custom animation loop to run the update shader
    let animationId: number | null = null;
    let frameCount = 0;
    const sweepsPerFrame = 10; // Number of Monte Carlo sweeps per rendered frame

    function frame() {
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        // 1. Evolve the system with multiple sweeps
        for (let i = 0; i < sweepsPerFrame; i++) {
            // Update the 'update_subset' uniform to avoid race conditions.
            // This cycles through 4 checkerboard patterns (0, 1, 2, 3).
            // We need to ensure the subset index is unique for each update step.
            const subsetIndex = (frameCount * sweepsPerFrame + i) % 4;
            device.queue.writeBuffer(
                paramsBuffer,
                4, // Offset of update_subset (f32 beta is 4 bytes)
                new Uint32Array([subsetIndex])
            );

            passEncoder.setPipeline(updatePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
        }
        
        // 2. Measure the result of the evolution
        if (vizMode === 'vortex') {
            passEncoder.setPipeline(measureVortexPipeline);
        } else {
            passEncoder.setPipeline(measurePlaquettePipeline);
        }
        passEncoder.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);

        passEncoder.end();

        // --- Render Pass ---
        // Get the current texture from the canvas to render to.
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
        };
        const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.setBindGroup(0, renderBindGroup);
        renderPassEncoder.draw(6, L_squared, 0, 0); // Draw L*L instances of a 6-vertex quad.
        renderPassEncoder.end();
        device.queue.submit([commandEncoder.finish()]);

        frameCount++;
        animationId = requestAnimationFrame(frame);
    }
    frame(); // Start the loop

    // Return a "stopper" function to be called when this animation should end.
    return () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
    };
}
