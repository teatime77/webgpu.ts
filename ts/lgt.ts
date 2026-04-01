import { fetchText, msg } from "@i18n";
import { stopAnimation } from "./instance";

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
export async function runLGT(device: GPUDevice, theory: 'U1' | 'SU2' = 'U1'): Promise<() => void> {
    msg(`Starting 2D ${theory} Lattice Gauge Theory simulation.`);
    stopAnimation(); // Stop any previous animation

    const L = 32; // Must match const in lgt_u1.wgsl
    const L_squared = L * L;
    const WORKGROUP_SIZE = 8; // Must match @workgroup_size in lgt_u1.wgsl

    // --- Create UI for controlling beta ---
    let betaSlider: HTMLInputElement;
    let betaValueSpan: HTMLSpanElement;
    let initialBeta = 5.5;
    let vizMode: 'plaquette' | 'vortex' = theory === 'U1' ? 'vortex' : 'plaquette';

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
        // Vortices are a U(1) concept.
        if (theory === 'U1') {
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
        }

        const avgPlaquetteSpan = document.createElement('span');
        avgPlaquetteSpan.id = 'avg-plaquette-value';
        avgPlaquetteSpan.style.display = 'block';
        avgPlaquetteSpan.style.margin = '10px';
        avgPlaquetteSpan.innerText = 'Avg Plaquette: ...';
        controlsContainer.appendChild(avgPlaquetteSpan);

        document.getElementById('span-buttons')?.insertAdjacentElement('afterend', controlsContainer);
    } else {
        betaSlider = document.getElementById('beta-slider') as HTMLInputElement;
        betaValueSpan = document.getElementById('beta-value') as HTMLSpanElement;
        initialBeta = parseFloat(betaSlider.value);
        if (theory === 'U1') {
            const vortexRadio = document.getElementById('viz-vortex') as HTMLInputElement;
            vizMode = vortexRadio.checked ? 'vortex' : 'plaquette';
        }
    }

    if (!device) {
        msg("GPUDevice not found. LGT simulation cannot start.");
        return () => {};
    }

    // 1. Fetch shader code
    const shaderCode = await fetchText(theory === 'U1' ? './wgsl/lgt_u1.wgsl' : './wgsl/lgt_su2.wgsl');
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // 2. Create Buffers
    // Uniforms: { beta: f32, update_subset: u32 }
    const simParams = new Float32Array([initialBeta, 0]); 
    const paramsBuffer = device.createBuffer({
        size: simParams.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, simParams);

    // --- Measurement Setup ---
    // Bessel functions for theoretical calculation
    function besselI0(x: number, terms = 15): number {
        let sum = 1.0;
        let term = 1.0;
        for (let k = 1; k <= terms; k++) {
            term *= (x * x) / (4 * k * k);
            sum += term;
        }
        return sum;
    }

    function besselI1(x: number, terms = 15): number {
        let sum = 0.0;
        let term = x / 2.0;
        sum += term;
        for (let k = 1; k <= terms; k++) {
            term *= (x * x) / (4 * k * (k + 1));
            sum += term;
        }
        return sum;
    }

    function besselI2(x: number, terms = 15): number {
        let sum = 0.0;
        // k=0 term: 1/(0! * 2!) * (x/2)^2 = (x*x)/8
        let term = (x * x) / 8.0;
        sum += term;
        for (let k = 1; k <= terms; k++) {
            term *= (x * x) / (4 * k * (k + 2));
            sum += term;
        }
        return sum;
    }

    // Debounce helper to avoid overwhelming the system with measurements
    function debounce(func: (...args: any[]) => void, delay: number) {
        let timeoutId: number;
        return (...args: any[]) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                func(...args);
            }, delay);
        };
    }

    const avgPlaquetteSpan = document.getElementById('avg-plaquette-value') as HTMLSpanElement;
    async function performMeasurement() {
        // This function now only reads the buffer and calculates the average.
        // The caller is responsible for ensuring the correct data is in the readback buffer.

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readbackBuffer.getMappedRange());
        let sum = 0;
        for (let i = 0; i < L_squared; i++) {
            sum += data[i];
        }
        const simulatedAvg = sum / L_squared;
        readbackBuffer.unmap();

        const currentBeta = parseFloat(betaSlider.value);
        let text: string;

        if (theory === 'U1') {
            const theoreticalAvg = besselI1(currentBeta) / besselI0(currentBeta);
            text = `Avg Plaquette: Sim = ${simulatedAvg.toFixed(4)}, Theory = ${theoreticalAvg.toFixed(4)}`;
            avgPlaquetteSpan.innerText = text;
        } else { // SU(2)
            const theoreticalAvg = besselI2(currentBeta) / besselI1(currentBeta);
            text = `Avg Plaquette: Sim = ${simulatedAvg.toFixed(4)}, Theory = ${theoreticalAvg.toFixed(4)}`;
            avgPlaquetteSpan.innerText = text;
        }
        console.log(`Beta = ${currentBeta.toFixed(1)} -> ${text}`);
    }

    // Link variables: vec2<f32> per link
    const link_vec_size = theory === 'U1' ? 2 : 4; // U(1) uses vec2, SU(2) uses vec4
    const linksBuffer = device.createBuffer({
        // 2 directions * L*L sites * vecN<f32> (vec2 for U(1), vec4 for SU(2))
        size: 2 * L_squared * link_vec_size * 4,
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

    // A buffer on the CPU side to read back the results.
    const readbackBuffer = device.createBuffer({
        size: L_squared * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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
    let measureVortexPipeline: GPUComputePipeline | null = null;
    if (theory === 'U1') {
        measureVortexPipeline = await device.createComputePipelineAsync({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'measure_vortices' },
        });
    }

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

    // --- State, Constants, and New Simulation Logic ---
    let animationId: number | null = null;
    let isThermalizing = false;
    const thermalizationSweeps = 1500; // Number of sweeps for equilibration
    const sweepsPerFrame = 10; // Sweeps for live visualization

    // Create a staging buffer for subset indices. This is more efficient than
    // calling writeBuffer every time, as it allows us to schedule all uniform
    // updates within a single command encoder.
    const subsetStagingBuffer = device.createBuffer({
        size: 16, // 4 * u32
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint32Array(subsetStagingBuffer.getMappedRange()).set([0, 1, 2, 3]);
    subsetStagingBuffer.unmap();

    // This new function handles the entire thermalization and measurement process.
    async function thermalizeAndMeasure() {
        if (isThermalizing) return; // Don't start a new process if one is running
        isThermalizing = true;

        // 1. Update UI to show we're busy and stop the animation loop.
        // The animation loop will just idle by checking `isThermalizing`.
        betaSlider.disabled = true;
        avgPlaquetteSpan.innerText = `Thermalizing for ${thermalizationSweeps} sweeps...`;

        // 2. Run thermalization sweeps. We do this in chunks to avoid making the
        //    GPU unresponsive, which could cause the OS to reset the driver (TDR).
        const sweepsPerChunk = 100;
        const numChunks = Math.ceil(thermalizationSweeps / sweepsPerChunk);

        for (let chunk = 0; chunk < numChunks; chunk++) {
            const commandEncoder = device.createCommandEncoder();
            for (let i = 0; i < sweepsPerChunk; i++) {
                // A full Monte Carlo sweep requires updating all 4 checkerboard subsets.
                for (let j = 0; j < 4; j++) {
                    commandEncoder.copyBufferToBuffer(subsetStagingBuffer, j * 4, paramsBuffer, 4, 4);
                    const pass = commandEncoder.beginComputePass();
                    pass.setPipeline(updatePipeline);
                    pass.setBindGroup(0, bindGroup);
                    pass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
                    pass.end();
                }
            }
            device.queue.submit([commandEncoder.finish()]);
            await device.queue.onSubmittedWorkDone(); // Wait for the chunk to complete.
        }

        // 3. Now that the system is in equilibrium, perform a measurement.
        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(measurePlaquettePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
        pass.end();

        // Copy the measurement result to a buffer we can read on the CPU.
        commandEncoder.copyBufferToBuffer(vizResultBuffer, 0, readbackBuffer, 0, vizResultBuffer.size);
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // 4. Read the data and update the UI text.
        await performMeasurement();

        // 5. Re-enable the UI. The animation loop will resume its work automatically.
        isThermalizing = false;
        betaSlider.disabled = false;
    }

    // --- New Event Listeners ---
    const debouncedThermalizer = debounce(() => {
        thermalizeAndMeasure();
    }, 500);

    betaSlider.oninput = () => {
        const newBeta = parseFloat(betaSlider.value);
        betaValueSpan.innerText = newBeta.toFixed(1);
        // Update the beta value in the uniform buffer.
        device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([newBeta]));
        // Trigger a new thermalization and measurement cycle.
        debouncedThermalizer();
    };

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

    // Perform an initial thermalization and measurement on startup.
    thermalizeAndMeasure();

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
    const renderParams = new Uint32Array([vizMode === 'vortex' ? 1 : 0]); // 0=plaquette, 1=vortex (U(1) only)
    const renderParamsBuffer = device.createBuffer({
        size: renderParams.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(renderParamsBuffer, 0, renderParams);
    
    // Update vizMode and the render parameter when the radio button changes.
    // This no longer triggers a measurement.
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
    function frame() {
        // If thermalization is in progress, just idle and wait for the next frame.
        if (isThermalizing) {
            animationId = requestAnimationFrame(frame);
            return;
        }

        const commandEncoder = device.createCommandEncoder();

        // 1. Evolve the system for a few sweeps for live visualization.
        // This loop is now correct, using multiple compute passes to update the
        // checkerboard subsets properly.
        for (let i = 0; i < sweepsPerFrame; i++) {
            for (let j = 0; j < 4; j++) {
                commandEncoder.copyBufferToBuffer(subsetStagingBuffer, j * 4, paramsBuffer, 4, 4);
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(updatePipeline);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
                pass.end();
            }
        }
        
        // 2. Run the measurement kernel for the current visualization mode.
        const vizPass = commandEncoder.beginComputePass();
        if (vizMode === 'vortex' && measureVortexPipeline) {
            vizPass.setPipeline(measureVortexPipeline);
        } else {
            vizPass.setPipeline(measurePlaquettePipeline);
        }
        vizPass.setBindGroup(0, bindGroup);
        vizPass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
        vizPass.end();

        // --- Render Pass ---
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
        
        animationId = requestAnimationFrame(frame);
    }

    frame(); // Start the animation loop.
    
    // Return a "stopper" function to be called when this animation should end.
    return () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        // Clean up GPU resources
        subsetStagingBuffer.destroy();
        // Other buffers are managed by the parent context that passed in the device.
    };
}
