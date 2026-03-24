import { fetchText } from "@i18n";

// Helper to get a seeded random number generator
function mulberry32(a: number) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

// Box-Muller transform to get a random number from a standard normal distribution
function randomNormal(randFn: () => number): () => number {
    let z2: number | null = null;
    return function(): number {
        if (z2 !== null) {
            const tmp = z2;
            z2 = null;
            return tmp;
        }
        let u1 = 0, u2 = 0;
        while (u1 === 0) u1 = randFn(); //Converting [0,1) to (0,1)
        while (u2 === 0) u2 = randFn();
        const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        z2 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
        return z1;
    };
}

// Helper to create a buffer and write data to it
async function createBuffer(device: GPUDevice, data: Float32Array, usage: GPUBufferUsageFlags): Promise<GPUBuffer> {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
}

// Helper to read data from a buffer
async function readBuffer(device: GPUDevice, buffer: GPUBuffer): Promise<Float32Array> {
    const size = buffer.size;
    const readBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readBuffer.getMappedRange());
    const result = new Float32Array(data); // Copy data before unmapping
    readBuffer.unmap();
    readBuffer.destroy();
    return result;
}

// This function runs the entire HMC simulation.
export async function runHMC() {

    // --- 0. Setup ---
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("WebGPU not supported");
    }
    const device = await adapter.requestDevice();
    const rand = mulberry32(42);
    const randn = randomNormal(rand);

    // --- 1. Data Preparation (same as Python example) ---
    const N = 30;
    const X1 = new Float32Array(N).map(() => 5 + rand() * 15);
    const X2 = new Float32Array(N).map(() => 20 + rand() * 15);
    const true_beta0 = 100, true_beta1 = 20, true_beta2 = 10, true_sigma = 30;
    const Y = new Float32Array(N).map((_, i) => {
        const mu_true = true_beta0 + true_beta1 * X1[i] + true_beta2 * X2[i];
        return mu_true + randn() * true_sigma;
    });

    // --- 2. HMC Configuration ---
    const iterations = 30000;
    const burn_in = 3000;
    const num_chains = 256; // Run multiple independent chains in parallel
    const workgroup_size = 64;
    const epsilon = 0.005;
    const L = 15;
    const initial_theta = new Float32Array([0.0, 0.0, 0.0, Math.log(10.0)]);

    // --- 3. WebGPU Resources ---
    const shaderCode = await fetchText('./wgsl/hmc.wgsl');
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // --- Constant Uniforms and Data Buffers ---
    const paramsBuffer = device.createBuffer({
        // Padded to 32 bytes for uniform buffer alignment best practices
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsData = new ArrayBuffer(32);
    const paramsView = new DataView(paramsData);
    // With eps_grad removed from the shader struct, the offsets have shifted.
    paramsView.setFloat32(0, epsilon, true);    // epsilon is now at offset 0
    paramsView.setUint32(4, L, true);           // L is now at offset 4
    paramsView.setUint32(8, N, true);           // N is now at offset 8
    paramsView.setUint32(12, num_chains, true); // num_chains is now at offset 12
    paramsView.setUint32(16, Math.floor(Math.random() * 100000), true); // seed at offset 16
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const dataX1Buffer = await createBuffer(device, X1, GPUBufferUsage.STORAGE);
    const dataX2Buffer = await createBuffer(device, X2, GPUBufferUsage.STORAGE);
    const dataYBuffer = await createBuffer(device, Y, GPUBufferUsage.STORAGE);
    
    // --- Buffers for the HMC step (read/write) ---
    // Create initial positions for all chains
    const initial_qs = new Float32Array(num_chains * 4);
    for (let i = 0; i < num_chains; i++) {
        initial_qs.set(initial_theta, i * 4);
    }

    // Ping-pong buffers for chain positions. We read from one and write to the other.
    const q_buffer_A = await createBuffer(device, initial_qs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const q_buffer_B = device.createBuffer({
        size: q_buffer_A.size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    
    // Output buffer for acceptance count (a single atomic integer)
    const accepted_buffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Large buffer on the GPU to accumulate all samples from all iterations
    const samples_buffer = device.createBuffer({
        size: num_chains * iterations * 4 * 4, // num_chains * iterations * sizeof(vec4f)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Buffer to hold the state of the PRNG for each chain on the GPU.
    const initial_rand_state = new Uint32Array(num_chains).map(() => Math.floor(Math.random() * 100000));
    const rand_state_buffer = await createBuffer(device, initial_rand_state as any, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Bind Group Layouts and Pipelines
    const commonBindGroupLayout = device.createBindGroupLayout({ entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } } ] });
    const hmcStepBindGroupLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // q_in
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // q_out
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4 } }, // accepted_out (atomic u32)
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // rand_state
    ] });
    const hmcStepPipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [commonBindGroupLayout, hmcStepBindGroupLayout] }), compute: { module: shaderModule, entryPoint: 'hmc_step_kernel' } });

    // Bind Groups - we need two for ping-ponging q_buffer_A and q_buffer_B
    const commonBindGroup = device.createBindGroup({ layout: commonBindGroupLayout, entries: [ { binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: dataX1Buffer } }, { binding: 2, resource: { buffer: dataX2Buffer } }, { binding: 3, resource: { buffer: dataYBuffer } } ] });
    const hmcStepBindGroupA = device.createBindGroup({ layout: hmcStepBindGroupLayout, entries: [
        { binding: 0, resource: { buffer: q_buffer_A } },        // Read from A
        { binding: 1, resource: { buffer: q_buffer_B } },        // Write to B
        { binding: 2, resource: { buffer: accepted_buffer } },
        { binding: 3, resource: { buffer: rand_state_buffer } },
    ] });
    const hmcStepBindGroupB = device.createBindGroup({ layout: hmcStepBindGroupLayout, entries: [
        { binding: 0, resource: { buffer: q_buffer_B } },        // Read from B
        { binding: 1, resource: { buffer: q_buffer_A } },        // Write to A
        { binding: 2, resource: { buffer: accepted_buffer } },
        { binding: 3, resource: { buffer: rand_state_buffer } },
    ] });

    // --- 4. HMC Main Loop ---
    console.log(`HMC sampling started for ${num_chains} chains...`);
    const startTime = performance.now();

    // A smaller chunk size is needed to prevent the OS from terminating the GPU task (TDR).
    // 1000 was still too large; 100 is a much safer value.
    const chunk_size = 1000; // Process in chunks to avoid OS timeout (TDR)

    // Zero out the acceptance buffer before starting the loop
    {
        const initialEncoder = device.createCommandEncoder();
        initialEncoder.clearBuffer(accepted_buffer);
        device.queue.submit([initialEncoder.finish()]);
    }
    
    // The main loop. A standard for-loop is used for maximum performance.
    for (let i = 0; i < iterations; i++) {
        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(hmcStepPipeline);
        pass.setBindGroup(0, commonBindGroup);

        const q_out_buffer = (i % 2 === 0) ? q_buffer_B : q_buffer_A;
        pass.setBindGroup(1, (i % 2 === 0) ? hmcStepBindGroupA : hmcStepBindGroupB);
        
        pass.dispatchWorkgroups(Math.ceil(num_chains / workgroup_size));
        pass.end();

        commandEncoder.copyBufferToBuffer(
            q_out_buffer, 0,
            samples_buffer, i * num_chains * 4 * 4,
            num_chains * 4 * 4
        );
        device.queue.submit([commandEncoder.finish()]);

        // Periodically wait for the GPU to finish a chunk to prevent TDR errors.
        if ((i + 1) % chunk_size === 0) {
            await device.queue.onSubmittedWorkDone();
            console.log(`Dispatched and completed ${i + 1}/${iterations} iterations...`);
        }
    }
    
    // --- 5. Finalization: Read results and clean up ---
    await device.queue.onSubmittedWorkDone(); // Ensure all work is done before reading

    const finalEncoder = device.createCommandEncoder();
    const resultsReadBuffer = device.createBuffer({ size: samples_buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const acceptedReadBuffer = device.createBuffer({ size: accepted_buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    
    finalEncoder.copyBufferToBuffer(samples_buffer, 0, resultsReadBuffer, 0, samples_buffer.size);
    finalEncoder.copyBufferToBuffer(accepted_buffer, 0, acceptedReadBuffer, 0, accepted_buffer.size);
    device.queue.submit([finalEncoder.finish()]);

    await Promise.all([
        resultsReadBuffer.mapAsync(GPUMapMode.READ),
        acceptedReadBuffer.mapAsync(GPUMapMode.READ)
    ]);

    const samples = new Float32Array(resultsReadBuffer.getMappedRange()).slice();
    const total_accepted = new Uint32Array(acceptedReadBuffer.getMappedRange())[0];

    resultsReadBuffer.unmap();
    acceptedReadBuffer.unmap();

    const endTime = performance.now();
    console.log(`HMC finished in ${(endTime - startTime) / 1000} seconds.`);
    console.log(`Overall acceptance rate: ${(total_accepted / (iterations * num_chains)).toFixed(2)}`);

    const total_valid_samples = (iterations - burn_in) * num_chains;
    const beta1_samples = new Float32Array(total_valid_samples);
    let sample_idx = 0;
    for (let i = burn_in; i < iterations; i++) {
        for (let c = 0; c < num_chains; c++) {
            const sample_start_index = (i * num_chains + c) * 4;
            beta1_samples[sample_idx++] = samples[sample_start_index + 1];
        }
    }

    const mean_beta1 = beta1_samples.reduce((a, b) => a + b, 0) / total_valid_samples;
    beta1_samples.sort();
    const p2_5 = beta1_samples[Math.floor(total_valid_samples * 0.025)];
    const p97_5 = beta1_samples[Math.floor(total_valid_samples * 0.975)];

    console.log("\n--- Estimation Results ---");
    console.log(`True ad effect (beta1): ${true_beta1}`);
    console.log(`HMC estimated ad effect mean: ${mean_beta1.toFixed(2)}`);
    console.log(`HMC estimated 95% credible interval: ${p2_5.toFixed(2)} to ${p97_5.toFixed(2)}`);

    // Cleanup
    paramsBuffer.destroy();
    dataX1Buffer.destroy();
    dataX2Buffer.destroy();
    dataYBuffer.destroy();
    q_buffer_A.destroy();
    q_buffer_B.destroy();
    rand_state_buffer.destroy();
    accepted_buffer.destroy();
    samples_buffer.destroy();
    resultsReadBuffer.destroy();
    acceptedReadBuffer.destroy();
}