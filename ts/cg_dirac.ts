import { fetchText } from "@i18n";

export async function runDiracCGSolver(device: GPUDevice) {
    const L = 128;
    const N = L * L;
    const spinorSize = N * 16; // vec4<f32> = 16 bytes

    // --- 1. バッファの作成 ---
    const xBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const bBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const rBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const qBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE });
    const tmpBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE });
    
    const scalarsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    
    // 背景ゲージ場 (テスト用: 全て0 = 真空)
    const linksBuffer = device.createBuffer({ size: N * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(linksBuffer, 0, new Float32Array(N * 2));

    // 質量パラメータ (m = 0.1)
    const paramsBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([0.1]));

    // --- 2. 初期条件のセット (点電荷) ---
    const bData = new Float32Array(N * 4);
    // 中央のサイトの、上スピンの実部だけに 1.0 を置く
    bData[(Math.floor(L/2) * L + Math.floor(L/2)) * 4 + 0] = 1.0; 
    
    device.queue.writeBuffer(bBuffer, 0, bData);
    device.queue.writeBuffer(xBuffer, 0, new Float32Array(N * 4));
    device.queue.writeBuffer(rBuffer, 0, bData);
    device.queue.writeBuffer(pBuffer, 0, bData);

    let initialRho = 0;
    for (let i = 0; i < N * 4; i++) initialRho += bData[i] * bData[i];
    device.queue.writeBuffer(scalarsBuffer, 0, new Float32Array([initialRho, 0, 0, 0, 0]));

    // --- 3. パイプラインの構築 ---
    const code = await fetchText('./wgsl/cg_dirac.wgsl');
    const module = device.createShaderModule({ code });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: xBuffer } },
            { binding: 1, resource: { buffer: bBuffer } },
            { binding: 2, resource: { buffer: pBuffer } },
            { binding: 3, resource: { buffer: rBuffer } },
            { binding: 4, resource: { buffer: qBuffer } },
            { binding: 5, resource: { buffer: scalarsBuffer } },
            { binding: 6, resource: { buffer: linksBuffer } },
            { binding: 7, resource: { buffer: paramsBuffer } },
            { binding: 8, resource: { buffer: tmpBuffer } },
        ]
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const createPipeline = (entryPoint: string) => device.createComputePipelineAsync({ layout, compute: { module, entryPoint } });

    const applyDPipeline = await createPipeline('apply_D');
    const applyDDagPipeline = await createPipeline('apply_D_dag');
    const calcPQPipeline = await createPipeline('calc_p_q');
    const updateXRPipeline = await createPipeline('update_x_r');
    const calcNewRhoPipeline = await createPipeline('calc_new_rho');
    const updatePPipeline = await createPipeline('update_p');

    // --- 4. 実行ループ ---
    console.log("Starting Dirac CG Solver...");
    const startTime = performance.now();
    const maxIters = 400; // クォークは収束に少し時間がかかります
    const workgroups = Math.ceil(N / 64);

    const commandEncoder = device.createCommandEncoder();

    for (let i = 0; i < maxIters; i++) {
        const pass = commandEncoder.beginComputePass();
        pass.setBindGroup(0, bindGroup);

        pass.setPipeline(applyDPipeline);     pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(applyDDagPipeline);  pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(calcPQPipeline);     pass.dispatchWorkgroups(1);
        pass.setPipeline(updateXRPipeline);   pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(calcNewRhoPipeline); pass.dispatchWorkgroups(1);
        pass.setPipeline(updatePPipeline);    pass.dispatchWorkgroups(workgroups);
        
        pass.end();
    }

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const endTime = performance.now();
    console.log(`Dirac CG Solver finished in ${endTime - startTime} ms`);

    const readbackBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(scalarsBuffer, 0, readbackBuffer, 0, 32);
    device.queue.submit([copyEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const finalScalars = new Float32Array(readbackBuffer.getMappedRange());
    console.log(`Final Residual Norm Squared (rho): ${finalScalars[0]}`); 
    readbackBuffer.unmap();
}