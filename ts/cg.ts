import { fetchText } from "@i18n";

export async function runCGSolver(device: GPUDevice) {
    const L = 128;
    const N = L * L;
    const arraySize = N * 4; // f32 = 4 bytes

    // --- 1. バッファの初期化 ---
    const xBuffer = device.createBuffer({ size: arraySize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const bBuffer = device.createBuffer({ size: arraySize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pBuffer = device.createBuffer({ size: arraySize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const rBuffer = device.createBuffer({ size: arraySize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const qBuffer = device.createBuffer({ size: arraySize, usage: GPUBufferUsage.STORAGE });
    
    // スカラー用: [rho, p_q, alpha, beta, new_rho] (5 * 4 = 20 bytes, パディングして32 bytes)
    const scalarsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    // テストデータの作成: 中央に1.0 (点電荷)、他は0.0
    const bData = new Float32Array(N);
    bData[Math.floor(L/2) * L + Math.floor(L/2)] = 1.0; 
    device.queue.writeBuffer(bBuffer, 0, bData);

    // CG法の初期条件: x0 = 0, r0 = b - Ax0 = b, p0 = r0
    const initialZeros = new Float32Array(N);
    device.queue.writeBuffer(xBuffer, 0, initialZeros);
    device.queue.writeBuffer(rBuffer, 0, bData); // r = b
    device.queue.writeBuffer(pBuffer, 0, bData); // p = b

    // 初期 rho = r0 * r0 の計算 (CPU側でやってGPUに渡す)
    let initialRho = 0;
    for (let i = 0; i < N; i++) initialRho += bData[i] * bData[i];
    const initialScalars = new Float32Array([initialRho, 0, 0, 0, 0]);
    device.queue.writeBuffer(scalarsBuffer, 0, initialScalars);

    // --- 2. シェーダーとパイプラインの準備 ---
    const code = await fetchText('./wgsl/lgt/cg_solver.wgsl');
    const module = device.createShaderModule({ code });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
        ]
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const createPipeline = (entryPoint: string) => device.createComputePipelineAsync({ layout, compute: { module, entryPoint } });

    const applyAPipeline = await createPipeline('apply_A');
    const calcPQPipeline = await createPipeline('calc_p_q');
    const updateXRPipeline = await createPipeline('update_x_r');
    const calcNewRhoPipeline = await createPipeline('calc_new_rho');
    const updatePPipeline = await createPipeline('update_p');

    // --- 3. メインループのエンコードと送信 ---
    console.log("Starting GPU CG Solver...");
    const startTime = performance.now();
    const maxIters = 200; // ラプラシアンなら数百回で収束
    const workgroups = Math.ceil(N / 64);

    const commandEncoder = device.createCommandEncoder();

    // CPUはGPUを待たずに、maxIters回分のコマンドを構築！
    for (let i = 0; i < maxIters; i++) {
        const pass = commandEncoder.beginComputePass();
        pass.setBindGroup(0, bindGroup);

        pass.setPipeline(applyAPipeline);     pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(calcPQPipeline);     pass.dispatchWorkgroups(1);
        pass.setPipeline(updateXRPipeline);   pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(calcNewRhoPipeline); pass.dispatchWorkgroups(1);
        pass.setPipeline(updatePPipeline);    pass.dispatchWorkgroups(workgroups);
        
        pass.end();
    }

    // コマンドを一気にGPUへ送信
    device.queue.submit([commandEncoder.finish()]);

    // --- 4. 結果の待機と確認 ---
    await device.queue.onSubmittedWorkDone();
    const endTime = performance.now();
    console.log(`CG Solver finished in ${endTime - startTime} ms`);

    // 結果（rho = 残差の2乗ノルム）を読み出して収束を確認
    const readbackBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(scalarsBuffer, 0, readbackBuffer, 0, 32);
    device.queue.submit([copyEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const finalScalars = new Float32Array(readbackBuffer.getMappedRange());
    console.log(`Final Residual Norm Squared (rho): ${finalScalars[0]}`); // ゼロに近ければ成功！
    readbackBuffer.unmap();
}