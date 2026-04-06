import { fetchText, msg } from "@i18n";
import { stopAnimation } from "./instance";

let betaValue: number = 4.0;
let kappaValue: number = 1.0;

/**
 * U(1) Gauge-Higgs モデル (HMC版)
 */
export async function runHMCGaugeHiggs(device: GPUDevice, mode: "C" | "E"): Promise<() => void> {
    msg("Starting U(1) Gauge-Higgs HMC simulation...");
    stopAnimation();

    const L = 64;
    const L_squared = L * L;
    const WORKGROUP_SIZE = 8; // WGSLのワークグループサイズに合わせて調整(8x8=64)

    // --- 1. UI要素の取得 ---
    const ghPanel = document.getElementById('gauge-higgs-panel') as HTMLDivElement;
    const betaSlider = document.getElementById('beta-inp') as HTMLInputElement;
    const kappaSlider = document.getElementById('kappa-inp') as HTMLInputElement;
    const betaValueSpan = document.getElementById('gh-beta-value') as HTMLSpanElement;
    const kappaValueSpan = document.getElementById('gh-kappa-value') as HTMLSpanElement;

    if (!ghPanel || !betaSlider || !kappaSlider) {
        msg("UI elements not found.");
        return () => {};
    }

    ghPanel.style.display = 'block';
    betaValue = parseFloat(betaSlider.value);
    kappaValue = parseFloat(kappaSlider.value);

    // --- 2. シェーダーの読み込み ---
    const computeShaderCode = await fetchText('./wgsl/hmc_u1_higgs.wgsl');
    const renderName = mode === "E" ? "u1_higgs_render-E" : "u1_higgs_render-C";
    const renderShaderCode = await fetchText(`./wgsl/${renderName}.wgsl`);

    const computeModule = device.createShaderModule({ code: computeShaderCode });
    const renderModule = device.createShaderModule({ code: renderShaderCode });

    // --- 3. バッファの作成 ---
    // 物理場
    const linksBuffer = device.createBuffer({ size: 2 * L_squared * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const higgsBuffer = device.createBuffer({ size: L_squared * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    
    // 運動量 (HMC専用)
    const pLinksBuffer = device.createBuffer({ size: 2 * L_squared * 4, usage: GPUBufferUsage.STORAGE });
    const pHiggsBuffer = device.createBuffer({ size: L_squared * 4, usage: GPUBufferUsage.STORAGE });
    
    // ロールバック用バックアップ (HMC専用)
    const oldLinksBuffer = device.createBuffer({ size: 2 * L_squared * 4, usage: GPUBufferUsage.STORAGE });
    const oldHiggsBuffer = device.createBuffer({ size: L_squared * 4, usage: GPUBufferUsage.STORAGE });

    // 観測結果とエネルギー、スカラー、乱数
    const vizResultBuffer = device.createBuffer({ size: L_squared * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const energyBuffer = device.createBuffer({ size: L_squared * 4, usage: GPUBufferUsage.STORAGE });
    const scalarsBuffer = device.createBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const rngState = new Uint32Array(L_squared);
    for (let i = 0; i < L_squared; i++) rngState[i] = Math.random() * 0xFFFFFFFF;
    const rngBuffer = device.createBuffer({
        size: rngState.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(rngBuffer.getMappedRange()).set(rngState);
    rngBuffer.unmap();

    // HMCパラメータ用 Uniform Buffer (Dynamic Offset対応)
    // 4つのスライスを用意 (0: 初期化用, 256: eps/2用, 512: eps等倍用, 768: H_new用)
    const paramsBuffer = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    let md_eps = 0.05;

    function updateParams(beta: number, kappa: number) {
        betaValue = beta;
        kappaValue = kappa;

        const simParams = new Float32Array(256 * 4 / 4);
        const uintView = new Uint32Array(simParams.buffer);

        // Slice 0 (offset 0): H_old 用 (eps=0, is_new_H=0)
        simParams[0] = betaValue; simParams[1] = kappaValue; simParams[2] = 0.0; uintView[3] = 0;
        
        // Slice 1 (offset 256): リープフロッグ半歩用 (eps = md_eps / 2)
        const off1 = 256 / 4;
        simParams[off1+0] = betaValue; simParams[off1+1] = kappaValue; simParams[off1+2] = md_eps / 2.0; uintView[off1+3] = 0;
        
        // Slice 2 (offset 512): リープフロッグ1歩用 (eps = md_eps)
        const off2 = 512 / 4;
        simParams[off2+0] = betaValue; simParams[off2+1] = kappaValue; simParams[off2+2] = md_eps; uintView[off2+3] = 0;
        
        // Slice 3 (offset 768): H_new 用 (eps=0, is_new_H=1)
        const off3 = 768 / 4;
        simParams[off3+0] = betaValue; simParams[off3+1] = kappaValue; simParams[off3+2] = 0.0; uintView[off3+3] = 1;

        device.queue.writeBuffer(paramsBuffer, 0, simParams);
        betaValueSpan.innerText = betaValue.toFixed(1);
        kappaValueSpan.innerText = kappaValue.toFixed(2);
    }

    updateParams(betaValue, kappaValue);
    const onBetaChange = () => updateParams(parseFloat(betaSlider.value), kappaValue);
    const onKappaChange = () => updateParams(betaValue, parseFloat(kappaSlider.value));
    betaSlider.addEventListener('input', onBetaChange);
    kappaSlider.addEventListener('input', onKappaChange);

    // --- 4. コンピュートパイプラインの作成 ---
    const computeBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // viz_results
        ]
    });

    const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] });
    const createCompute = async (entry: string) => device.createComputePipelineAsync({ layout: computePipelineLayout, compute: { module: computeModule, entryPoint: entry } });

    const initTrajPipeline = await createCompute('init_trajectory');
    const calcHPipeline = await createCompute('calc_local_H');
    const reduceHPipeline = await createCompute('reduce_H');
    const updateQPipeline = await createCompute('update_q');
    const updatePPipeline = await createCompute('update_P');
    const acceptRejectPipeline = await createCompute('accept_reject');
    const initHotPipeline = await createCompute('init_hot');

    const measureEntryPoint = mode === "E" ? "measure_observables_E" : "measure_observables_C";
    const measurePipeline = await createCompute(measureEntryPoint);

    const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer, size: 16 } },
            { binding: 1, resource: { buffer: linksBuffer } },
            { binding: 2, resource: { buffer: higgsBuffer } },
            { binding: 3, resource: { buffer: pLinksBuffer } },
            { binding: 4, resource: { buffer: pHiggsBuffer } },
            { binding: 5, resource: { buffer: oldLinksBuffer } },
            { binding: 6, resource: { buffer: oldHiggsBuffer } },
            { binding: 7, resource: { buffer: rngBuffer } },
            { binding: 8, resource: { buffer: scalarsBuffer } },
            { binding: 9, resource: { buffer: energyBuffer } },
            { binding: 10, resource: { buffer: vizResultBuffer } },
        ]
    });

    // --- 5. レンダーパイプライン ---
    const canvas = document.querySelector("#world") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu")!;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: presentationFormat, alphaMode: "premultiplied" });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }]
    });

    const renderBuffer = mode === "C" ? vizResultBuffer : higgsBuffer;
    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: renderBuffer } }]
    });

    const renderPipeline = await device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
        vertex: { module: renderModule, entryPoint: 'vs_main' },
        fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: presentationFormat }] },
        primitive: { topology: 'triangle-list' }
    });

    // --- 5.5 初期化の実行 (Hot Start) ---
    let initEncoder = device.createCommandEncoder();
    let initPass = initEncoder.beginComputePass();
    initPass.setPipeline(initHotPipeline);
    // dynamicOffsetは0番目(offset 0)を使用
    initPass.setBindGroup(0, computeBindGroup, [0]);
    // ワークグループ数は他と同じ (L*L / 64)
    initPass.dispatchWorkgroups(Math.ceil(L_squared / 64));
    initPass.end();
    device.queue.submit([initEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    msg("HMC System initialized with Hot Start.");


    // --- 6. HMC アニメーションループ ---
    let animationId: number | null = null;
    const workgroups = Math.ceil(L_squared / 64);
    const md_steps = 15;

    function frame() {
        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();

        // 1. 軌道の初期化
        pass.setPipeline(initTrajPipeline);
        pass.setBindGroup(0, computeBindGroup, [0]); // offset 0
        pass.dispatchWorkgroups(workgroups);

        // 2. 古いハミルトニアンの計算
        pass.setPipeline(calcHPipeline);
        pass.setBindGroup(0, computeBindGroup, [0]); // offset 0
        pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(reduceHPipeline);
        pass.dispatchWorkgroups(1);

        // 3. リープフロッグ積分開始 (Pを半歩)
        pass.setPipeline(updatePPipeline);
        pass.setBindGroup(0, computeBindGroup, [256]); // offset 256 (eps/2)
        pass.dispatchWorkgroups(workgroups);

        // メインループ
        for(let i = 0; i < md_steps - 1; i++) {
            pass.setBindGroup(0, computeBindGroup, [512]); // offset 512 (eps)
            pass.setPipeline(updateQPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(updatePPipeline); pass.dispatchWorkgroups(workgroups);
        }

        // 最後のステップ (qを1歩、Pを半歩)
        pass.setBindGroup(0, computeBindGroup, [512]); 
        pass.setPipeline(updateQPipeline); pass.dispatchWorkgroups(workgroups);
        
        pass.setBindGroup(0, computeBindGroup, [256]); 
        pass.setPipeline(updatePPipeline); pass.dispatchWorkgroups(workgroups);

        // 4. 新しいハミルトニアンの計算
        pass.setPipeline(calcHPipeline);
        pass.setBindGroup(0, computeBindGroup, [768]); // offset 768 (is_new_H=1)
        pass.dispatchWorkgroups(workgroups);
        pass.setPipeline(reduceHPipeline);
        pass.dispatchWorkgroups(1);

        // 5. 受容判定
        pass.setPipeline(acceptRejectPipeline);
        pass.setBindGroup(0, computeBindGroup, [0]); 
        pass.dispatchWorkgroups(workgroups);

        // 6. 観測量の測定
        // スレッド数は 8x8 = 64用 に組んだものを使用
        pass.setPipeline(measurePipeline);
        pass.setBindGroup(0, computeBindGroup, [0]); 
        pass.dispatchWorkgroups(L / 8, L / 8, 1); 

        pass.end();

        // 7. 描画
        const renderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
        });
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.setBindGroup(0, renderBindGroup);
        renderPassEncoder.draw(6, L_squared, 0, 0); 
        renderPassEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        animationId = requestAnimationFrame(frame);
    }

    frame();

    return () => {
        if (animationId) cancelAnimationFrame(animationId);
        betaSlider.removeEventListener('input', onBetaChange);
        kappaSlider.removeEventListener('input', onKappaChange);
        ghPanel.style.display = 'none';
    };
}