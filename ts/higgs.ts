import { fetchText, msg } from "@i18n"; // 環境に合わせてインポート元を調整してください
import { stopAnimation } from "./instance";

let kappaValue: number = 0.3;
let lambdaValue: number = 1.0;

/**
 * 2D ヒッグス場 (O(2) 複素スカラー場) シミュレーション
 */
export async function runHiggs(device: GPUDevice): Promise<() => void> {
    msg("Starting 2D Higgs Field (Spontaneous Symmetry Breaking) simulation.");
    stopAnimation();

    const L = 128; 
    const L_squared = L * L;
    const WORKGROUP_SIZE = 8; 

    // --- 1. UI要素の取得と表示 ---
    const higgsPanel = document.getElementById('higgs-panel') as HTMLDivElement;
    const kappaSlider = document.getElementById('kappa-slider') as HTMLInputElement;
    const lambdaSlider = document.getElementById('lambda-slider') as HTMLInputElement;
    const kappaValueSpan = document.getElementById('kappa-value') as HTMLSpanElement;
    const lambdaValueSpan = document.getElementById('lambda-value') as HTMLSpanElement;

    if (!higgsPanel || !kappaSlider || !lambdaSlider) {
        msg("Higgs UI elements not found in HTML.");
        return () => {};
    }

    // パネルを表示
    higgsPanel.style.display = 'block';

    // HTMLの初期値を読み取る
    kappaValue = parseFloat(kappaSlider.value);
    lambdaValue = parseFloat(lambdaSlider.value);

    if (!device) {
        msg("GPUDevice not found. Simulation cannot start.");
        return () => {};
    }

    // --- 2. シェーダーの読み込み ---
    const computeShaderCode = await fetchText(`./wgsl/higgs_field.wgsl`);
    const renderShaderCode = await fetchText(`./wgsl/higgs_render.wgsl`);
    const computeModule = device.createShaderModule({ code: computeShaderCode });
    const renderModule = device.createShaderModule({ code: renderShaderCode });

    // --- 3. バッファの作成 ---
    const fieldBuffer = device.createBuffer({
        size: L_squared * 8, // 2 * f32 per site
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const rngState = new Uint32Array(L_squared);
    for (let i = 0; i < L_squared; i++) rngState[i] = Math.random() * 0xFFFFFFFF;
    const rngBuffer = device.createBuffer({
        size: rngState.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(rngBuffer.getMappedRange()).set(rngState);
    rngBuffer.unmap();

    const paramsBuffer = device.createBuffer({
        size: 256 * 2,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // パラメータ書き込み関数
    function writeParams(kappa: number, lambda: number) {
        kappaValue = kappa;
        lambdaValue = lambda;

        const simParams = new Float32Array(256 * 2 / 4);
        const uintView = new Uint32Array(simParams.buffer);

        for (let i = 0; i < 2; i++) {
            const offset = (256 * i) / 4;
            simParams[offset + 0] = kappaValue;
            simParams[offset + 1] = lambdaValue;
            uintView[offset + 2] = i;           
            simParams[offset + 3] = 0.0;        
        }

        device.queue.writeBuffer(paramsBuffer, 0, simParams);
        kappaValueSpan.innerText = kappaValue.toFixed(2);
        lambdaValueSpan.innerText = lambdaValue.toFixed(1);
    }

    // 初期値の書き込み
    writeParams(kappaValue, lambdaValue);

    // イベントリスナーの定義
    const onKappaChange = () => writeParams(parseFloat(kappaSlider.value), lambdaValue);
    const onLambdaChange = () => writeParams(kappaValue, parseFloat(lambdaSlider.value));
    
    // リスナー登録
    kappaSlider.addEventListener('input', onKappaChange);
    lambdaSlider.addEventListener('input', onLambdaChange);

    // --- 4. パイプラインの作成 ---
    const computeBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
        ]
    });

    const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] });

    const initPipeline = await device.createComputePipelineAsync({
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'init_hot' }
    });

    const updatePipeline = await device.createComputePipelineAsync({
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'metropolis_update' }
    });

    const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer, size: 16 } },
            { binding: 1, resource: { buffer: fieldBuffer } },
            { binding: 2, resource: { buffer: rngBuffer } },
        ]
    });

    // --- 5. レンダーパイプラインの作成 ---
    const canvas = document.querySelector("#world") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu")!;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device, format: presentationFormat, alphaMode: "premultiplied" });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
        ]
    });

    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: fieldBuffer } }
        ]
    });

    const renderPipeline = await device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
        vertex: { module: renderModule, entryPoint: 'vs_main' },
        fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: presentationFormat }] },
        primitive: { topology: 'triangle-list' }
    });

    // --- 6. 初期化の実行 ---
    let initEncoder = device.createCommandEncoder();
    let initPass = initEncoder.beginComputePass();
    initPass.setPipeline(initPipeline);
    initPass.setBindGroup(0, computeBindGroup, [0]);
    initPass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
    initPass.end();
    device.queue.submit([initEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    msg("Higgs Field initialized.");

    // --- 7. アニメーションループ ---
    let animationId: number | null = null;
    const sweepsPerFrame = 15;

    function frame() {
        const commandEncoder = device.createCommandEncoder();

        for (let s = 0; s < sweepsPerFrame; s++) {
            for (let j = 0; j < 2; j++) {
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(updatePipeline);
                pass.setBindGroup(0, computeBindGroup, [256 * j]); 
                pass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
                pass.end();
            }
        }

        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear', storeOp: 'store'
            }]
        };

        const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.setBindGroup(0, renderBindGroup);
        renderPassEncoder.draw(6, L_squared, 0, 0); 
        renderPassEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        animationId = requestAnimationFrame(frame);
    }

    frame();

    // --- 8. クリーンアップ関数 (他のシミュに切り替わる時呼ばれる) ---
    return () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        // イベントリスナーの解除
        kappaSlider.removeEventListener('input', onKappaChange);
        lambdaSlider.removeEventListener('input', onLambdaChange);
        
        // パネルを非表示に戻す
        higgsPanel.style.display = 'none';
    };
}