import { fetchText, msg } from "@i18n";
import { stopAnimation } from "./instance";

let betaValue: number = 2.0;
let kappaValue: number = 0.5;

/**
 * フルスケール SU(2) Gauge-Higgs モデル
 * HTML側のUIを使用してパラメータを制御します。
 */
export async function runGaugeHiggs(device: GPUDevice, theory: 'U1' | 'SU2', mode :"C" | "E"): Promise<() => void> {
    msg("Starting Full Scale SU(2) Gauge-Higgs (Electroweak) simulation.");
    stopAnimation();

    const L = 64; 
    const L_squared = L * L;
    const WORKGROUP_SIZE = 8; 

    // --- 1. UI要素の取得と表示 ---
    const ghPanel = document.getElementById('gauge-higgs-panel') as HTMLDivElement;
    const betaSlider = document.getElementById('beta-inp') as HTMLInputElement;
    const kappaSlider = document.getElementById('kappa-inp') as HTMLInputElement;
    const betaValueSpan = document.getElementById('gh-beta-value') as HTMLSpanElement;
    const kappaValueSpan = document.getElementById('gh-kappa-value') as HTMLSpanElement;

    if (!ghPanel || !betaSlider || !kappaSlider) {
        msg("Gauge-Higgs UI elements not found in HTML.");
        return () => {};
    }

    // パネルを表示
    ghPanel.style.display = 'block';

    // HTMLの初期値を読み取る
    betaValue = parseFloat(betaSlider.value);
    kappaValue = parseFloat(kappaSlider.value);

    if (!device) {
        msg("GPUDevice not found. Simulation cannot start.");
        return () => {};
    }

    let computeName : string;
    let renderName  : string;
    let dataSize : number;
    let renderShaderCode : string;

    if(theory == "U1"){
        renderName = (mode == "E" ? "u1_higgs_render-E" : "u1_higgs_render-C");
        [computeName, dataSize] = [ "lgt_u1_higgs", 4];

        renderShaderCode = "const L: f32 = 64.0;\n" + await fetchText(`./wgsl/lgt/${renderName}.wgsl`);
    }
    else{
        [computeName, renderName, dataSize] = [ "lgt_gauge_higgs", "gauge_higgs_render", 16];
        renderShaderCode = await fetchText(`./wgsl/lgt/${renderName}.wgsl`);
    }

    // --- 2. シェーダーの読み込み ---
    const computeShaderCode = await fetchText(`./wgsl/lgt/${computeName}.wgsl`);
    const computeModule = device.createShaderModule({ code: computeShaderCode });
    const renderModule = device.createShaderModule({ code: renderShaderCode });

    // --- 3. バッファの作成 ---
    // ゲージ場 (リンク): 2方向 * L*Lサイト * 16バイト(vec4)
    const linksBuffer = device.createBuffer({
        size: 2 * L_squared * dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // ヒッグス場 (サイト): 1方向 * L*Lサイト * 16バイト(vec4)
    const higgsBuffer = device.createBuffer({
        size: L_squared * dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // 乱数シード
    const rngState = new Uint32Array(L_squared);
    for (let i = 0; i < L_squared; i++) rngState[i] = Math.random() * 0xFFFFFFFF;
    const rngBuffer = device.createBuffer({
        size: rngState.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(rngBuffer.getMappedRange()).set(rngState);
    rngBuffer.unmap();

    // 観測結果バッファ (描画用)
    const vizResultBuffer = device.createBuffer({
        size: L_squared * 4, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // ユニフォームバッファ (Dynamic Offset対応: 4サブセット * 256バイト = 1024バイト)
    const paramsBuffer = device.createBuffer({
        size: 256 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // パラメータ書き込み関数
    function writeParams(beta: number, kappa: number) {
        betaValue = beta;
        kappaValue = kappa;

        const simParams = new Float32Array(256 * 4 / 4);
        const uintView = new Uint32Array(simParams.buffer);

        for (let i = 0; i < 4; i++) {
            const offset = (256 * i) / 4;
            simParams[offset + 0] = betaValue;
            simParams[offset + 1] = kappaValue;
            uintView[offset + 2] = i;           
            simParams[offset + 3] = 0.0;        
        }

        device.queue.writeBuffer(paramsBuffer, 0, simParams);
        betaValueSpan.innerText = betaValue.toFixed(1);
        kappaValueSpan.innerText = kappaValue.toFixed(2);
    }

    // 初期値書き込み
    writeParams(betaValue, kappaValue);

    // イベントリスナーの定義
    const onBetaChange = () => writeParams(parseFloat(betaSlider.value), kappaValue);
    const onKappaChange = () => writeParams(betaValue, parseFloat(kappaSlider.value));
    
    // リスナー登録
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
        ]
    });

    const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] });

    const initPipeline = await device.createComputePipelineAsync({
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'init_hot' }
    });

    const updateHiggsPipeline = await device.createComputePipelineAsync({
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'update_higgs' }
    });

    const updateGaugePipeline = await device.createComputePipelineAsync({
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'update_gauge' }
    });

    let measureEntryPoint : string;
    if(theory == "SU2"){
        measureEntryPoint = "measure_observables";
    }
    else{
        if(mode == "E"){
            measureEntryPoint = "measure_observables_E";
        }
        else{
            measureEntryPoint = "measure_observables_C";
        }
    }
    const measurePipeline = await device.createComputePipelineAsync({
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: measureEntryPoint }
    });

    const computeBindGroup = device.createBindGroup({
        layout: computeBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer, size: 16 } },
            { binding: 1, resource: { buffer: linksBuffer } },
            { binding: 2, resource: { buffer: higgsBuffer } },
            { binding: 3, resource: { buffer: rngBuffer } },
            { binding: 4, resource: { buffer: vizResultBuffer } },
        ]
    });

    // --- 5. レンダーパイプラインの作成 ---
    const canvas = document.querySelector("#world") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu")!;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device, format: presentationFormat, alphaMode: "premultiplied" });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
        ]
    });

    // 描画モード用パラメータ (0=Plaquetteモードのカラーマップを流用)
    const renderParamsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(renderParamsBuffer, 0, new Uint32Array([0]));

    const renderBuffer = (theory === "U1" && mode == "C" ? vizResultBuffer : higgsBuffer);
    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: renderBuffer } }
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
    msg("Gauge-Higgs System initialized.");

    // --- 7. アニメーションループ ---
    let animationId: number | null = null;
    const sweepsPerFrame = 1;

    function frame() {
        const commandEncoder = device.createCommandEncoder();

        // 物理更新ステップ
        for (let s = 0; s < sweepsPerFrame; s++) {
            // ① ヒッグス場の更新 (サイト上なので チェッカーボード 0, 1 のみ)
            for (let j = 0; j < 2; j++) {
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(updateHiggsPipeline);
                pass.setBindGroup(0, computeBindGroup, [256 * j]); 
                pass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
                pass.end();
            }

            // ② ゲージ場の更新 (リンク上なので 4色チェッカーボード 0, 1, 2, 3 すべて)
            for (let j = 0; j < 4; j++) {
                const pass = commandEncoder.beginComputePass();
                pass.setPipeline(updateGaugePipeline);
                pass.setBindGroup(0, computeBindGroup, [256 * j]); 
                pass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
                pass.end();
            }
        }

        // 物理量の測定 (描画用バッファへ書き込み)
        const measurePass = commandEncoder.beginComputePass();
        measurePass.setPipeline(measurePipeline);
        measurePass.setBindGroup(0, computeBindGroup, [0]);
        measurePass.dispatchWorkgroups(L / WORKGROUP_SIZE, L / WORKGROUP_SIZE, 1);
        measurePass.end();

        // 描画ステップ
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

    // --- 8. クリーンアップ関数 ---
    return () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        // イベントリスナーの解除
        betaSlider.removeEventListener('input', onBetaChange);
        kappaSlider.removeEventListener('input', onKappaChange);
        
        // パネルを非表示に戻す
        ghPanel.style.display = 'none';
    };
}