import { $, $inp, fetchText, msg } from "@i18n";
import { stopAnimation } from "../instance";

let betaValue: number = 4.0;
let kappaValue: number = 1.0;

// ↓ クォークを重くして方程式を解きやすくする
let massValue: number = 1.0; // クォークの軽さ (小さいほどCGが重くなる)

export async function runDynamicalFermions(device: GPUDevice, mode: "C" | "E"): Promise<() => void> {
    msg("Starting Dynamical Fermions (HMC + CG Solver) simulation...");
    stopAnimation();

    const L = 32;
    const N = L * L;
    const workgroups = Math.ceil(N / 64);

    $("gauge-higgs-panel").style.display = "block";

    $inp("beta-inp").value  = `${betaValue}`;
    $inp("kappa-inp").value = `${kappaValue}`;
    $inp("mass-inp").value  = `${massValue}`;

    for(const inp of [$inp("beta-inp"), $inp("kappa-inp"), $inp("mass-inp")]){
        inp.addEventListener("change", (ev:Event)=>{
            betaValue  = parseFloat($inp("beta-inp").value);
            kappaValue = parseFloat($inp("kappa-inp").value);
            massValue  = parseFloat($inp("mass-inp").value);

            writeParams();
            msg(`beta :${betaValue} kappa :${kappaValue} mass :${massValue}`);
        });
    }

    // ========================================================================
    // 1. バッファの作成 (Multi-BindGroup Architecture)
    // ========================================================================
    const floatSize = N * 4;
    const spinorSize = N * 16;

    // --- BindGroup 0: HMC Core ---
    const linksBuffer = device.createBuffer({ size: floatSize * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const higgsBuffer = device.createBuffer({ size: floatSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pLinksBuffer = device.createBuffer({ size: floatSize * 2, usage: GPUBufferUsage.STORAGE });
    const pHiggsBuffer = device.createBuffer({ size: floatSize, usage: GPUBufferUsage.STORAGE });
    const oldLinksBuffer = device.createBuffer({ size: floatSize * 2, usage: GPUBufferUsage.STORAGE });
    const oldHiggsBuffer = device.createBuffer({ size: floatSize, usage: GPUBufferUsage.STORAGE });
    
    const rngState = new Uint32Array(N).map(() => Math.random() * 0xFFFFFFFF);
    const rngBuffer = device.createBuffer({ size: rngState.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint32Array(rngBuffer.getMappedRange()).set(rngState); rngBuffer.unmap();
    const workspaceBuffer = device.createBuffer({ size: floatSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const paramsBuffer = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // --- BindGroup 1: Fermion Core ---
    const phiBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const xBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE });
    const cgPBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE });
    const cgRBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE });
    const cgQBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const tmpYBuffer = device.createBuffer({ size: spinorSize, usage: GPUBufferUsage.STORAGE });
    const cgScalarsBuffer = device.createBuffer({ size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); // [rho, p_q, alpha, beta, new_rho]


    // --- デバッグ・検証用の読み出し設定 ---
    const debugBuffer = device.createBuffer({ size: 9 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });// scalars配列のサイズ (9個のf32)

    let triggerDebug = false;
    window.addEventListener('keydown', (e) => {
        if (e.key === 'd' || e.key === 'D') {
            const timerId = setInterval(()=>{
                triggerDebug = true;
            }, 10 * 1000);
        }
    });    

    // ========================================================================
    // 2. レイアウトとパイプラインの構築
    // ========================================================================
    const shaderCode = await fetchText('./wgsl/lgt/fermion_hmc_u1_gauge_higgs.wgsl');
    const module = device.createShaderModule({ code: shaderCode });

    const hmcGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // links
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // higgs
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // p_links
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // p_higgs
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // old_links
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // old_higgs
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // rng
            { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // hmc_scalars
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
    });

    const fermionGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // phi
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // x
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // cg_p
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // cg_r
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // cg_q
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // tmp_Y
        ]
    });

    const hmcBindGroup = device.createBindGroup({
        layout: hmcGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer, size: 16 } },
            { binding: 1, resource: { buffer: linksBuffer } },
            { binding: 2, resource: { buffer: higgsBuffer } },
            { binding: 3, resource: { buffer: pLinksBuffer } },
            { binding: 4, resource: { buffer: pHiggsBuffer } },
            { binding: 5, resource: { buffer: oldLinksBuffer } },
            { binding: 6, resource: { buffer: oldHiggsBuffer } },
            { binding: 7, resource: { buffer: rngBuffer } },
            { binding: 8, resource: { buffer: cgScalarsBuffer } }, // HMCとCGでスカラーは共有可能（用途がかぶらないため）
            { binding: 9, resource: { buffer: workspaceBuffer } },
        ]
    });

    const fermionBindGroup = device.createBindGroup({
        layout: fermionGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: phiBuffer } },
            { binding: 1, resource: { buffer: xBuffer } },
            { binding: 2, resource: { buffer: cgPBuffer } },
            { binding: 3, resource: { buffer: cgRBuffer } },
            { binding: 4, resource: { buffer: cgQBuffer } },
            { binding: 5, resource: { buffer: tmpYBuffer } },
        ]
    });

    const layout = device.createPipelineLayout({ bindGroupLayouts: [hmcGroupLayout, fermionGroupLayout] });
    const createCmp = async (entry: string) => device.createComputePipelineAsync({ layout, compute: { module, entryPoint: entry } });

    // HMC Pipelines
    const initHotPipeline = await createCmp('init_hot');
    const initColdPipeline = await createCmp('init_cold'); // ★この1行を追加！
    const initTrajPipeline = await createCmp('init_trajectory');
    const calcHPipeline = await createCmp('calc_local_H');
    const reduceHPipeline = await createCmp('reduce_H');
    const updateQPipeline = await createCmp('update_q');
    const updatePPipeline = await createCmp('update_P');
    const acceptRejectPipeline = await createCmp('accept_reject');
    const measurePipeline = await createCmp(mode === "E" ? "measure_observables_E" : "measure_observables_C");

    // Fermion Pipelines
    const generateXiPipeline = await createCmp('generate_xi');
    const applyDDagForCgPipeline = await createCmp('apply_D_dag_for_cg');
    const applyDForCgPipeline = await createCmp('apply_D_for_cg');
    const resetCgPipeline = await createCmp('reset_cg');
    const calcInitialRhoPipeline = await createCmp('calc_initial_rho');
    const calcPQPipeline = await createCmp('calc_p_q');
    const updateXRPipeline = await createCmp('update_x_r');
    const calcNewRhoPipeline = await createCmp('calc_new_rho');
    const updateCGPPipeline = await createCmp('update_p');
    const computeYPipeline = await createCmp('compute_Y_from_x');
    const calcFermionForcePipeline = await createCmp('calc_fermion_force');

    // ========================================================================
    // レンダーパイプラインの作成
    // ========================================================================
    const renderName = mode === "E" ? "u1_higgs_render-E" : "u1_higgs_render-C";
    const renderShaderCode = "const L: f32 = 32.0;\n" + await fetchText(`./wgsl/lgt/${renderName}.wgsl`);
    const renderModule = device.createShaderModule({ code: renderShaderCode });

    const canvas = document.querySelector("#world-webgpu") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu") as GPUCanvasContext;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: presentationFormat, alphaMode: "premultiplied" });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }]
    });

    // モードC (トポロジカル電荷) なら workspaceBuffer、モードE (ヒッグス位相) なら higgsBuffer を描画
    const renderBuffer = mode === "C" ? workspaceBuffer : higgsBuffer;
    
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

    // ========================================================================
    // 3. アニメーションループ (The Great Orchestration)
    // ========================================================================
    let animationId: number | null = null;

    const md_steps = 20; // 軌道を保つために歩数を増やす

    // ↓ CGソルバーが答えを見つけるまで、たっぷり回数を回してあげる。質量が軽いほど多く必要
    const cg_iters = 100; 

    let eps = 0.005;     // 歩幅を細かくしてカーブを正確に曲がる

    // パラメータ更新関数 (Dynamic Offset用)
    function writeParams() {
        const simParams = new Float32Array(256 * 4 / 4);
        const uintView = new Uint32Array(simParams.buffer);
        // [beta, kappa, eps, mass_or_isNewH]
        // Offset 0: H_old (eps=0, isNew=0)
        simParams[0] = betaValue; simParams[1] = kappaValue; simParams[2] = 0.0; uintView[3] = 0;
        // Offset 256: eps/2
        simParams[64] = betaValue; simParams[65] = kappaValue; simParams[66] = eps / 2.0; simParams[67] = massValue;
        // Offset 512: eps
        simParams[128] = betaValue; simParams[129] = kappaValue; simParams[130] = eps; simParams[131] = massValue;
        // Offset 768: H_new (eps=0, isNew=1)
        simParams[192] = betaValue; simParams[193] = kappaValue; simParams[194] = 0.0; uintView[195] = 1;
        device.queue.writeBuffer(paramsBuffer, 0, simParams);
    }
    writeParams();

    // Hot Start
    const initEnc = device.createCommandEncoder();
    const initPass = initEnc.beginComputePass();
    // initPass.setPipeline(initHotPipeline); 
    initPass.setPipeline(initColdPipeline); // ✅ Cold Start に変更！
    initPass.setBindGroup(0, hmcBindGroup, [0]); 
    initPass.setBindGroup(1, fermionBindGroup);
    initPass.dispatchWorkgroups(workgroups); 
    initPass.end();
    device.queue.submit([initEnc.finish()]);

// ========================================================================
    // 3. アニメーションループ (ステートマシン版 The Great Orchestration)
    // ========================================================================

    // --- ステートマシン用の変数 ---
    let hmc_state = 0; 
    let current_md_step = 0;

    // (writeParams と Hot Start の部分はそのまま残してください)

    function frame() {
        const commandEncoder = device.createCommandEncoder();

        if (hmc_state === 0) {
            const pass = commandEncoder.beginComputePass();
            pass.setBindGroup(1, fermionBindGroup);

            // 1. 軌道初期化と P のサンプリング
            pass.setBindGroup(0, hmcBindGroup, [0]);
            pass.setPipeline(initTrajPipeline); pass.dispatchWorkgroups(workgroups);
            
            // ↓↓↓ 修正①：フェルミオン生成には「質量(mass)」が必要なので offset [512] に切り替え！ ↓↓↓
            pass.setBindGroup(0, hmcBindGroup, [512]); 
            pass.setPipeline(generateXiPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(applyDDagForCgPipeline); pass.dispatchWorkgroups(workgroups);
            pass.end(); 

            commandEncoder.copyBufferToBuffer(cgQBuffer, 0, phiBuffer, 0, spinorSize);

            // 3. 初期位置 Q_0 での CGソルバー (x_0 を求める)
            const passCg = commandEncoder.beginComputePass();
            passCg.setBindGroup(1, fermionBindGroup);
            
            // ↓↓↓ 修正②：最初のCGソルバーにも「質量(mass)」が必要なので offset [512] をセット！ ↓↓↓
            passCg.setBindGroup(0, hmcBindGroup, [512]); 
            
            passCg.setPipeline(resetCgPipeline); passCg.dispatchWorkgroups(workgroups);
            passCg.setPipeline(calcInitialRhoPipeline); passCg.dispatchWorkgroups(1);
            for(let i = 0; i < cg_iters; i++) {
                passCg.setPipeline(applyDForCgPipeline); passCg.dispatchWorkgroups(workgroups);
                passCg.setPipeline(applyDDagForCgPipeline); passCg.dispatchWorkgroups(workgroups);
                passCg.setPipeline(calcPQPipeline); passCg.dispatchWorkgroups(1);
                passCg.setPipeline(updateXRPipeline); passCg.dispatchWorkgroups(workgroups);
                passCg.setPipeline(calcNewRhoPipeline); passCg.dispatchWorkgroups(1);
                passCg.setPipeline(updateCGPPipeline); passCg.dispatchWorkgroups(workgroups);
            }
            passCg.end();

            // 4. H_old の計算 と 【最初の P 半歩】
            const pass2 = commandEncoder.beginComputePass();
            pass2.setBindGroup(1, fermionBindGroup);
            
            // ↓↓↓ 修正③：H_old の計算には isNewH=0 のフラグが必要なので offset [0] に戻す ↓↓↓
            pass2.setBindGroup(0, hmcBindGroup, [0]); 
            pass2.setPipeline(calcHPipeline); pass2.dispatchWorkgroups(workgroups);
            pass2.setPipeline(reduceHPipeline); pass2.dispatchWorkgroups(1);

            // 最初のP半歩には eps/2 と質量が必要なので offset [256]
            pass2.setBindGroup(0, hmcBindGroup, [256]); // eps/2
            pass2.setPipeline(computeYPipeline); pass2.dispatchWorkgroups(workgroups);
            pass2.setPipeline(calcFermionForcePipeline); pass2.dispatchWorkgroups(workgroups);
            pass2.setPipeline(updatePPipeline); pass2.dispatchWorkgroups(workgroups);
            
            pass2.end();

            hmc_state = 1;
            current_md_step = 0;

        } 
        else if (hmc_state === 1) {
            const pass = commandEncoder.beginComputePass();
            pass.setBindGroup(1, fermionBindGroup);

            // 1. Q を 1歩 進める
            pass.setBindGroup(0, hmcBindGroup, [512]); // eps
            pass.setPipeline(updateQPipeline); pass.dispatchWorkgroups(workgroups);
            current_md_step++;

            // 2. 新しい Q で CGソルバーを解く
            pass.setPipeline(resetCgPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(calcInitialRhoPipeline); pass.dispatchWorkgroups(1);
            for(let i = 0; i < cg_iters; i++) {
                pass.setPipeline(applyDForCgPipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(applyDDagForCgPipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(calcPQPipeline); pass.dispatchWorkgroups(1);
                pass.setPipeline(updateXRPipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(calcNewRhoPipeline); pass.dispatchWorkgroups(1);
                pass.setPipeline(updateCGPPipeline); pass.dispatchWorkgroups(workgroups);
            }

            // 3. P を更新する
            if (current_md_step < md_steps) {
                // 途中なら P を 1歩進める (eps)
                pass.setBindGroup(0, hmcBindGroup, [512]);
                pass.setPipeline(computeYPipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(calcFermionForcePipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(updatePPipeline); pass.dispatchWorkgroups(workgroups);
            } else {
                // ↓↓↓ 修正の核心: 最後の半歩(eps/2)でもフェルミオン力を確実に計算！ ↓↓↓
                pass.setBindGroup(0, hmcBindGroup, [256]); // eps/2
                pass.setPipeline(computeYPipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(calcFermionForcePipeline); pass.dispatchWorkgroups(workgroups);
                pass.setPipeline(updatePPipeline); pass.dispatchWorkgroups(workgroups);
                hmc_state = 2;
                // ↑↑↑
            }
            pass.end();

        } else if (hmc_state === 2) {
            const pass = commandEncoder.beginComputePass();
            pass.setBindGroup(1, fermionBindGroup);

            // 【State 2: H_new 計算と受容判定】
            pass.setBindGroup(0, hmcBindGroup, [768]); // isNewH = 1
            pass.setPipeline(calcHPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(reduceHPipeline); pass.dispatchWorkgroups(1);
            
            pass.setBindGroup(0, hmcBindGroup, [0]);
            pass.setPipeline(acceptRejectPipeline); pass.dispatchWorkgroups(workgroups);

            pass.end();

            if (triggerDebug) {
                triggerDebug = false;
                commandEncoder.copyBufferToBuffer(cgScalarsBuffer, 0, debugBuffer, 0, 9 * 4);
                (window as any).shouldReadDebug = true; 
            }

            hmc_state = 0; 
        }

        // ↓↓↓ 追加：描画の直前に必ず Q を再計算してバッファを上書きする ↓↓↓
        const vizPass = commandEncoder.beginComputePass();
        vizPass.setPipeline(measurePipeline);
        vizPass.setBindGroup(0, hmcBindGroup, [0]); 
        vizPass.setBindGroup(1, fermionBindGroup); // レイアウトの都合上これも必須
        vizPass.dispatchWorkgroups(L/8, L/8, 1);
        vizPass.end();
        // ↑↑↑

        // --- 描画パス (毎フレーム必ず実行して画面フリーズを防ぐ) ---
        const canvas = document.querySelector("#world-webgpu") as HTMLCanvasElement;
        const context = canvas.getContext("webgpu") as GPUCanvasContext;
        const renderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{ 
                view: context.getCurrentTexture().createView(), 
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, 
                loadOp: 'clear', storeOp: 'store' 
            }]
        });
        renderPassEncoder.setPipeline(renderPipeline);
        renderPassEncoder.setBindGroup(0, renderBindGroup);
        renderPassEncoder.draw(6, N, 0, 0); 
        renderPassEncoder.end();

        device.queue.submit([commandEncoder.finish()]);

        // --- 修正: コピー命令が送信されたフレームの直後だけ読み出す ---
        if ((window as any).shouldReadDebug) {
            (window as any).shouldReadDebug = false; // フラグを戻す
            
            debugBuffer.mapAsync(GPUMapMode.READ).then(() => {
                const data = new Float32Array(debugBuffer.getMappedRange());
                const rho = data[0];       // CGソルバー最終ステップの残差 |r|^2
                const H_old = data[5];     // 軌道開始時のエネルギー
                const H_new = data[6];     // 軌道終了時のエネルギー
                const accepted = data[7];  // 1なら受容、0なら棄却
                const dH = H_new - H_old;  // エネルギー変動

            msg(`--- Debug Readout Triggered ---
[HMC] H_old: ${H_old.toFixed(2)}, H_new: ${H_new.toFixed(2)}, ΔH: ${dH.toFixed(4)} (Accepted: ${accepted})
[CG]  Final Residual ρ: ${rho.toExponential(3)}`);

                debugBuffer.unmap();
            });
        }
        
        animationId = requestAnimationFrame(frame);
    }

    frame();
    return () => { if (animationId) cancelAnimationFrame(animationId); };
}