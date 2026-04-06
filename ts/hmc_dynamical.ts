import { $, $inp, fetchText, msg } from "@i18n";
import { stopAnimation } from "./instance";

let betaValue: number = 4.0;
let kappaValue: number = 1.0;
let massValue: number = 0.1; // クォークの軽さ (小さいほどCGが重くなる)

export async function runDynamicalFermions(device: GPUDevice, mode: "C" | "E"): Promise<() => void> {
    msg("Starting Dynamical Fermions (HMC + CG Solver) simulation...");
    stopAnimation();

    const L = 64;
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
    const cgScalarsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE }); // [rho, p_q, alpha, beta, new_rho]

    // ========================================================================
    // 2. レイアウトとパイプラインの構築
    // ========================================================================
    const shaderCode = await fetchText('./wgsl/lgt_dynamical.wgsl');
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
    const renderShaderCode = await fetchText(`./wgsl/${renderName}.wgsl`);
    const renderModule = device.createShaderModule({ code: renderShaderCode });

    const canvas = document.querySelector("#world") as HTMLCanvasElement;
    const context = canvas.getContext("webgpu")!;
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
    const md_steps = 10;
    const cg_iters = 30; // 質量が軽いほど多く必要
    let eps = 0.05;

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
    initPass.setPipeline(initHotPipeline); initPass.setBindGroup(0, hmcBindGroup, [0]); initPass.setBindGroup(1, fermionBindGroup);
    initPass.dispatchWorkgroups(workgroups); initPass.end();
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
        
        // --- 物理計算パス ---
        const pass = commandEncoder.beginComputePass();
        pass.setBindGroup(1, fermionBindGroup);

        if (hmc_state === 0) {
            // 【State 0: 軌道初期化と熱浴】
            pass.setBindGroup(0, hmcBindGroup, [0]);
            pass.setPipeline(initTrajPipeline); pass.dispatchWorkgroups(workgroups);
            
            // 擬フェルミオン phi の生成 (xi -> tmp_Y -> cg_q)
            pass.setPipeline(generateXiPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(applyDDagForCgPipeline); pass.dispatchWorkgroups(workgroups);
            pass.end(); // 一旦パスを閉じてバッファコピー

            // cg_q に入った結果を phi バッファにコピー (超重要！)
            commandEncoder.copyBufferToBuffer(cgQBuffer, 0, phiBuffer, 0, spinorSize);

            const pass2 = commandEncoder.beginComputePass();
            pass2.setBindGroup(1, fermionBindGroup);
            pass2.setBindGroup(0, hmcBindGroup, [0]);

            // H_old の計算
            pass2.setPipeline(calcHPipeline); pass2.dispatchWorkgroups(workgroups);
            pass2.setPipeline(reduceHPipeline); pass2.dispatchWorkgroups(1);

            // P を半歩進める
            pass2.setBindGroup(0, hmcBindGroup, [256]); // eps/2
            pass2.setPipeline(updatePPipeline); pass2.dispatchWorkgroups(workgroups);
            pass2.end();

            hmc_state = 1;
            current_md_step = 0;

        } 
        else if (hmc_state === 1) {
            pass.setBindGroup(0, hmcBindGroup, [256]); 

            // 【State 1: リープフロッグ1歩分 ＋ CGソルバー】(毎フレーム1歩だけ進める)
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

            // フェルミオン力
            pass.setBindGroup(0, hmcBindGroup, [512]); // eps
            pass.setPipeline(computeYPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(calcFermionForcePipeline); pass.dispatchWorkgroups(workgroups);

            // ゲージ・ヒッグス力
            pass.setPipeline(updatePPipeline); pass.dispatchWorkgroups(workgroups);

            if (current_md_step < md_steps - 1) {
                // まだ軌道の途中なら q を1歩進める
                pass.setPipeline(updateQPipeline); pass.dispatchWorkgroups(workgroups);
                current_md_step++;
            } else {
                // 最後のステップ：qを1歩、Pを半歩
                pass.setPipeline(updateQPipeline); pass.dispatchWorkgroups(workgroups);
                pass.setBindGroup(0, hmcBindGroup, [256]); // eps/2
                pass.setPipeline(updatePPipeline); pass.dispatchWorkgroups(workgroups);
                hmc_state = 2; // 次は判定へ
            }
            pass.end();

        } else if (hmc_state === 2) {
            // 【State 2: 受容判定と観測】
            pass.setBindGroup(0, hmcBindGroup, [768]); // H_new
            pass.setPipeline(calcHPipeline); pass.dispatchWorkgroups(workgroups);
            pass.setPipeline(reduceHPipeline); pass.dispatchWorkgroups(1);
            
            pass.setBindGroup(0, hmcBindGroup, [0]);
            pass.setPipeline(acceptRejectPipeline); pass.dispatchWorkgroups(workgroups);

            pass.setPipeline(measurePipeline); pass.dispatchWorkgroups(L/8, L/8, 1);
            pass.end();
            
            hmc_state = 0; // 次の新しい軌道へループ！
        }

        // --- 描画パス (毎フレーム必ず実行して画面フリーズを防ぐ) ---
        const canvas = document.querySelector("#world") as HTMLCanvasElement;
        const context = canvas.getContext("webgpu")!;
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
        animationId = requestAnimationFrame(frame);
    }

    frame();
    return () => { if (animationId) cancelAnimationFrame(animationId); };
}