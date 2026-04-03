import { msg } from "@i18n";

export async function runFeynmanSimulation(device: GPUDevice) {
    msg("Starting Beta Decay Phase Space Simulation...");

    // --- 1. シミュレーション定数 ---
    const NUM_BINS = 100;           // ヒストグラムの分割数 (WGSLと一致させる)
    const WORKGROUP_SIZE = 64;      // 1ワークグループあたりのスレッド数
    const NUM_WORKGROUPS = 1024;    // ワークグループの数
    const TOTAL_THREADS = WORKGROUP_SIZE * NUM_WORKGROUPS; // 計 65,536 スレッド
    const EVENTS_PER_THREAD = 100;  // 1スレッドあたりの試行回数 (WGSL内のループ回数と一致)
    const TOTAL_EVENTS = TOTAL_THREADS * EVENTS_PER_THREAD; // 約 650万イベント！

    console.log(`Total virtual decay events: ${TOTAL_EVENTS}`);

    // --- 2. シェーダーの読み込み ---
    // 先ほど作成した feynman_beta.wgsl を読み込みます
    const response = await fetch('./wgsl/feynman_beta.wgsl');
    const shaderCode = await response.text();
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // --- 3. バッファの作成 ---
    // ヒストグラム用バッファ (u32配列、初期値はすべて0)
    const histogramBuffer = device.createBuffer({
        size: NUM_BINS * 4, // 100 bins * 4 bytes (u32)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // バッファをゼロクリアする (毎回ゼロからカウントするため)
    const zeroData = new Uint32Array(NUM_BINS);
    device.queue.writeBuffer(histogramBuffer, 0, zeroData);

    // 乱数シード用バッファ
    const rngState = new Uint32Array(TOTAL_THREADS);
    for (let i = 0; i < TOTAL_THREADS; i++) {
        rngState[i] = Math.floor(Math.random() * 0xFFFFFFFF);
    }
    const rngBuffer = device.createBuffer({
        size: TOTAL_THREADS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(rngBuffer, 0, rngState);

    // CPU読み戻し用バッファ
    const readbackBuffer = device.createBuffer({
        size: NUM_BINS * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // --- 4. パイプラインとバインドグループの作成 ---
    const computePipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'generate_events',
        },
    });

    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: histogramBuffer } },
            { binding: 1, resource: { buffer: rngBuffer } },
        ],
    });

    // --- 5. GPUコンピュートパスの実行 ---
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(NUM_WORKGROUPS, 1, 1);
    pass.end();

    // 結果を readbackBuffer にコピー
    commandEncoder.copyBufferToBuffer(histogramBuffer, 0, readbackBuffer, 0, NUM_BINS * 4);
    device.queue.submit([commandEncoder.finish()]);

    // GPUの処理完了を待機
    await device.queue.onSubmittedWorkDone();

    // --- 6. 結果の読み取り ---
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const histogramData = new Uint32Array(readbackBuffer.getMappedRange()).slice(); // コピーを取得
    readbackBuffer.unmap();

    // --- 7. 結果の描画 (Canvas) ---
    drawHistogram(histogramData);

    const text = Array.from(histogramData).map(i => `${i}`).join("\n");
    msg(`histogram\n${text}`);
}

// 簡易的な棒グラフ描画関数
function drawHistogram(data: Uint32Array) {
    // Canvasタグを取得 (既存の格子ゲージ理論用のcanvasを再利用するか、新規作成)
    let canvas = document.querySelector("plot-canvas") as HTMLCanvasElement;
    if (!canvas) {
        canvas = document.createElement("canvas");
        document.body.appendChild(canvas);
    }
    
    // グラフが見やすいようにアスペクト比を調整
    canvas.width = 800;
    canvas.height = 400;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // 背景を黒に
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // データの中の最大値を探す (グラフの高さを正規化するため)
    let maxCount = 0;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > maxCount) maxCount = data[i];
    }

    // 描画設定
    const padding = 40;
    const chartWidth = canvas.width - padding * 2;
    const chartHeight = canvas.height - padding * 2;
    const barWidth = chartWidth / data.length;

    // 棒グラフを描画
    ctx.fillStyle = "#00ff88"; // 蛍光グリーン
    for (let i = 0; i < data.length; i++) {
        const height = (data[i] / maxCount) * chartHeight;
        const x = padding + i * barWidth;
        const y = canvas.height - padding - height;
        
        // 棒を描画
        ctx.fillRect(x, y, barWidth - 1, height);
    }

    // 軸とラベルを描画
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.font = "14px Arial";
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    ctx.fillText("0 keV", padding, canvas.height - padding + 20);
    ctx.fillText("Energy (K_e) ->", canvas.width / 2 - 40, canvas.height - padding + 35);
    ctx.fillText("18.6 keV (Q-value)", canvas.width - padding - 100, canvas.height - padding + 20);
    ctx.fillText("Events", 10, padding - 10);
    
    console.log("Histogram drawing complete!");
}
