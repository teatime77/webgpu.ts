import { fetchJson, fetchText } from '@i18n';
import { OrbitCamera } from './camera';
import { SimulationSchema } from './control';

/**
 * 外部のWGSLファイルを読み込み、Record<string, string> に分割して返すパーサー
 */
async function loadShaders(url: string): Promise<Record<string, string>> {
    const response = await fetch(url);
    const text = await response.text();

    const shaders: Record<string, string> = {};
    
    // 正規表現で行頭にある "// @shader: [名前]" を見つけて分割する
    // (\s* はスペースの揺らぎを許容、(.+)$ は行末までの名前を取得)
    const parts = text.split(/^\/\/\s*@shader:\s*(.+)$/gm);

    // parts[0] は最初の @shader より前にある文字列（グローバルなコメントなど）なので無視
    // parts は [ "ゴミ", "init_particles", "WGSLコード...", "update_particles", "WGSLコード..." ] のように並ぶ
    for (let i = 1; i < parts.length; i += 2) {
        const name = parts[i].trim();
        const code = parts[i + 1].trim();
        shaders[name] = parts[0] + code;
    }

    return shaders;
}

import { GraphManager } from './control';
import { simulationAssetBaseUrl } from './schema_public_path';
import { ShapeInfo } from './package';
import { makeGeodesicPolyhedron, makeArrowMesh } from './primitive';
import { buildUI } from './sim_ui';

// ホットリロード(HMR)時の二重起動を防ぐためのグローバル変数
let animationId: number | null = null;

/** Default wall time for `schema=all` before switching to the next physics engine. */
export const PHYSICS_SCHEMA_DEMO_DWELL_MS = 3_000;

export function stopGraphSchemaAnimation(): void {
    if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

export interface InitControlOptions {
    /** If set, stop the render loop after this many ms (sequential `schema=all` demos). */
    dwellMs?: number;
}

function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function captureCanvasPng(canvas: HTMLCanvasElement, filename: string): Promise<void> {
    await new Promise<void>((resolve, reject) => {
        canvas.toBlob(blob => {
            if (!blob) {
                reject(new Error("Failed to capture canvas image."));
                return;
            }
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            resolve();
        }, "image/png");
    });
}

function setupCapturePanel(canvas: HTMLCanvasElement, schemaName: string): void {
    if (document.getElementById("capture-panel")) return;

    const app = document.getElementById("app-container");
    if (!app) return;

    const panel = document.createElement("div");
    panel.id = "capture-panel";
    panel.style.position = "absolute";
    panel.style.left = "10px";
    panel.style.bottom = "10px";
    panel.style.padding = "8px";
    panel.style.borderRadius = "8px";
    panel.style.background = "rgba(20, 20, 25, 0.85)";
    panel.style.color = "white";
    panel.style.fontFamily = "sans-serif";
    panel.style.border = "1px solid #444";
    panel.style.display = "flex";
    panel.style.gap = "8px";
    panel.style.alignItems = "center";
    panel.style.zIndex = "20";

    const captureBtn = document.createElement("button");
    captureBtn.textContent = "Capture";
    captureBtn.style.fontSize = "14px";

    const burstBtn = document.createElement("button");
    burstBtn.textContent = "Burst xN";
    burstBtn.style.fontSize = "14px";

    const countInput = document.createElement("input");
    countInput.type = "number";
    countInput.min = "1";
    countInput.max = "200";
    countInput.step = "1";
    countInput.value = "10";
    countInput.style.width = "60px";
    countInput.title = "number of images";

    const intervalInput = document.createElement("input");
    intervalInput.type = "number";
    intervalInput.min = "10";
    intervalInput.max = "5000";
    intervalInput.step = "10";
    intervalInput.value = "100";
    intervalInput.style.width = "70px";
    intervalInput.title = "interval ms";

    const countLabel = document.createElement("span");
    countLabel.textContent = "N";
    countLabel.style.fontSize = "12px";
    const intervalLabel = document.createElement("span");
    intervalLabel.textContent = "ms";
    intervalLabel.style.fontSize = "12px";

    let isCapturing = false;
    const setBusy = (busy: boolean) => {
        isCapturing = busy;
        captureBtn.disabled = busy;
        burstBtn.disabled = busy;
    };

    captureBtn.addEventListener("click", async () => {
        if (isCapturing) return;
        setBusy(true);
        try {
            await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
            const stamp = Date.now();
            await captureCanvasPng(canvas, `${schemaName}_${stamp}.png`);
        } finally {
            setBusy(false);
        }
    });

    burstBtn.addEventListener("click", async () => {
        if (isCapturing) return;
        const count = Math.max(1, Math.min(200, Number(countInput.value) || 1));
        const intervalMs = Math.max(10, Math.min(5000, Number(intervalInput.value) || 100));
        setBusy(true);
        try {
            const base = Date.now();
            for (let i = 0; i < count; i++) {
                await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
                await captureCanvasPng(canvas, `${schemaName}_${base}_${String(i).padStart(3, "0")}.png`);
                if (i < count - 1) {
                    await sleep(intervalMs);
                }
            }
        } finally {
            setBusy(false);
        }
    });

    panel.appendChild(captureBtn);
    panel.appendChild(burstBtn);
    panel.appendChild(countInput);
    panel.appendChild(countLabel);
    panel.appendChild(intervalInput);
    panel.appendChild(intervalLabel);
    app.appendChild(panel);
}

// ※ 上部に sphereSchema と wgslCodes が定義されている前提です。

export async function initControl(schemaName: string, options?: InitControlOptions): Promise<void> {
    stopGraphSchemaAnimation();

    // 1. WebGPUデバイスとCanvasの取得
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const canvas = document.getElementById('world-webgpu') as HTMLCanvasElement;
    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    context.unconfigure(); 
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'premultiplied' });

    const camera = new OrbitCamera(canvas);
    // ========================================================
    // 🌟 ステップ 3: テクスチャのロードとリソースの注入
    // ========================================================
    
    // ① MatCap画像の読み込みとテクスチャ化
    // ※ プロジェクト内に 'matcap.jpg' などの画像を用意してください
    const response = await fetch('matcap.png'); 
    const imageBitmap = await createImageBitmap(await response.blob());
    const texture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: texture },
        [imageBitmap.width, imageBitmap.height]
    );

    const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
    });

    // ② エンジンのインスタンス化とコンテキスト設定
    const engine = new GraphManager(device, schemaName);
    
    engine.setContext(context, format, canvas.width, canvas.height);

    // ③ 外部リソース（テクスチャ）をエンジンに登録
    engine.setExternalResource('MatCapTex', texture.createView());
    engine.setExternalResource('MatCapSampler', sampler);

    // 🌟 修正: スキーマをロードする【前】に実際の頂点データを生成し、メタデータを書き換える
    const sphereVertices = makeGeodesicPolyhedron({divideCount : 3} as ShapeInfo);
    const arrowVertices = makeArrowMesh({numDivision: 16} as ShapeInfo);
    
    const schema = await fetchJson(`${simulationAssetBaseUrl(schemaName)}/${schemaName}.json`) as SimulationSchema;

    schema.metadata.baseSphereFloatCount = sphereVertices.length;
    schema.metadata.baseSphereVertexCount = sphereVertices.length / 6; // x,y,z,nx,ny,nz

    schema.metadata.baseArrowFloatCount = arrowVertices.length;
    schema.metadata.baseArrowVertexCount = arrowVertices.length / 6;

    await engine.parseSchemaScript(schema);


    // ④ スキーマのロード（ここでバッファが確保される）
    engine.loadSchema(schema);

    if(engine.resources.has('BaseSphere')){
        device.queue.writeBuffer(
            engine.resources.get('BaseSphere')!.getBuffer(0), 
            0, 
            sphereVertices
        );
    }
    if(engine.resources.has('BaseArrow')){
        device.queue.writeBuffer(
            engine.resources.get('BaseArrow')!.getBuffer(0), 
            0, 
            arrowVertices
        );
    }

    // ========================================================
    // パイプラインコンパイルと変数更新
    // ========================================================
    const codes = await loadShaders(`${simulationAssetBaseUrl(schemaName)}/${schemaName}.wgsl`);
    await engine.compilePipelines(codes);

    // カメラ行列や粒子数の初期設定
    // ※ glMatrixなどのライブラリを使って、実際の4x4行列(16要素の配列)を計算して渡します
    const identityMatrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; 

    // カメラを少し後ろに下げて全体を映す簡易的な行列 (Orthographicな縮小)
    const scale = 0.05;
    const viewProjMatrix = [
        scale, 0, 0, 0, 
        0, scale, 0, 0, 
        0, 0, scale, 0, 
        0, 0, 0,   1
    ];

    const initialParticleCount = typeof schema.metadata.particleCount === "number"
        ? schema.metadata.particleCount
        : 1024;
    engine.updateVariables({
        particleCount: initialParticleCount,
        viewProjection: viewProjMatrix,
        view: identityMatrix,
    });

    // ========================================================
    // UIパネルの構築 (schema.uis があれば自動生成)
    // ========================================================
    buildUI(engine, schema);
    setupCapturePanel(canvas, schemaName);

    // ========================================================
    // メインループ
    // ========================================================
    function frame() {
        // 1. 画面のアスペクト比(縦横比)を計算
        const aspect = canvas.width / canvas.height;
        
        // 2. マウス入力から最新の行列を計算して取得
        const matrices = camera.getMatrices(aspect);

        // 3. エンジンのメタデータ(GPUのUniformバッファ)を更新
        engine.updateVariables({
            time: performance.now() / 1000.0,
            viewProjection: matrices.viewProjection as number[],
            view: matrices.view
        });
        
        // 4. シミュレーションと描画を1ステップ進める
        engine.step();
        
        animationId = requestAnimationFrame(frame);
        (window as any).currentAnimationId = animationId;
    }
    
    frame();

    if (options?.dwellMs != null && options.dwellMs > 0) {
        await sleep(options.dwellMs);
        stopGraphSchemaAnimation();
    }
}