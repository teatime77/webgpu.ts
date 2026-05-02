import { fetchText } from '@i18n';
import { OrbitCamera } from './camera';
import { SimulationSchema, sphereSchema } from './control';

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
        shaders[name] = code;
    }

    return shaders;
}


export const gameOfLifeSchema: SimulationSchema = {
    name: "Conway's Game of Life",
    version: "1.0",
    
    // 1. メタデータ (シミュレーションの解像度など)
    metadata: {
        gridWidth: 256,
        gridHeight: 256
    },

    // 2. リソース定義
    resources: {
        "GlobalUniforms": {
            type: "uniform",
            fields: {
                "gridWidth": "f32",
                "gridHeight": "f32"
            }
        },
        // Ping-Pongバッファとして宣言！システムが裏で2つ作ります。
        "CellBuffer": {
            type: "storage",
            access: "read_write",
            format: "u32",
            count: "$metadata.gridWidth * $metadata.gridHeight",
            pingPong: true 
        }
    },

    // 3. ノード定義
    nodes: {
        // [A] 初期化ノード (ランダムに生と死を配置)
        "InitCells": {
            type: "compute",
            shader: "init_cells",
            dispatch: {
                type: "direct",
                workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1]
            },
            bindings: [
                { group: 0, binding: 0, resource: "GlobalUniforms" },
                { group: 0, binding: 1, resource: "CellBuffer", state: "current" }
            ]
        },
        // [B] 更新ノード (ライフゲームのルール適用)
        "UpdateCells": {
            type: "compute",
            shader: "update_cells",
            dispatch: {
                type: "direct",
                workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1]
            },
            bindings: [
                { group: 0, binding: 0, resource: "GlobalUniforms" },
                { group: 0, binding: 1, resource: "CellBuffer", state: "previous" }, // 前のフレームを読んで
                { group: 0, binding: 2, resource: "CellBuffer", state: "current" }   // 新しいフレームに書く
            ]
        },
        // [C] 描画ノード (フルスクリーンクアッドでグリッドを描画)
        "RenderCells": {
            type: "render",
            shader: "render_cells",
            topology: "triangle-list",
            draw: { type: "direct", vertexCount: 6 }, // 2つの三角形(6頂点)を描画
            bindings: [
                { group: 0, binding: 0, resource: "GlobalUniforms" },
                { group: 0, binding: 1, resource: "CellBuffer", state: "current" }
            ]
        }
    },

    // 4. ステートマシン (実行制御)
    stateMachine: {
        initialState: "State_Initialize",
        states: {
            "State_Initialize": {
                execute: ["InitCells", "RenderCells"],
                // 🌟 追加: 初期化が終わったら、必ずバッファを裏返す！
                onExit: [{ action: "swapPingPong", resources: ["CellBuffer"] }],
                transitions: [{ condition: "true", "target": "State_Simulate" }]
            },
            "State_Simulate": {
                execute: ["UpdateCells", "RenderCells"],
                onExit: [{ action: "swapPingPong", resources: ["CellBuffer"] }],
                transitions: [{ condition: "true", "target": "State_Simulate" }]
            }
        }
    }
};



import { GraphManager } from './control';
import { ShapeInfo } from './package';
import { makeGeodesicPolyhedron } from './primitive';

// ホットリロード(HMR)時の二重起動を防ぐためのグローバル変数
let animationId: number | null = null;

// ※ 上部に sphereSchema と wgslCodes が定義されている前提です。

export async function initControl() {
    // --- (HMR二重起動防止処理などそのまま) ---
    if ((window as any).isSimulationRunning) {
        if ((window as any).currentAnimationId) {
            cancelAnimationFrame((window as any).currentAnimationId);
        }
    }
    (window as any).isSimulationRunning = true;

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
    const engine = new GraphManager(device);
    engine.setContext(context, format, canvas.width, canvas.height);

    // ③ 外部リソース（テクスチャ）をエンジンに登録
    engine.setExternalResource('MatCapTex', texture.createView());
    engine.setExternalResource('MatCapSampler', sampler);

    // 🌟 修正: スキーマをロードする【前】に実際の頂点データを生成し、メタデータを書き換える
    // ⑤ GeodesicPolyhedron の頂点データを BaseSphere バッファに書き込む
    const sphereVertices = makeGeodesicPolyhedron({divideCount : 3} as ShapeInfo);
    
    sphereSchema.metadata.baseSphereFloatCount = sphereVertices.length;
    sphereSchema.metadata.baseSphereVertexCount = sphereVertices.length / 6; // x,y,z,nx,ny,nz

    // ④ スキーマのロード（ここでバッファが確保される）
    engine.loadSchema(sphereSchema); // 🚨 gameOfLifeSchema から変更！

    device.queue.writeBuffer(
        engine.resources.get('BaseSphere')!.getCurrentBuffer(), 
        0, 
        sphereVertices
    );

    // ========================================================
    // パイプラインコンパイルと変数更新
    // ========================================================
    const codes = await loadShaders("./wgsl/ball/ball.wgsl");
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

    engine.updateVariables({
        particleCount: 1024, // 描画する粒子の数
        viewProjection: viewProjMatrix, // カメラのプロジェクション * ビュー行列
        view: identityMatrix            // カメラのビュー行列 (MatCap計算用)
    });

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
            viewProjection: matrices.viewProjection as number[],
            view: matrices.view
        });
        
        // 4. シミュレーションと描画を1ステップ進める
        engine.step();
        
        animationId = requestAnimationFrame(frame);
        (window as any).currentAnimationId = animationId;
    }
    
    frame();
}