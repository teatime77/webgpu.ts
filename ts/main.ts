import { SimulationSchema, sphereSchema } from './control';

// wgsl_life.ts 等に分離して保存することをおすすめします
export const lifeWgslCodes: Record<string, string> = {

    "init_cells": `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32(GlobalUniforms.gridWidth);
    let h = u32(GlobalUniforms.gridHeight);
    if (id.x >= w || id.y >= h) { return; }

    let idx = id.y * w + id.x;
    
    // シンプルな擬似乱数ハッシュで20%の確率でセルを「生(1)」にする
    let hash = (id.x * 374761393u + id.y * 668265263u) ^ 1013904223u;
    if (hash % 100u < 20u) {
        CellBuffer[idx] = 1u;
    } else {
        CellBuffer[idx] = 0u;
    }
}
`,

    "update_cells": `
fn get_cell(x: i32, y: i32, w: i32, h: i32) -> u32 {
    // トーラス状に世界を繋ぐ (Wrap around)
    let nx = (x + w) % w;
    let ny = (y + h) % h;
    return CellBufferPrev[u32(ny * w + nx)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = i32(GlobalUniforms.gridWidth);
    let h = i32(GlobalUniforms.gridHeight);
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= w || y >= h) { return; }

    // 周囲8マスの生存セルをカウント
    var alive_neighbors = 0u;
    alive_neighbors += get_cell(x - 1, y - 1, w, h);
    alive_neighbors += get_cell(x,     y - 1, w, h);
    alive_neighbors += get_cell(x + 1, y - 1, w, h);
    alive_neighbors += get_cell(x - 1, y,     w, h);
    alive_neighbors += get_cell(x + 1, y,     w, h);
    alive_neighbors += get_cell(x - 1, y + 1, w, h);
    alive_neighbors += get_cell(x,     y + 1, w, h);
    alive_neighbors += get_cell(x + 1, y + 1, w, h);

    let idx = u32(y * w + x);
    let current_state = CellBufferPrev[idx];

    // ライフゲームのルール適用
    if (current_state == 1u) {
        if (alive_neighbors == 2u || alive_neighbors == 3u) {
            CellBuffer[idx] = 1u;
        } else {
            CellBuffer[idx] = 0u;
        }
    } else {
        if (alive_neighbors == 3u) {
            CellBuffer[idx] = 1u;
        } else {
            CellBuffer[idx] = 0u;
        }
    }
}
`,

    "render_cells": `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vertex_idx], 0.0, 1.0);
    out.uv = pos[vertex_idx] * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y; 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let w = GlobalUniforms.gridWidth;
    let h = GlobalUniforms.gridHeight;
    
    let max_x = u32(w) - 1u;
    let max_y = u32(h) - 1u;
    let x = min(u32(in.uv.x * w), max_x);
    let y = min(u32(in.uv.y * h), max_y);
    
    let idx = y * u32(w) + x;
    let cell_state = CellBuffer[idx];

    if (cell_state == 1u) {
        return vec4<f32>(0.0, 0.8, 0.2, 1.0); 
    }
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
}
`
};

// wgsl_spheres.ts 等に分離して保存することをおすすめします
export const sphereWgslCodes: Record<string, string> = {
    // 🌟 追加: 初期化 (簡易乱数で空中にばらまく)
    "init_particles": `
fn hash(n: u32) -> f32 {
    var x = n;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return f32(x) / 4294967295.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    
    // -10.0 ~ 10.0 の範囲の空中にランダム配置
    let px = (hash(idx * 3u + 0u) - 0.5) * 20.0;
    let py = (hash(idx * 3u + 1u) - 0.5) * 20.0 + 10.0; // 上の方に配置
    let pz = (hash(idx * 3u + 2u) - 0.5) * 20.0;
    
    // ランダムな初速
    let vx = (hash(idx * 3u + 10u) - 0.5) * 0.2;
    let vy = (hash(idx * 3u + 11u) - 0.5) * 0.2;
    let vz = (hash(idx * 3u + 12u) - 0.5) * 0.2;

    posOut[idx] = vec4<f32>(px, py, pz, 1.0);
    velOut[idx] = vec4<f32>(vx, vy, vz, 0.0);
}
`,

    // 🌟 追加: 毎フレームの物理計算 (重力 + 床でのバウンド)
    "update_particles": `
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;

    var p = posIn[idx].xyz;
    var v = velIn[idx].xyz;

    // 1. 重力を適用
    v.y -= 0.005;
    
    // 2. 中心へ少し集まる力を加えて、画面外に飛び散るのを防ぐ
    v -= p * 0.0001;

    // 3. 速度を位置に加算
    p += v;

    // 4. 床 (Y = -10.0) でのバウンド処理
    if (p.y < -10.0) {
        p.y = -10.0;
        v.y = -v.y * 0.8; // 反発係数 0.8 で跳ね返る
    }

    posOut[idx] = vec4<f32>(p, 1.0);
    velOut[idx] = vec4<f32>(v, 0.0);
}
`,
    "matcap_spheres": `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    let offset = v_idx * 6u;
    
    let local_pos = vec3<f32>(
        baseSphere[offset],
        baseSphere[offset + 1u],
        baseSphere[offset + 2u]
    );
    
    let local_normal = vec3<f32>(
        baseSphere[offset + 3u],
        baseSphere[offset + 4u],
        baseSphere[offset + 5u]
    );
    
    let center = particlePos[i_idx].xyz;
    let radius = 0.5; 
    
    let world_pos = (local_pos * radius) + center;

    var out: VertexOutput;
    out.position = camera.viewProjection * vec4<f32>(world_pos, 1.0);
    
    let view_n = camera.view * vec4<f32>(local_normal, 0.0);
    out.view_normal = normalize(view_n.xyz);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.view_normal);
    
    let matcap_uv = n.xy * 0.48 + 0.5;
    let final_uv = vec2<f32>(matcap_uv.x, 1.0 - matcap_uv.y);
    
    return textureSample(matcapTex, matcapSampler, final_uv);
}
`
};



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
    await engine.compilePipelines(sphereWgslCodes);

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
        // ※ ここで毎フレームカメラを動かす場合は updateVariables を呼ぶ
        
        engine.step();
        
        animationId = requestAnimationFrame(frame);
        (window as any).currentAnimationId = animationId;
    }
    
    frame();
}