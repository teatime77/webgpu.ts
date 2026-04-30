import { SimulationSchema } from './control';

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

export const wgslCodes: Record<string, string> = {

    // ==========================================
    // 1. 初期化シェーダ (init_cells)
    // ==========================================
    "init_cells": `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32(global_uniforms.gridWidth);
    let h = u32(global_uniforms.gridHeight);
    if (id.x >= w || id.y >= h) { return; }

    let idx = id.y * w + id.x;
    
    // シンプルな擬似乱数ハッシュで20%の確率でセルを「生(1)」にする
    let hash = (id.x * 374761393u + id.y * 668265263u) ^ 1013904223u;
    if (hash % 100u < 20u) {
        cell_buffer[idx] = 1u;
    } else {
        cell_buffer[idx] = 0u;
    }
}
`,

    // ==========================================
    // 2. 更新シェーダ (update_cells)
    // ==========================================
    "update_cells": `
fn get_cell(x: i32, y: i32, w: i32, h: i32) -> u32 {
    // トーラス状に世界を繋ぐ (Wrap around)
    let nx = (x + w) % w;
    let ny = (y + h) % h;
    return cell_buffer_prev[u32(ny * w + nx)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = i32(global_uniforms.gridWidth);
    let h = i32(global_uniforms.gridHeight);
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
    let current_state = cell_buffer_prev[idx];

    // ライフゲームのルール適用
    if (current_state == 1u) {
        if (alive_neighbors == 2u || alive_neighbors == 3u) {
            cell_buffer[idx] = 1u; // 生存
        } else {
            cell_buffer[idx] = 0u; // 過疎 or 過密で死亡
        }
    } else {
        if (alive_neighbors == 3u) {
            cell_buffer[idx] = 1u; // 誕生
        } else {
            cell_buffer[idx] = 0u; // 死のまま
        }
    }
}
`,

    // ==========================================
    // 3. 描画シェーダ (render_cells)
    // ==========================================
    "render_cells": `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    // 頂点バッファなしでフルスクリーンクアッドを生成する定石トリック
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vertex_idx], 0.0, 1.0);
    // UV座標を [0.0, 1.0] に変換
    out.uv = pos[vertex_idx] * 0.5 + 0.5;
    // WebGPUはY軸が下向きなので反転
    out.uv.y = 1.0 - out.uv.y; 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let w = global_uniforms.gridWidth;
    let h = global_uniforms.gridHeight;
    
    // UV座標からグリッドのインデックスを計算 (安全のために最大値をクランプ)
    let max_x = u32(w) - 1u;
    let max_y = u32(h) - 1u;
    let x = min(u32(in.uv.x * w), max_x);
    let y = min(u32(in.uv.y * h), max_y);
    
    let idx = y * u32(w) + x;
    let cell_state = cell_buffer[idx];

    // 生(1)なら緑色、死(0)なら黒色
    if (cell_state == 1u) {
        return vec4<f32>(0.0, 0.8, 0.2, 1.0); 
    }
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
}
`
};

import { GraphManager } from './control';

// ホットリロード(HMR)時の二重起動を防ぐためのグローバル変数
let animationId: number | null = null;

export async function initControl() {
    // 既にシミュレーションが動いている場合は、古いループを停止する
    if ((window as any).isSimulationRunning) {
        if ((window as any).currentAnimationId) {
            cancelAnimationFrame((window as any).currentAnimationId);
        }
    }
    (window as any).isSimulationRunning = true;

    // 1. WebGPUデバイスの取得
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();

    // 2. Canvasのセットアップ
    const canvas = document.getElementById('world-webgpu') as HTMLCanvasElement;
    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    
    // 一度Canvasの設定を解除してから再設定する (HMR対策)
    context.unconfigure(); 
    
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'premultiplied' });

    // ========================================================
    // V2エンジンによる物理シミュレーション起動シーケンス
    // ========================================================
    
    const engine = new GraphManager(device);
    engine.setContext(context, format, canvas.width, canvas.height);
    engine.loadSchema(gameOfLifeSchema);
    await engine.compilePipelines(wgslCodes);

    engine.updateVariables({
        gridWidth: 256,
        gridHeight: 256
    });

    // ========================================================
    // メインループ
    // ========================================================
    function frame() {
        engine.step();
        
        // 次のフレームを要求し、IDをグローバルに保存 (キャンセル用)
        animationId = requestAnimationFrame(frame);
        (window as any).currentAnimationId = animationId;
    }
    
    frame();
}

// initControl();