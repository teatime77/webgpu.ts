// @shader: compute_surface

// 擬似的な波の曲面を計算する関数
fn calc_height(x: f32, y: f32, t: f32) -> f32 {
    let d = sqrt(x*x + y*y);
    return sin(d * 10.0 - t * 5.0) * 0.2;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ix = global_id.x;
    let iy = global_id.y;
    
    let gw = u32(Params.gridWidth);
    let gh = u32(Params.gridHeight);

    if (ix >= gw || iy >= gh) {
        return;
    }

    // グリッド座標を -1.0 ~ 1.0 に正規化
    let u = f32(ix) / (Params.gridWidth - 1.0);
    let v = f32(iy) / (Params.gridHeight - 1.0);
    let x = u * 2.0 - 1.0;
    let z = v * 2.0 - 1.0; // Z軸を奥行きに
    
    let y = calc_height(x, z, Params.time); // Y軸を高さに
    let pos = vec3<f32>(x, y, z);

    // 簡易的な法線計算（隣接ピクセルとの差分）
    let eps = 0.01;
    let dydx = calc_height(x + eps, z, Params.time) - calc_height(x - eps, z, Params.time);
    let dydz = calc_height(x, z + eps, Params.time) - calc_height(x, z - eps, Params.time);
    let normal = normalize(vec3<f32>(-dydx, eps * 2.0, -dydz));

    // バッファへの書き込み (pos と normal を交互に格納)
    let index = (iy * gw + ix) * 2u;
    SurfaceData[index] = vec4<f32>(pos, 1.0);
    SurfaceData[index + 1u] = vec4<f32>(normal, 0.0);
}

// @shader: render_surface

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let gw = u32(Params.gridWidth);
    let quads_x = gw - 1u;
    let quad_id = vertex_index / 6u; // 1つの四角形(Quad)は6頂点
    let qx = quad_id % quads_x;
    let qy = quad_id / quads_x;

    // 四角形を構成する2つの三角形の頂点オフセット
    var offsets = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), // 1つ目の三角形
        vec2<u32>(0u, 1u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u)  // 2つ目の三角形
    );
    let offset = offsets[vertex_index % 6u];
    
    let vx = qx + offset.x;
    let vy = qy + offset.y;

    // コンピュートシェーダで書き込んだデータを読み出し
    let index = (vy * gw + vx) * 2u;
    let pos = SurfaceData[index].xyz;
    let normal = SurfaceData[index + 1u].xyz;

    var out: VertexOutput;
    out.position = Camera.viewProjection * vec4<f32>(pos, 1.0);
    out.normal = normal;
    out.world_pos = pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 簡易的なライティング（法線を色として表示）
    let n = normalize(in.normal);
    let color = n * 0.5 + 0.5;
    return vec4<f32>(color, 1.0);
}