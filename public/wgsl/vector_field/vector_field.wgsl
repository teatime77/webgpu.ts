// @shader: compute_vector_field

// 空間上のベクトル場を計算する関数（例：回転する磁場のような動き）
fn calc_vector(x: f32, z: f32, t: f32) -> vec3<f32> {
    let dx = sin(z * 3.0 + t) * 0.5;
    let dy = cos(x * 3.0 + t) * 0.5;
    let dz = sin(x * 2.0 - t) * 0.5;
    return vec3<f32>(dx, dy, dz);
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
    let z = v * 2.0 - 1.0;
    
    let vec_val = calc_vector(x, z, Params.time) * 0.2;

    // バッファへの書き込み (xyzにベクトル、wはパディングとして1.0)
    let index = iy * gw + ix;
    VectorFieldData[index] = vec4<f32>(vec_val, 1.0);
}

// @shader: render_vector_field

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

// 任意の方向ベクトルから回転行列（基底ベクトル）を作成する関数
fn get_rotation_matrix(dir: vec3<f32>) -> mat3x3<f32> {
    let up = normalize(dir);
    var right = cross(vec3<f32>(0.0, 1.0, 0.0), up);
    
    // ベクトルが真上や真下を向いている場合の特例処理
    if (length(right) < 0.001) {
        right = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        right = normalize(right);
    }
    
    let forward = cross(right, up);
    return mat3x3<f32>(right, up, forward);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    
    // 1. ベースとなる矢印の頂点データを読み込む (x, y, z, nx, ny, nz)
    let base_idx = vertex_index * 6u;
    var local_pos = vec3<f32>(BaseArrow[base_idx], BaseArrow[base_idx + 1u], BaseArrow[base_idx + 2u]);
    var local_norm = vec3<f32>(BaseArrow[base_idx + 3u], BaseArrow[base_idx + 4u], BaseArrow[base_idx + 5u]);

    // 2. インスタンス（グリッド位置）のベクトルデータを読み込む
    let vec_val = VectorFieldData[instance_index].xyz;
    let vec_len = length(vec_val);
    
    // 3. ベクトルの大きさに応じて矢印を可変長に変形する
    // local_pos.y <= 1.0 が軸（円柱）、> 1.0 が傘（円錐）
    if (local_pos.y <= 1.0) {
        local_pos.y = local_pos.y * vec_len;
    } else {
        local_pos.y = vec_len + (local_pos.y - 1.0);
    }

    // 4. ベクトルの向きに合わせて回転させる
    if (vec_len > 0.0001) {
        let rot_mat = get_rotation_matrix(vec_val);
        local_pos = rot_mat * local_pos;
        local_norm = rot_mat * local_norm;
    }

    // 5. グリッド上の正しい位置に平行移動させる
    let gw = u32(Params.gridWidth);
    let ix = instance_index % gw;
    let iy = instance_index / gw;
    let u = f32(ix) / (Params.gridWidth - 1.0);
    let v = f32(iy) / (Params.gridHeight - 1.0);
    let grid_x = u * 2.0 - 1.0;
    let grid_z = v * 2.0 - 1.0;
    
    let world_pos = local_pos + vec3<f32>(grid_x, 0.0, grid_z);

    var out: VertexOutput;
    out.position = Camera.viewProjection * vec4<f32>(world_pos, 1.0);
    out.normal = local_norm;
    
    // ベクトルの向きを色にする（可視化用）
    out.color = normalize(vec_val) * 0.5 + 0.5;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 簡易的なライティング
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.0));
    let n = normalize(in.normal);
    let diff = max(dot(n, light_dir), 0.2); // アンビエント0.2
    
    let final_color = in.color * diff;
    return vec4<f32>(final_color, 1.0);
}