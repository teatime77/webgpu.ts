// --- 空間ハッシュ・物理シミュレーション用の共通定数 ---
const NUM_PARTICLES: u32 = 1024u;
const NUM_CELLS: u32 = 4096u;
const GRID_RES: vec3<u32> = vec3<u32>(16u, 16u, 16u);
const CELL_SIZE: f32 = 2.0;
const MAX_PARTICLES_PER_CELL: u32 = 8u;
const PARTICLE_RADIUS: f32 = 0.5;
const PARTICLE_MASS: f32 = 1.0;

// 空間座標からグリッドインデックスを取得するヘルパー関数
fn get_cell_index(pos: vec3<f32>) -> u32 {
    // 空間座標 (-15.0 ~ 15.0) をグリッド座標 (0.0 ~ 32.0) にシフト
    let shifted_pos = pos + vec3<f32>(16.0, 16.0, 16.0);
    
    let cx = u32(clamp(shifted_pos.x / CELL_SIZE, 0.0, f32(GRID_RES.x - 1u)));
    let cy = u32(clamp(shifted_pos.y / CELL_SIZE, 0.0, f32(GRID_RES.y - 1u)));
    let cz = u32(clamp(shifted_pos.z / CELL_SIZE, 0.0, f32(GRID_RES.z - 1u)));

    return cz * (GRID_RES.x * GRID_RES.y) + cy * GRID_RES.x + cx;
}


// @shader: init_particles
fn hash(n: u32) -> f32 {
    var x = n;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return f32(x) / 4294967295.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    
    // 箱の中 (-10.0 ~ 10.0) にランダム配置
    let px = (hash(idx * 3u + 0u) - 0.5) * 20.0;
    let py = (hash(idx * 3u + 1u) - 0.5) * 20.0 + 10.0;
    let pz = (hash(idx * 3u + 2u) - 0.5) * 20.0;
    
    let vx = (hash(idx * 3u + 10u) - 0.5) * 0.2;
    let vy = (hash(idx * 3u + 11u) - 0.5) * 0.2;
    let vz = (hash(idx * 3u + 12u) - 0.5) * 0.2;

    posOut[idx] = vec4<f32>(px, py, pz, 1.0);
    velOut[idx] = vec4<f32>(vx, vy, vz, 0.0);
}


// @shader: update_particles
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    var p = posIn[idx].xyz;
    var v = velIn[idx].xyz;

    v.y -= 0.005;     // 重力
    v -= p * 0.0001;  // 中心に集まる力
    p += v;

    // XYZの壁でのバウンド処理 (半径を考慮してめり込みを防止)
    let bound = 15.0 - PARTICLE_RADIUS;
    
    if (p.x < -bound) { p.x = -bound; v.x = -v.x * 0.8; }
    if (p.x >  bound) { p.x =  bound; v.x = -v.x * 0.8; }
    
    if (p.y < -10.0 + PARTICLE_RADIUS) { 
        p.y = -10.0 + PARTICLE_RADIUS; 
        v.y = -v.y * 0.8; 
    }
    
    if (p.z < -bound) { p.z = -bound; v.z = -v.z * 0.8; }
    if (p.z >  bound) { p.z =  bound; v.z = -v.z * 0.8; }

    posOut[idx] = vec4<f32>(p, 1.0);
    velOut[idx] = vec4<f32>(v, 0.0);
}


// @shader: clear_grid
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= NUM_CELLS) { return; }
    
    // カウントをゼロクリア
    atomicStore(&gridCounts[index], 0u);
}


// @shader: build_grid
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_index = id.x;
    if (p_index >= NUM_PARTICLES) { return; }

    let pos = posBuffer[p_index].xyz;
    let cell_index = get_cell_index(pos);

    // アトミックにカウントを加算して格納先インデックスを取得
    let count = atomicAdd(&gridCounts[cell_index], 1u);

    if (count < MAX_PARTICLES_PER_CELL) {
        let storage_index = cell_index * MAX_PARTICLES_PER_CELL + count;
        gridCells[storage_index] = p_index;
    }
}


// @shader: resolve_collisions
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_index = id.x;
    if (p_index >= NUM_PARTICLES) { return; }

    var p_pos = posBuffer[p_index].xyz;
    var p_vel = velBuffer[p_index].xyz;

    let shifted_pos = p_pos + vec3<f32>(16.0, 16.0, 16.0);
    let cell_coord = vec3<i32>(
        i32(shifted_pos.x / CELL_SIZE),
        i32(shifted_pos.y / CELL_SIZE),
        i32(shifted_pos.z / CELL_SIZE)
    );

    var pos_correction = vec3<f32>(0.0);
    var vel_correction = vec3<f32>(0.0);

    // 周囲27セルを探索
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_coord = cell_coord + vec3<i32>(dx, dy, dz);
                
                if (neighbor_coord.x < 0 || neighbor_coord.x >= i32(GRID_RES.x) ||
                    neighbor_coord.y < 0 || neighbor_coord.y >= i32(GRID_RES.y) ||
                    neighbor_coord.z < 0 || neighbor_coord.z >= i32(GRID_RES.z)) {
                    continue;
                }

                let neighbor_cell_index = u32(neighbor_coord.z) * (GRID_RES.x * GRID_RES.y) + u32(neighbor_coord.y) * GRID_RES.x + u32(neighbor_coord.x);
                let count = min(atomicLoad(&gridCounts[neighbor_cell_index]), MAX_PARTICLES_PER_CELL);

                for (var i = 0u; i < count; i++) {
                    let other_p_index = gridCells[neighbor_cell_index * MAX_PARTICLES_PER_CELL + i];
                    
                    if (p_index == other_p_index) { continue; }

                    let other_pos = posBuffer[other_p_index].xyz;
                    let other_vel = velBuffer[other_p_index].xyz;

                    let diff = p_pos - other_pos;
                    let dist = length(diff);
                    let min_dist = PARTICLE_RADIUS * 2.0;

                    if (dist > 0.0 && dist < min_dist) {
                        let normal = diff / dist;
                        let penetration = min_dist - dist;
                        
                        // --- 1. 位置補正 (めり込み解決) ---
                        pos_correction += normal * (penetration * 0.5); // 質量が同じなので0.5

                        // --- 2. 速度補正 (完全弾性衝突 e=1.0) ---
                        let v_rel = p_vel - other_vel;
                        let v_n = dot(v_rel, normal);

                        if (v_n < 0.0) {
                            let e = 1.0;
                            let j_impulse = -(1.0 + e) * v_n / ((1.0 / PARTICLE_MASS) + (1.0 / PARTICLE_MASS));
                            vel_correction += (j_impulse / PARTICLE_MASS) * normal;
                        }
                    }
                }
            }
        }
    }

    // 更新 (インプレース)
    posBuffer[p_index] = vec4<f32>(p_pos + pos_correction, 1.0);
    velBuffer[p_index] = vec4<f32>(p_vel + vel_correction, 0.0);
}


// @shader: matcap_spheres
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
    let radius = PARTICLE_RADIUS; 
    
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