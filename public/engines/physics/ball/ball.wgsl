// @shader: init_particles
// 初期化 (簡易乱数で空中にばらまく)
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


// @shader: update_particles
// 毎フレームの物理計算 (重力 + 床でのバウンド)
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

// @shader: matcap_spheres
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    let offset = v_idx * 6u;
    
    let local_pos = vec3<f32>(
        BaseSphere[offset],
        BaseSphere[offset + 1u],
        BaseSphere[offset + 2u]
    );
    
    let local_normal = vec3<f32>(
        BaseSphere[offset + 3u],
        BaseSphere[offset + 4u],
        BaseSphere[offset + 5u]
    );
    
    let center = ParticlePos[i_idx].xyz;
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
    
    return textureSample(MatCapTex, MatCapSampler, final_uv);
}
