// Lennard-Jones MD: periodic minimum-image convention, symplectic Euler step.

// @shader: init_particles
fn hash_u32(n: u32) -> f32 {
    var x = n;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return f32(x) / 4294967295.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let n = u32(MdParams.particle_count);
    if (idx >= n) { return; }

    let L = MdParams.box_size;

    let rx = (hash_u32(idx * 3u + 1u) - 0.5) * 0.92 * L;
    let ry = (hash_u32(idx * 3u + 2u) - 0.5) * 0.92 * L;
    let rz = (hash_u32(idx * 3u + 3u) - 0.5) * 0.92 * L;

    let vx = (hash_u32(idx * 7u + 11u) - 0.5) * 0.15;
    let vy = (hash_u32(idx * 7u + 12u) - 0.5) * 0.15;
    let vz = (hash_u32(idx * 7u + 13u) - 0.5) * 0.15;

    posOut[idx] = vec4<f32>(rx, ry, rz, 1.0);
    velOut[idx] = vec4<f32>(vx, vy, vz, 0.0);
}


// @shader: md_step
fn min_image(dr: vec3<f32>, L: f32) -> vec3<f32> {
    return dr - L * round(dr / L);
}

fn lj_accel_on_i(pi: vec3<f32>, pj: vec3<f32>, eps: f32, sig: f32, L: f32) -> vec3<f32> {
    var r = min_image(pj - pi, L);
    var r2 = dot(r, r);
    let soft = sig * sig * 0.18 * 0.18;
    r2 = max(r2, soft);
    let inv_r2 = 1.0 / r2;
    let inv_r = sqrt(inv_r2);
    let s = sig * inv_r;
    let s6 = s * s * s * s * s * s;
    let s12 = s6 * s6;
    // F_pair / m with m=1: magnitude along r direction from i toward j
    let mag = 24.0 * eps * inv_r2 * (2.0 * s12 - s6);
    return mag * r;
}

fn wrap_p(p: vec3<f32>, L: f32) -> vec3<f32> {
    return p - L * round(p / L);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = u32(MdParams.particle_count);
    if (i >= n) { return; }

    let dt = MdParams.dt;
    let eps = MdParams.lj_epsilon;
    let sig = MdParams.lj_sigma;
    let L = MdParams.box_size;

    let pi = posIn[i].xyz;
    var acc = vec3<f32>(0.0);
    for (var j = 0u; j < n; j = j + 1u) {
        if (j == i) { continue; }
        let pj = posIn[j].xyz;
        acc += lj_accel_on_i(pi, pj, eps, sig, L);
    }

    let vi = velIn[i].xyz;
    let v_new = vi + dt * acc;
    let p_new = wrap_p(pi + dt * v_new, L);

    posOut[i] = vec4<f32>(p_new, 1.0);
    velOut[i] = vec4<f32>(v_new, 0.0);
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
    let radius = 0.22;

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
