struct Particle {
  meshPos : vec4<f32>,
  meshVec : vec4<f32>,
  color   : vec4<f32>,
}

struct Env {
    time   : f32,
    tick   : f32,
    filler1: f32,
    filler2: f32,
}
struct Uniforms {
    env: Env,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particlesA: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB: array<Particle>;

const PI = 3.14159265359;
const Q_RANGE = 2.0 * PI;
const P_RANGE = 5.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // These must match the grid dimensions in the TS file
    let grid_dim_q = 32u;
    let grid_dim_p = 32u;

    if (global_id.x >= grid_dim_q || global_id.y >= grid_dim_p) {
        return;
    }

    let idx = global_id.y * grid_dim_q + global_id.x;

    // Only compute on the first frame; the vector field is static.
    if (u.env.tick > 0.0) {
        particlesB[idx] = particlesA[idx];
        return;
    }

    // 1. Map grid indices to (q, p) coordinates in phase space
    let u = f32(global_id.x) / f32(grid_dim_q - 1u);
    let v = f32(global_id.y) / f32(grid_dim_p - 1u);

    let q = -Q_RANGE / 2.0 + u * Q_RANGE;
    let p = -P_RANGE / 2.0 + v * P_RANGE;

    // 2. Calculate the Hamiltonian vector field X_H = (dH/dp, -dH/dq)
    let dq_dt = p;
    let dp_dt = -sin(q);
    let vector = vec3<f32>(dq_dt, dp_dt, 0.0);

    // 3. Store arrow position, vector, and color for the instanced renderer
    particlesB[idx].meshPos = vec4<f32>(q, p, 0.0, 1.0);

    let vec_len = length(vector);

    // Color based on magnitude (blue for slow, red for fast)
    // Max length is around sqrt(2.5^2 + 1^2) ~ 2.7
    let max_len = 2.7;
    let norm_len = clamp(vec_len / max_len, 0.0, 1.0);
    // Blue (0,0,1) to Red (1,0,0) via Purple
    particlesB[idx].color = vec4<f32>(norm_len, 0.0, 1.0 - norm_len, 1.0);

    // Normalize and scale the vector for good visibility
    if (vec_len > 0.001) {
        particlesB[idx].meshVec = vec4<f32>(vector / vec_len * 0.3, 0.0);
    } else {
        particlesB[idx].meshVec = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}