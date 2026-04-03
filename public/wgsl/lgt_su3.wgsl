// ============================================================================
// 2D SU(3) Lattice Gauge Theory
// ============================================================================

const L: u32 = 32u;
const L_squared: u32 = L * L;
const WORKGROUP_SIZE: u32 = 8u;

const PI: f32 = 3.1415926535;
const TWO_PI: f32 = 2.0 * PI;

// --- SU(3) Matrix Representation ---
// 3x3 Complex Matrix. Each element is vec2<f32>(real, imag).
// To ensure strict 16-byte alignment in WebGPU storage buffers,
// we add a 'pad' field to make the struct exactly 80 bytes (20 floats).
struct SU3Mat {
    r0c0: vec2<f32>, r0c1: vec2<f32>, r0c2: vec2<f32>,
    r1c0: vec2<f32>, r1c1: vec2<f32>, r1c2: vec2<f32>,
    r2c0: vec2<f32>, r2c1: vec2<f32>, r2c2: vec2<f32>,
    pad: vec2<f32> // Padding to 80 bytes (multiple of 16)
};

// --- Complex Math Helpers ---
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2(a.x, -a.y);
}

// --- SU(3) Math Helpers ---
fn su3_identity() -> SU3Mat {
    return SU3Mat(
        vec2(1.0, 0.0), vec2(0.0, 0.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(0.0, 0.0), vec2(1.0, 0.0),
        vec2(0.0, 0.0)
    );
}

// SU(3) Matrix Multiplication: A * B
fn su3_mult(a: SU3Mat, b: SU3Mat) -> SU3Mat {
    var m: SU3Mat;
    m.r0c0 = cmul(a.r0c0, b.r0c0) + cmul(a.r0c1, b.r1c0) + cmul(a.r0c2, b.r2c0);
    m.r0c1 = cmul(a.r0c0, b.r0c1) + cmul(a.r0c1, b.r1c1) + cmul(a.r0c2, b.r2c1);
    m.r0c2 = cmul(a.r0c0, b.r0c2) + cmul(a.r0c1, b.r1c2) + cmul(a.r0c2, b.r2c2);

    m.r1c0 = cmul(a.r1c0, b.r0c0) + cmul(a.r1c1, b.r1c0) + cmul(a.r1c2, b.r2c0);
    m.r1c1 = cmul(a.r1c0, b.r0c1) + cmul(a.r1c1, b.r1c1) + cmul(a.r1c2, b.r2c1);
    m.r1c2 = cmul(a.r1c0, b.r0c2) + cmul(a.r1c1, b.r1c2) + cmul(a.r1c2, b.r2c2);

    m.r2c0 = cmul(a.r2c0, b.r0c0) + cmul(a.r2c1, b.r1c0) + cmul(a.r2c2, b.r2c0);
    m.r2c1 = cmul(a.r2c0, b.r0c1) + cmul(a.r2c1, b.r1c1) + cmul(a.r2c2, b.r2c1);
    m.r2c2 = cmul(a.r2c0, b.r0c2) + cmul(a.r2c1, b.r1c2) + cmul(a.r2c2, b.r2c2);
    m.pad = vec2(0.0, 0.0);
    return m;
}

// SU(3) Inverse (Conjugate Transpose / Dagger)
fn su3_inv(a: SU3Mat) -> SU3Mat {
    var m: SU3Mat;
    m.r0c0 = cconj(a.r0c0); m.r0c1 = cconj(a.r1c0); m.r0c2 = cconj(a.r2c0);
    m.r1c0 = cconj(a.r0c1); m.r1c1 = cconj(a.r1c1); m.r1c2 = cconj(a.r2c1);
    m.r2c0 = cconj(a.r0c2); m.r2c1 = cconj(a.r1c2); m.r2c2 = cconj(a.r2c2);
    m.pad = vec2(0.0, 0.0);
    return m;
}

// Real part of the Trace of SU(3) matrix
fn su3_trace_real(a: SU3Mat) -> f32 {
    return a.r0c0.x + a.r1c1.x + a.r2c2.x;
}

// Re-unitarization (Gram-Schmidt Orthogonalization)
// Computes u' = u/|u|, v' = v - (u' \cdot v)u', v'' = v'/|v'|, w'' = (u' x v'')*
fn su3_reunitarize(m: SU3Mat) -> SU3Mat {
    // 1. Normalize row 0
    let norm0 = sqrt(m.r0c0.x*m.r0c0.x + m.r0c0.y*m.r0c0.y +
                     m.r0c1.x*m.r0c1.x + m.r0c1.y*m.r0c1.y +
                     m.r0c2.x*m.r0c2.x + m.r0c2.y*m.r0c2.y);
    let u0 = vec2(m.r0c0.x / norm0, m.r0c0.y / norm0);
    let u1 = vec2(m.r0c1.x / norm0, m.r0c1.y / norm0);
    let u2 = vec2(m.r0c2.x / norm0, m.r0c2.y / norm0);

    // 2. Project row 1 onto row 0: dot = <u, v> = sum(u_i^* v_i)
    let dot01 = vec2(
        u0.x*m.r1c0.x + u0.y*m.r1c0.y + u1.x*m.r1c1.x + u1.y*m.r1c1.y + u2.x*m.r1c2.x + u2.y*m.r1c2.y,
        u0.x*m.r1c0.y - u0.y*m.r1c0.x + u1.x*m.r1c1.y - u1.y*m.r1c1.x + u2.x*m.r1c2.y - u2.y*m.r1c2.x
    );

    // 3. Subtract projection from row 1
    var v0 = vec2(m.r1c0.x - (dot01.x*u0.x - dot01.y*u0.y), m.r1c0.y - (dot01.x*u0.y + dot01.y*u0.x));
    var v1 = vec2(m.r1c1.x - (dot01.x*u1.x - dot01.y*u1.y), m.r1c1.y - (dot01.x*u1.y + dot01.y*u1.x));
    var v2 = vec2(m.r1c2.x - (dot01.x*u2.x - dot01.y*u2.y), m.r1c2.y - (dot01.x*u2.y + dot01.y*u2.x));

    // 4. Normalize row 1
    let norm1 = sqrt(v0.x*v0.x + v0.y*v0.y + v1.x*v1.x + v1.y*v1.y + v2.x*v2.x + v2.y*v2.y);
    v0 = vec2(v0.x / norm1, v0.y / norm1);
    v1 = vec2(v1.x / norm1, v1.y / norm1);
    v2 = vec2(v2.x / norm1, v2.y / norm1);

    // 5. Row 2 is the conjugate of the cross product of row 0 and row 1 (ensures det = 1)
    let w0 = vec2(u1.x*v2.x - u1.y*v2.y - u2.x*v1.x + u2.y*v1.y, -(u1.x*v2.y + u1.y*v2.x - u2.x*v1.y - u2.y*v1.x));
    let w1 = vec2(u2.x*v0.x - u2.y*v0.y - u0.x*v2.x + u0.y*v2.y, -(u2.x*v0.y + u2.y*v0.x - u0.x*v2.y - u0.y*v2.x));
    let w2 = vec2(u0.x*v1.x - u0.y*v1.y - u1.x*v0.x + u1.y*v0.y, -(u0.x*v1.y + u0.y*v1.x - u1.x*v0.y - u1.y*v0.x));

    return SU3Mat(u0, u1, u2, v0, v1, v2, w0, w1, w2, vec2(0.0));
}

// --- Random Number Generation (Corrected PCG) ---
fn random_u32(state: ptr<function, u32>) -> u32 {
    let old_state = *state;
    *state = old_state * 747796405u + 2891336453u;
    var word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    word = (word >> 22u) ^ word;
    return word;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
    return f32(random_u32(state)) / 4294967295.0;
}

// --- SU(2) Kennedy-Pendleton Heat Bath Helpers ---

fn sample_su2_x0(alpha: f32, seed: ptr<function, u32>) -> f32 {
    if (alpha < 0.0001) { return 2.0 * random_f32(seed) - 1.0; }
    var x0: f32 = 0.0;
    for (var i = 0u; i < 1000u; i++) {
        let r1 = random_f32(seed) + 1e-7;
        let r2 = random_f32(seed) + 1e-7;
        let r3 = random_f32(seed);
        let r4 = random_f32(seed);
        let x = -log(r1) / alpha;
        let y = -log(r2) / alpha;
        let z = cos(2.0 * PI * r3);
        let lambda = x + y * z * z;
        x0 = 1.0 - lambda;
        if (x0 < -1.0) { continue; }
        if (r4 * r4 <= 0.5 * (1.0 + x0)) { break; }
    }
    return x0;
}

fn generate_su2_heatbath(alpha: f32, seed: ptr<function, u32>) -> vec4<f32> {
    let x0 = sample_su2_x0(alpha, seed);
    let rho = sqrt(max(0.0, 1.0 - x0 * x0));
    let r5 = random_f32(seed);
    let r6 = random_f32(seed);
    let cos_theta = 2.0 * r5 - 1.0;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * r6;
    let x1 = rho * sin_theta * cos(phi);
    let x2 = rho * sin_theta * sin(phi);
    let x3 = rho * cos_theta;
    return vec4(x0, x1, x2, x3);
}

fn su2_mult(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

fn embed_su2(a: vec4<f32>, idx: u32) -> SU3Mat {
    var m = su3_identity();
    let u00 = vec2(a.x, a.w);  let u01 = vec2(a.z, a.y);
    let u10 = vec2(-a.z, a.y); let u11 = vec2(a.x, -a.w);
    if (idx == 0u) {
        m.r0c0 = u00; m.r0c1 = u01; m.r1c0 = u10; m.r1c1 = u11;
    } else if (idx == 1u) {
        m.r1c1 = u00; m.r1c2 = u01; m.r2c1 = u10; m.r2c2 = u11;
    } else {
        m.r0c0 = u00; m.r0c2 = u01; m.r2c0 = u10; m.r2c2 = u11;
    }
    return m;
}

// --- Cabibbo-Marinari Pseudo-Heat Bath ---
fn cabibbo_marinari_step(U_in: SU3Mat, V: SU3Mat, beta: f32, seed: ptr<function, u32>) -> SU3Mat {
    var U = U_in;
    var W = su3_mult(U, V); // 現在の局所作用行列 W = U * V

    for (var idx = 0u; idx < 3u; idx++) {
        var a0: f32; var a1: f32; var a2: f32; var a3: f32;
        
        // W から 2x2 部分群の成分を抽出
        if (idx == 0u) {
            a0 = W.r0c0.x + W.r1c1.x;
            a1 = -(W.r1c0.y + W.r0c1.y);
            a2 = W.r1c0.x - W.r0c1.x;
            a3 = -(W.r0c0.y - W.r1c1.y);
        } else if (idx == 1u) {
            a0 = W.r1c1.x + W.r2c2.x;
            a1 = -(W.r2c1.y + W.r1c2.y);
            a2 = W.r2c1.x - W.r1c2.x;
            a3 = -(W.r1c1.y - W.r2c2.y);
        } else {
            a0 = W.r0c0.x + W.r2c2.x;
            a1 = -(W.r2c0.y + W.r0c2.y);
            a2 = W.r2c0.x - W.r0c2.x;
            a3 = -(W.r0c0.y - W.r2c2.y);
        }

        let k = sqrt(a0*a0 + a1*a1 + a2*a2 + a3*a3);
        var r_su2: vec4<f32>;

        if (k > 1e-6) {
            let a_hat = vec4(a0/k, a1/k, a2/k, a3/k);
            let alpha = (beta * k) / 3.0; // SU(3)の作用に基づくスケーリング
            let x_kp = generate_su2_heatbath(alpha, seed);
            r_su2 = su2_mult(x_kp, a_hat);
        } else {
            r_su2 = generate_su2_heatbath(0.0, seed);
        }

        let R = embed_su2(r_su2, idx);
        U = su3_mult(R, U);
        W = su3_mult(R, W); // 次の部分群のために W も更新しておく
    }

    return U;
}

// --- Uniforms & Storage Buffers ---
struct SimParams {
    beta: f32,
    update_subset: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> links: array<SU3Mat>;
@group(0) @binding(2) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(3) var<storage, read_write> viz_results: array<f32>;

fn get_link_idx(x: u32, y: u32, dir: u32) -> u32 {
    return (y * L + x) * 2u + dir;
}

// --- Kernels ---

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn init_cold(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }

    let site_idx = y * L + x;
    let ident = su3_identity();

    links[get_link_idx(x, y, 0u)] = ident;
    links[get_link_idx(x, y, 1u)] = ident;

    var rand_seed = rng_state[site_idx];
    let dummy = random_u32(&rand_seed); 
    rng_state[site_idx] = rand_seed;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn metropolis_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }

    if ((x + y) % 2u != params.update_subset % 2u) { return; }
    
    let dir_to_update = params.update_subset / 2u;
    let site_idx = y * L + x;
    var rand_seed = rng_state[site_idx];

    let xp1 = (x + 1u) % L; let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L; let ym1 = (y + L - 1u) % L;

    // --- Calculate Staple (V) ---
    var staple: SU3Mat;
    if (dir_to_update == 0u) { // x-link
        let staple_up = su3_mult(links[get_link_idx(xp1, y, 1u)], su3_mult(su3_inv(links[get_link_idx(x, yp1, 0u)]), su3_inv(links[get_link_idx(x, y, 1u)])));
        let staple_down = su3_mult(su3_inv(links[get_link_idx(xp1, ym1, 1u)]), su3_mult(su3_inv(links[get_link_idx(x, ym1, 0u)]), links[get_link_idx(x, ym1, 1u)]));
        // Add matrices
        staple.r0c0 = staple_up.r0c0 + staple_down.r0c0; staple.r0c1 = staple_up.r0c1 + staple_down.r0c1; staple.r0c2 = staple_up.r0c2 + staple_down.r0c2;
        staple.r1c0 = staple_up.r1c0 + staple_down.r1c0; staple.r1c1 = staple_up.r1c1 + staple_down.r1c1; staple.r1c2 = staple_up.r1c2 + staple_down.r1c2;
        staple.r2c0 = staple_up.r2c0 + staple_down.r2c0; staple.r2c1 = staple_up.r2c1 + staple_down.r2c1; staple.r2c2 = staple_up.r2c2 + staple_down.r2c2;
    } else { // y-link
        let staple_right = su3_mult(links[get_link_idx(x, yp1, 0u)], su3_mult(su3_inv(links[get_link_idx(xp1, y, 1u)]), su3_inv(links[get_link_idx(x, y, 0u)])));
        let staple_left = su3_mult(su3_inv(links[get_link_idx(xm1, yp1, 0u)]), su3_mult(su3_inv(links[get_link_idx(xm1, y, 1u)]), links[get_link_idx(xm1, y, 0u)]));
        staple.r0c0 = staple_right.r0c0 + staple_left.r0c0; staple.r0c1 = staple_right.r0c1 + staple_left.r0c1; staple.r0c2 = staple_right.r0c2 + staple_left.r0c2;
        staple.r1c0 = staple_right.r1c0 + staple_left.r1c0; staple.r1c1 = staple_right.r1c1 + staple_left.r1c1; staple.r1c2 = staple_right.r1c2 + staple_left.r1c2;
        staple.r2c0 = staple_right.r2c0 + staple_left.r2c0; staple.r2c1 = staple_right.r2c1 + staple_left.r2c1; staple.r2c2 = staple_right.r2c2 + staple_left.r2c2;
    }

    let link_idx = get_link_idx(x, y, dir_to_update);
    let old_link = links[link_idx];

    // Cabibbo-Marinari ヒートバス更新
    let new_link = cabibbo_marinari_step(old_link, staple, params.beta, &rand_seed);

    // 浮動小数点誤差の蓄積を防ぐため、厳密な再ユニタリ化を行ってから保存
    links[link_idx] = su3_reunitarize(new_link);
    rng_state[site_idx] = rand_seed;
}


@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn measure_plaquette(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }

    let site_idx = y * L + x;
    let xp1 = (x + 1u) % L;
    let yp1 = (y + 1u) % L;

    let U1 = links[get_link_idx(x, y, 0u)];
    let U2 = links[get_link_idx(xp1, y, 1u)];
    let U3_inv = su3_inv(links[get_link_idx(x, yp1, 0u)]);
    let U4_inv = su3_inv(links[get_link_idx(x, y, 1u)]);

    let plaquette_matrix = su3_mult(U1, su3_mult(U2, su3_mult(U3_inv, U4_inv)));

    // Plaquette observable is normalized by N=3: 1/3 * Re(Tr(U_p))
    viz_results[site_idx] = (1.0 / 3.0) * su3_trace_real(plaquette_matrix);
}