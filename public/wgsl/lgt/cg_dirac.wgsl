// ============================================================================
// WebGPU CG Solver for U(1) Wilson-Dirac Operator
// ============================================================================
const L: u32 = 128u;

// 複素スピノル: x=上スピン実部, y=上虚部, z=下実部, w=下虚部
alias Spinor = vec4<f32>;

@group(0) @binding(0) var<storage, read_write> x: array<Spinor>;
@group(0) @binding(1) var<storage, read>       b: array<Spinor>;
@group(0) @binding(2) var<storage, read_write> p: array<Spinor>;
@group(0) @binding(3) var<storage, read_write> r: array<Spinor>;
@group(0) @binding(4) var<storage, read_write> q: array<Spinor>;
@group(0) @binding(5) var<storage, read_write> scalars: array<f32>; 
@group(0) @binding(6) var<storage, read>       links: array<f32>;   // 背景ゲージ場
@group(0) @binding(7) var<uniform>             params: f32;         // 質量 (mass)
@group(0) @binding(8) var<storage, read_write> tmp: array<Spinor>;  // D^dagger計算用の一時バッファ

// --- 複素数・ガンマ行列のヘルパー ---
fn c_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ゲージ場 U = exp(i*theta) をスピノルにかける
fn apply_U(theta: f32, psi: Spinor) -> Spinor {
    let u = vec2<f32>(cos(theta), sin(theta));
    let s1 = c_mult(u, vec2<f32>(psi.x, psi.y));
    let s2 = c_mult(u, vec2<f32>(psi.z, psi.w));
    return Spinor(s1.x, s1.y, s2.x, s2.y);
}

// ガンマ行列の掛け算
fn gamma1_mul(psi: Spinor) -> Spinor { return Spinor(psi.z, psi.w, psi.x, psi.y); }
fn gamma2_mul(psi: Spinor) -> Spinor { return Spinor(psi.w, -psi.z, -psi.y, psi.x); }

// --- Kernel 1: D を掛ける (tmp = D * p) ---
@compute @workgroup_size(64)
fn apply_D(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }

    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;
    let xm1 = (cx + L - 1u) % L; let ym1 = (cy + L - 1u) % L;

    var out = p[idx] * (params + 2.0); // 質量項

    // 右 (1 - gamma_1)
    let psi_R = apply_U(links[(cy * L + cx) * 2u + 0u], p[cy * L + xp1]);
    out -= 0.5 * (psi_R - gamma1_mul(psi_R));

    // 左 (1 + gamma_1)
    let psi_L = apply_U(-links[(cy * L + xm1) * 2u + 0u], p[cy * L + xm1]);
    out -= 0.5 * (psi_L + gamma1_mul(psi_L));

    // 上 (1 - gamma_2)
    let psi_U = apply_U(links[(cy * L + cx) * 2u + 1u], p[yp1 * L + cx]);
    out -= 0.5 * (psi_U - gamma2_mul(psi_U));

    // 下 (1 + gamma_2)
    let psi_D = apply_U(-links[(ym1 * L + cx) * 2u + 1u], p[ym1 * L + cx]);
    out -= 0.5 * (psi_D + gamma2_mul(psi_D));

    tmp[idx] = out;
}

// --- Kernel 2: D^\dagger を掛ける (q = D^\dagger * tmp) ---
@compute @workgroup_size(64)
fn apply_D_dag(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }

    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;
    let xm1 = (cx + L - 1u) % L; let ym1 = (cy + L - 1u) % L;

    var out = tmp[idx] * (params + 2.0);

    // エルミート共役をとるため、ガンマ行列の符号が反転します！
    let psi_R = apply_U(links[(cy * L + cx) * 2u + 0u], tmp[cy * L + xp1]);
    out -= 0.5 * (psi_R + gamma1_mul(psi_R)); // (-) -> (+)

    let psi_L = apply_U(-links[(cy * L + xm1) * 2u + 0u], tmp[cy * L + xm1]);
    out -= 0.5 * (psi_L - gamma1_mul(psi_L)); // (+) -> (-)

    let psi_U = apply_U(links[(cy * L + cx) * 2u + 1u], tmp[yp1 * L + cx]);
    out -= 0.5 * (psi_U + gamma2_mul(psi_U)); // (-) -> (+)

    let psi_D = apply_U(-links[(ym1 * L + cx) * 2u + 1u], tmp[ym1 * L + cx]);
    out -= 0.5 * (psi_D - gamma2_mul(psi_D)); // (+) -> (-)

    q[idx] = out;
}

// --- Kernel 3: 内積 ---
@compute @workgroup_size(1)
fn calc_p_q() {
    var sum: f32 = 0.0;
    for(var i = 0u; i < L * L; i++) {
        sum += dot(p[i], q[i]); // vec4 の内積
    }
    scalars[1] = sum;
    scalars[2] = scalars[0] / (sum + 1e-12);
}

// --- Kernel 4: x, r の更新 ---
@compute @workgroup_size(64)
fn update_x_r(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= L * L) { return; }
    let alpha = scalars[2];
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * q[i];
}

// --- Kernel 5: 新しい rho ---
@compute @workgroup_size(1)
fn calc_new_rho() {
    var sum: f32 = 0.0;
    for(var i = 0u; i < L * L; i++) {
        sum += dot(r[i], r[i]);
    }
    scalars[4] = sum;
    scalars[3] = sum / (scalars[0] + 1e-12);
    scalars[0] = sum;
}

// --- Kernel 6: p の更新 ---
@compute @workgroup_size(64)
fn update_p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= L * L) { return; }
    let beta = scalars[3];
    p[i] = r[i] + beta * p[i];
}