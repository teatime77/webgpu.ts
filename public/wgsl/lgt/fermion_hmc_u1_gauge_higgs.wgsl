// ============================================================================
// WebGPU Dynamical Fermions (HMC + CG Solver)
// ============================================================================
const PI: f32 = 3.1415926535;
const L: u32 = 32u;

alias Spinor = vec4<f32>;

// ----------------------------------------------------------------------------
// BindGroup 0: HMC Core (バッファ数削減・統合版)
// ----------------------------------------------------------------------------
struct SimParams { 
    beta: f32, 
    kappa: f32, 
    eps: f32, 
    mass_or_isNewH: f32 // TSから isNewH または mass として使われる
};

@group(0) @binding(0) var<uniform> params: SimParams;

// 場と運動量
@group(0) @binding(1) var<storage, read_write> links: array<f32>;
@group(0) @binding(2) var<storage, read_write> higgs: array<f32>;
@group(0) @binding(3) var<storage, read_write> p_links: array<f32>;
@group(0) @binding(4) var<storage, read_write> p_higgs: array<f32>;
@group(0) @binding(5) var<storage, read_write> old_links: array<f32>;
@group(0) @binding(6) var<storage, read_write> old_higgs: array<f32>;

// 乱数・スカラー・ワークスペース
@group(0) @binding(7) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(8) var<storage, read_write> scalars: array<f32>;   // HMCとCGで共有！
@group(0) @binding(9) var<storage, read_write> workspace: array<f32>; // energyとviz_resultsを統合！

// ----------------------------------------------------------------------------
// BindGroup 1: Fermion Core (クォーク専用)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read_write> phi: array<Spinor>;
@group(1) @binding(1) var<storage, read_write> x: array<Spinor>;
@group(1) @binding(2) var<storage, read_write> cg_p: array<Spinor>;
@group(1) @binding(3) var<storage, read_write> cg_r: array<Spinor>;
@group(1) @binding(4) var<storage, read_write> cg_q: array<Spinor>;
@group(1) @binding(5) var<storage, read_write> tmp_Y: array<Spinor>;
// ※ binding 6 の scalars は Group 0 に統合したため削除されました。

// --- ヘルパー関数 ---
fn get_idx(cx: u32, cy: u32) -> u32 { return cy * L + cx; }
fn get_link_idx(cx: u32, cy: u32, dir: u32) -> u32 { return (cy * L + cx) * 2u + dir; }
fn wrap(a: f32) -> f32 { return atan2(sin(a), cos(a)); }

// --- スピノル演算ヘルパー ---
fn c_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
fn apply_U(theta: f32, psi: Spinor) -> Spinor {
    let u = vec2(cos(theta), sin(theta));
    let s1 = c_mult(u, vec2(psi.x, psi.y)); let s2 = c_mult(u, vec2(psi.z, psi.w));
    return Spinor(s1.x, s1.y, s2.x, s2.y);
}
fn gamma1_mul(psi: Spinor) -> Spinor { return Spinor(psi.z, psi.w, psi.x, psi.y); }
fn gamma2_mul(psi: Spinor) -> Spinor { return Spinor(psi.w, -psi.z, -psi.y, psi.x); }
fn spinor_dot_real(a: Spinor, b: Spinor) -> f32 { return dot(a, b); }

// --- 乱数 ---
fn pcg_hash(state: ptr<function, u32>) -> u32 {
    let old = *state; *state = old * 747796405u + 2891336453u;
    var word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}
fn rand_f32(state: ptr<function, u32>) -> f32 { return f32(pcg_hash(state)) / 4294967295.0; }
fn rand_normal(state: ptr<function, u32>) -> f32 {
    let u1 = max(rand_f32(state), 1e-8); let u2 = rand_f32(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// ============================================================================
// Phase 1: 擬フェルミオン場の生成 (Heat Bath)
// ============================================================================
@compute @workgroup_size(64)
fn generate_xi(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    var seed = rng_state[idx];
    
    // ガウシアンノイズ xi を生成し、一時バッファ tmp_Y に保存
    tmp_Y[idx] = Spinor(rand_normal(&seed), rand_normal(&seed), rand_normal(&seed), rand_normal(&seed));
    rng_state[idx] = seed;

    // ※この後、TS側で apply_D_dag を tmp_Y に掛けて phi に保存します
}

// ============================================================================
// Phase 2: ディラック演算子 D (CGソルバー用 & Y=Dx 計算用)
// ============================================================================
// 入力 in_array から D を計算し、out_array へ出力する汎用関数
fn compute_D(in_array: ptr<storage, array<Spinor>, read_write>, idx: u32) -> Spinor {
    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;
    let xm1 = (cx + L - 1u) % L; let ym1 = (cy + L - 1u) % L;

    var out = in_array[idx] * (params.mass_or_isNewH + 2.0); // 質量項

    let p_R = apply_U(links[get_link_idx(cx, cy, 0u)], in_array[get_idx(xp1, cy)]);
    out -= 0.5 * (p_R - gamma1_mul(p_R));

    let p_L = apply_U(-links[get_link_idx(xm1, cy, 0u)], in_array[get_idx(xm1, cy)]);
    out -= 0.5 * (p_L + gamma1_mul(p_L));

    let p_U = apply_U(links[get_link_idx(cx, cy, 1u)], in_array[get_idx(cx, yp1)]);
    out -= 0.5 * (p_U - gamma2_mul(p_U));

    let p_D = apply_U(-links[get_link_idx(cx, ym1, 1u)], in_array[get_idx(cx, ym1)]);
    out -= 0.5 * (p_D + gamma2_mul(p_D));

    return out;
}

@compute @workgroup_size(64)
fn apply_D_for_cg(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < L * L) { tmp_Y[id.x] = compute_D(&cg_p, id.x); }
}


@compute @workgroup_size(64)
fn apply_D_dag_for_cg(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    
    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;
    let xm1 = (cx + L - 1u) % L; let ym1 = (cy + L - 1u) % L;

    var out = tmp_Y[idx] * (params.mass_or_isNewH + 2.0);

    let p_R = apply_U(links[get_link_idx(cx, cy, 0u)], tmp_Y[get_idx(xp1, cy)]);
    out -= 0.5 * (p_R + gamma1_mul(p_R)); // D^dagger はガンマ行列の符号反転

    let p_L = apply_U(-links[get_link_idx(xm1, cy, 0u)], tmp_Y[get_idx(xm1, cy)]);
    out -= 0.5 * (p_L - gamma1_mul(p_L));

    let p_U = apply_U(links[get_link_idx(cx, cy, 1u)], tmp_Y[get_idx(cx, yp1)]);
    out -= 0.5 * (p_U + gamma2_mul(p_U));

    let p_D = apply_U(-links[get_link_idx(cx, ym1, 1u)], tmp_Y[get_idx(cx, ym1)]);
    out -= 0.5 * (p_D - gamma2_mul(p_D));

    cg_q[idx] = out;
}

// ============================================================================
// Phase 3: フェルミオン力の計算 (Force)
// ============================================================================
@compute @workgroup_size(64)
fn compute_Y_from_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < L * L) { tmp_Y[id.x] = compute_D(&x, id.x); }
}

@compute @workgroup_size(64)
fn calc_fermion_force(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    
    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;

    let Y_c = tmp_Y[idx]; // Y(n)
    let x_c = x[idx];     // x(n)
    let eps = params.eps;

    // ==========================================
    // --- x方向の力 (Forward & Backward) ---
    // ==========================================
    let l_x = get_link_idx(cx, cy, 0u);
    let u_x = links[l_x];
    let x_R = x[get_idx(xp1, cy)];
    let Y_R = tmp_Y[get_idx(xp1, cy)];

    // Forward term (前方のサイトからの力)
    let dU_x = apply_U(u_x + PI / 2.0, x_R);
    let dD_x = Spinor(
        -0.5 * (dU_x.x - dU_x.z),
        -0.5 * (dU_x.y - dU_x.w),
        -0.5 * (dU_x.z - dU_x.x),
        -0.5 * (dU_x.w - dU_x.y)
    );
    let force_x_fwd = spinor_dot_real(Y_c, dD_x);

    // Backward term (後方のサイトからの力)
    let dU_x_rev = apply_U(PI / 2.0 - u_x, x_c);
    let dD_x_rev = Spinor(
        0.5 * (dU_x_rev.x + dU_x_rev.z),
        0.5 * (dU_x_rev.y + dU_x_rev.w),
        0.5 * (dU_x_rev.z + dU_x_rev.x),
        0.5 * (dU_x_rev.w + dU_x_rev.y)
    );
    let force_x_rev = spinor_dot_real(Y_R, dD_x_rev);

    let force_x = 2.0 * (force_x_fwd + force_x_rev);
    p_links[l_x] += eps * force_x; 

    // ==========================================
    // --- y方向の力 (Forward & Backward) ---
    // ==========================================
    let l_y = get_link_idx(cx, cy, 1u);
    let u_y = links[l_y];
    let x_U = x[get_idx(cx, yp1)];
    let Y_U = tmp_Y[get_idx(cx, yp1)];

    // Forward term (前方のサイトからの力)
    let dU_y = apply_U(u_y + PI / 2.0, x_U);
    let dD_y = Spinor(
        -0.5 * (dU_y.x - dU_y.w),
        -0.5 * (dU_y.y + dU_y.z),
        -0.5 * (dU_y.z + dU_y.y),
        -0.5 * (dU_y.w - dU_y.x)
    );
    let force_y_fwd = spinor_dot_real(Y_c, dD_y);

    // Backward term (後方のサイトからの力)
    let dU_y_rev = apply_U(PI / 2.0 - u_y, x_c);
    let dD_y_rev = Spinor(
        0.5 * (dU_y_rev.x + dU_y_rev.w),
        0.5 * (dU_y_rev.y - dU_y_rev.z),
        0.5 * (dU_y_rev.z - dU_y_rev.y),
        0.5 * (dU_y_rev.w + dU_y_rev.x)
    );
    let force_y_rev = spinor_dot_real(Y_U, dD_y_rev);

    let force_y = 2.0 * (force_y_fwd + force_y_rev);
    p_links[l_y] += eps * force_y;
}

// ============================================================================
// Phase 5: CGソルバーの更新ループ (cg_p, cg_r, cg_q に名前を変更)
// ============================================================================

// --- 内積 ---
@compute @workgroup_size(1)
fn calc_p_q() {
    var sum: f32 = 0.0;
    for(var i = 0u; i < L * L; i++) {
        sum += dot(cg_p[i], cg_q[i]); // p を cg_p, q を cg_q に修正
    }
    scalars[1] = sum;
    scalars[2] = scalars[0] / (sum + 1e-12);
}

// --- x, r の更新 ---
@compute @workgroup_size(64)
fn update_x_r(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= L * L) { return; }
    let alpha = scalars[2];
    x[i] = x[i] + alpha * cg_p[i];           // p を cg_p に修正
    cg_r[i] = cg_r[i] - alpha * cg_q[i];     // r を cg_r, q を cg_q に修正
}

// --- 新しい rho ---
@compute @workgroup_size(1)
fn calc_new_rho() {
    var sum: f32 = 0.0;
    for(var i = 0u; i < L * L; i++) {
        sum += dot(cg_r[i], cg_r[i]);        // r を cg_r に修正
    }
    scalars[4] = sum;
    scalars[3] = sum / (scalars[0] + 1e-12);
    scalars[0] = sum;
}

// --- p の更新 ---
@compute @workgroup_size(64)
fn update_p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= L * L) { return; }
    let beta = scalars[3];
    cg_p[i] = cg_r[i] + beta * cg_p[i];      // p を cg_p, r を cg_r に修正
}

// ============================================================================
// Phase 4: CGソルバーの初期化
// ============================================================================
@compute @workgroup_size(64)
fn reset_cg(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    
    x[idx] = Spinor(0.0, 0.0, 0.0, 0.0);
    cg_r[idx] = phi[idx];
    cg_p[idx] = phi[idx];
}

@compute @workgroup_size(1)
fn calc_initial_rho() {
    var sum = 0.0;
    for(var i = 0u; i < L * L; i++) { sum += spinor_dot_real(cg_r[i], cg_r[i]); }
    scalars[0] = sum; // rho
}

// ============================================================================
// Phase 6: HMC Core (軌道計算・エネルギー・受容判定)
// ============================================================================

@compute @workgroup_size(64)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    var seed = rng_state[idx];
    higgs[idx] = (rand_f32(&seed) * 2.0 - 1.0) * PI;
    links[idx * 2u + 0u] = (rand_f32(&seed) * 2.0 - 1.0) * PI;
    links[idx * 2u + 1u] = (rand_f32(&seed) * 2.0 - 1.0) * PI;
    rng_state[idx] = seed;
}

// ============================================================================
// コールドスタート (絶対零度・完全な真空での初期化)
// ============================================================================
@compute @workgroup_size(64)
fn init_cold(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    
    var seed = rng_state[idx];
    
    // 位相をすべて 0.0 (完全な真空・絶対零度) に揃える
    higgs[idx] = 0.0;
    links[idx * 2u + 0u] = 0.0;
    links[idx * 2u + 1u] = 0.0;
    
    // 乱数の状態を進めるための空回し
    let dummy = rand_f32(&seed);
    
    rng_state[idx] = seed;
}

@compute @workgroup_size(64)
fn init_trajectory(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    var seed = rng_state[idx];
    
    p_higgs[idx] = rand_normal(&seed);
    p_links[idx * 2u + 0u] = rand_normal(&seed);
    p_links[idx * 2u + 1u] = rand_normal(&seed);
    
    old_higgs[idx] = higgs[idx];
    old_links[idx * 2u + 0u] = links[idx * 2u + 0u];
    old_links[idx * 2u + 1u] = links[idx * 2u + 1u];

    if (idx == 0u) {
        scalars[7] = 0.0; // accepted_flag (インデックスを7にずらす)
        scalars[8] = rand_f32(&seed); // rand_val (インデックスを8にずらす)
    }
    rng_state[idx] = seed;
}

@compute @workgroup_size(64)
fn calc_local_H(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;

    var H = 0.5 * (p_higgs[idx] * p_higgs[idx] + 
                   p_links[idx * 2u + 0u] * p_links[idx * 2u + 0u] + 
                   p_links[idx * 2u + 1u] * p_links[idx * 2u + 1u]);

    let t_x = links[get_link_idx(cx, cy, 0u)];
    let t_y = links[get_link_idx(cx, cy, 1u)];
    
    let plaq = t_x + links[get_link_idx(xp1, cy, 1u)] - links[get_link_idx(cx, yp1, 0u)] - t_y;
    H -= params.beta * cos(plaq);
    H -= params.kappa * cos(higgs[idx] + t_x - higgs[get_idx(xp1, cy)]);
    H -= params.kappa * cos(higgs[idx] + t_y - higgs[get_idx(cx, yp1)]);

    // フェルミオンエネルギー S_F = phi^\dagger x の追加
    // (xが計算済みのH_newの時にフェルミオンの重みが足されます)
    H += spinor_dot_real(phi[idx], x[idx]);

    workspace[idx] = H; // energy_buffer を workspace に変更
}

@compute @workgroup_size(1)
fn reduce_H() {
    var sum = 0.0;
    for(var i = 0u; i < L * L; i++) { sum += workspace[i]; }
    
    // TS側で uintView を使って書き込んでいるため、ビットキャストで正確な 0 か 1 を読み取る
    let is_new = bitcast<u32>(params.mass_or_isNewH); 
    scalars[5u + is_new] = sum; // 5 が H_old, 6 が H_new
}

@compute @workgroup_size(64)
fn update_q(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    let eps = params.eps;
    higgs[idx] = wrap(higgs[idx] + eps * p_higgs[idx]);
    links[idx * 2u + 0u] = wrap(links[idx * 2u + 0u] + eps * p_links[idx * 2u + 0u]);
    links[idx * 2u + 1u] = wrap(links[idx * 2u + 1u] + eps * p_links[idx * 2u + 1u]);
}

@compute @workgroup_size(64)
fn update_P(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;
    let xm1 = (cx + L - 1u) % L; let ym1 = (cy + L - 1u) % L;

    var f_phi = 0.0;
    f_phi += params.kappa * sin(higgs[idx] + links[get_link_idx(cx, cy, 0u)] - higgs[get_idx(xp1, cy)]);
    f_phi -= params.kappa * sin(higgs[get_idx(xm1, cy)] + links[get_link_idx(xm1, cy, 0u)] - higgs[idx]);
    f_phi += params.kappa * sin(higgs[idx] + links[get_link_idx(cx, cy, 1u)] - higgs[get_idx(cx, yp1)]);
    f_phi -= params.kappa * sin(higgs[get_idx(cx, ym1)] + links[get_link_idx(cx, ym1, 1u)] - higgs[idx]);

    let t_x = links[get_link_idx(cx, cy, 0u)];
    var f_x = 0.0;
    f_x += params.beta * sin(t_x + links[get_link_idx(xp1, cy, 1u)] - links[get_link_idx(cx, yp1, 0u)] - links[get_link_idx(cx, cy, 1u)]);
    f_x += params.beta * sin(t_x - links[get_link_idx(xp1, ym1, 1u)] - links[get_link_idx(cx, ym1, 0u)] + links[get_link_idx(cx, ym1, 1u)]);
    f_x += params.kappa * sin(higgs[idx] + t_x - higgs[get_idx(xp1, cy)]);

    let t_y = links[get_link_idx(cx, cy, 1u)];
    var f_y = 0.0;
    f_y += params.beta * sin(t_y - links[get_link_idx(xp1, cy, 1u)] - links[get_link_idx(cx, cy, 0u)] + links[get_link_idx(cx, yp1, 0u)]);
    f_y += params.beta * sin(t_y + links[get_link_idx(xm1, cy, 0u)] - links[get_link_idx(xm1, yp1, 0u)] - links[get_link_idx(xm1, cy, 1u)]);
    f_y += params.kappa * sin(higgs[idx] + t_y - higgs[get_idx(cx, yp1)]);

    let eps = params.eps; 
    p_higgs[idx] -= eps * f_phi;
    p_links[get_link_idx(cx, cy, 0u)] -= eps * f_x;
    p_links[get_link_idx(cx, cy, 1u)] -= eps * f_y;
}

@compute @workgroup_size(64)
fn accept_reject(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }

    let H_old = scalars[5];
    let H_new = scalars[6];
    let rand_val = scalars[8];
    
    let dH = H_new - H_old;

    if (dH > 0.0 && exp(-dH) <= rand_val) {
        higgs[idx] = old_higgs[idx];
        links[idx * 2u + 0u] = old_links[idx * 2u + 0u];
        links[idx * 2u + 1u] = old_links[idx * 2u + 1u];
    } else if (idx == 0u) {
        scalars[7] = 1.0; 
    }
}

// ============================================================================
// Phase 7: 物理量の測定 (トポロジカル電荷とヒッグスエネルギー)
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn measure_observables_E(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);

    let phi_x = higgs[site_idx];
    let theta_x = links[get_link_idx(x, y, 0u)];
    let phi_xp1 = higgs[get_idx((x + 1u) % L, y)];

    workspace[site_idx] = cos(phi_x + theta_x - phi_xp1); // viz_results を workspace に変更
}

@compute @workgroup_size(8, 8, 1)
fn measure_observables_C(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);

    let xp1 = (x + 1u) % L;
    let yp1 = (y + 1u) % L;

    let dx_bottom = wrap(higgs[get_idx(xp1, y)] - higgs[get_idx(x, y)] - links[get_link_idx(x, y, 0u)]);
    let dy_right  = wrap(higgs[get_idx(xp1, yp1)] - higgs[get_idx(xp1, y)] - links[get_link_idx(xp1, y, 1u)]);
    let dx_top    = wrap(higgs[get_idx(xp1, yp1)] - higgs[get_idx(x, yp1)] - links[get_link_idx(x, yp1, 0u)]);
    let dy_left   = wrap(higgs[get_idx(x, yp1)] - higgs[get_idx(x, y)] - links[get_link_idx(x, y, 1u)]);

    let curl = dx_bottom + dy_right - dx_top - dy_left;

    let plaq = wrap(
        links[get_link_idx(x, y, 0u)] + 
        links[get_link_idx(xp1, y, 1u)] - 
        links[get_link_idx(x, yp1, 0u)] - 
        links[get_link_idx(x, y, 1u)]
    );

    let Q = round((curl + plaq) / (2.0 * PI)); 
    workspace[site_idx] = Q; // viz_results を workspace に変更
}
