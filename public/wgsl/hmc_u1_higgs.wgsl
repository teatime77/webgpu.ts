// ============================================================================
// WebGPU HMC for U(1) Gauge-Higgs Model (Pure Gauge & Higgs)
// ============================================================================
const PI: f32 = 3.1415926535;
const L: u32 = 128u;

struct SimParams {
    beta: f32,
    kappa: f32,
    eps: f32,       // ステップ幅 (例: 0.05)
    is_new_H: u32,  // 0ならH_old, 1ならH_new に保存
};

@group(0) @binding(0) var<uniform> params: SimParams;

// --- 場と運動量 ---
@group(0) @binding(1) var<storage, read_write> links: array<f32>;     // theta
@group(0) @binding(2) var<storage, read_write> higgs: array<f32>;     // phi
@group(0) @binding(3) var<storage, read_write> p_links: array<f32>;   // P_theta
@group(0) @binding(4) var<storage, read_write> p_higgs: array<f32>;   // P_phi

// --- ロールバック用バックアップ ---
@group(0) @binding(5) var<storage, read_write> old_links: array<f32>; 
@group(0) @binding(6) var<storage, read_write> old_higgs: array<f32>; 

// --- 計算用バッファ ---
@group(0) @binding(7) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(8) var<storage, read_write> scalars: array<f32>;   // [H_old, H_new, accepted, rand_val]
@group(0) @binding(9) var<storage, read_write> energy_buffer: array<f32>; // 各サイトの局所エネルギー
@group(0) @binding(10) var<storage, read_write> viz_results: array<f32>; // 観測結果用バッファ (トポロジカル電荷等)

fn get_idx(x: u32, y: u32) -> u32 { return y * L + x; }
fn get_link_idx(x: u32, y: u32, dir: u32) -> u32 { return (y * L + x) * 2u + dir; }
fn wrap(a: f32) -> f32 { return atan2(sin(a), cos(a)); }

// --- 乱数 ---
fn pcg_hash(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    var word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}
fn rand_f32(state: ptr<function, u32>) -> f32 { return f32(pcg_hash(state)) / 4294967295.0; }
fn rand_normal(state: ptr<function, u32>) -> f32 {
    let u1 = max(rand_f32(state), 1e-8);
    let u2 = rand_f32(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// ============================================================================
// 0. 初期化 (Hot Start)
// ============================================================================
@compute @workgroup_size(64)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    
    var seed = rng_state[idx];
    // -π から π のランダムな角度で宇宙を熱する
    higgs[idx] = (rand_f32(&seed) * 2.0 - 1.0) * PI;
    links[idx * 2u + 0u] = (rand_f32(&seed) * 2.0 - 1.0) * PI;
    links[idx * 2u + 1u] = (rand_f32(&seed) * 2.0 - 1.0) * PI;
    rng_state[idx] = seed;
}

// ============================================================================
// 1. 初期化とバックアップ
// ============================================================================
@compute @workgroup_size(64)
fn init_trajectory(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    
    var seed = rng_state[idx];
    
    // 運動量をサンプリング
    p_higgs[idx] = rand_normal(&seed);
    p_links[idx * 2u + 0u] = rand_normal(&seed);
    p_links[idx * 2u + 1u] = rand_normal(&seed);
    
    // 状態のバックアップ
    old_higgs[idx] = higgs[idx];
    old_links[idx * 2u + 0u] = links[idx * 2u + 0u];
    old_links[idx * 2u + 1u] = links[idx * 2u + 1u];

    if (idx == 0u) {
        scalars[2] = 0.0; // accepted_flag
        scalars[3] = rand_f32(&seed); // 受容判定用の乱数
    }
    rng_state[idx] = seed;
}

// ============================================================================
// 2. ハミルトニアン (全エネルギー) の計算
// ============================================================================
@compute @workgroup_size(64)
fn calc_local_H(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }
    let cx = idx % L; let cy = idx / L;
    let xp1 = (cx + 1u) % L; let yp1 = (cy + 1u) % L;

    // 運動エネルギー (K = 1/2 P^2)
    var H = 0.5 * (p_higgs[idx] * p_higgs[idx] + 
                   p_links[idx * 2u + 0u] * p_links[idx * 2u + 0u] + 
                   p_links[idx * 2u + 1u] * p_links[idx * 2u + 1u]);

    // ポテンシャルエネルギー (作用 S)
    // 重複カウントを防ぐため、自分を起点とする「右と上のリンク」だけを計算
    let t_x = links[get_link_idx(cx, cy, 0u)];
    let t_y = links[get_link_idx(cx, cy, 1u)];
    
    // 磁場
    let plaq = t_x + links[get_link_idx(xp1, cy, 1u)] - links[get_link_idx(cx, yp1, 0u)] - t_y;
    H -= params.beta * cos(plaq);

    // ヒッグス相互作用
    H -= params.kappa * cos(higgs[idx] + t_x - higgs[get_idx(xp1, cy)]);
    H -= params.kappa * cos(higgs[idx] + t_y - higgs[get_idx(cx, yp1)]);

    energy_buffer[idx] = H;
}

@compute @workgroup_size(1)
fn reduce_H() {
    var sum = 0.0;
    for(var i = 0u; i < L * L; i++) { sum += energy_buffer[i]; }
    scalars[params.is_new_H] = sum; 
}

// ============================================================================
// 3. リープフロッグ積分
// ============================================================================
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

    // --- 力 (Force = -dS/dq) の計算 ---
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

    // 運動量を更新
    let eps = params.eps; 
    p_higgs[idx] -= eps * f_phi;
    p_links[get_link_idx(cx, cy, 0u)] -= eps * f_x;
    p_links[get_link_idx(cx, cy, 1u)] -= eps * f_y;
}

// ============================================================================
// 4. 受容・棄却判定 (Metropolis)
// ============================================================================
@compute @workgroup_size(64)
fn accept_reject(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }

    let H_old = scalars[0];
    let H_new = scalars[1];
    let rand_val = scalars[3];
    
    let dH = H_new - H_old;

    if (dH > 0.0 && exp(-dH) <= rand_val) {
        higgs[idx] = old_higgs[idx];
        links[idx * 2u + 0u] = old_links[idx * 2u + 0u];
        links[idx * 2u + 1u] = old_links[idx * 2u + 1u];
    } else if (idx == 0u) {
        scalars[2] = 1.0; 
    }
}

// ============================================================================
// 5. 物理量の測定 (トポロジカル電荷とヒッグスエネルギー)
// ============================================================================

// Eモード: ヒッグス運動エネルギー (画面に波や揺らぎとして見えます)
@compute @workgroup_size(8, 8, 1)
fn measure_observables_E(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);

    let phi_x = higgs[site_idx];
    let theta_x = links[get_link_idx(x, y, 0u)];
    let phi_xp1 = higgs[get_idx((x + 1u) % L, y)];

    viz_results[site_idx] = cos(phi_x + theta_x - phi_xp1);
}

// Cモード: トポロジカル電荷 (赤と青の綺麗な「渦」の点として見えます)
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
    viz_results[site_idx] = Q; 
}