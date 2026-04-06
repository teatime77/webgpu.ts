// ============================================================================
// U(1) Abelian Higgs Model (2D Superconductor / Vortex physics)
// ============================================================================

const PI: f32 = 3.1415926535;
const L: u32 = 64u; // 渦を綺麗に見るため L=128 がおすすめ

struct SimParams {
    beta: f32,
    kappa: f32,
    update_subset: u32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> links: array<f32>; // 角度 theta
@group(0) @binding(2) var<storage, read_write> higgs: array<f32>; // 角度 phi
@group(0) @binding(3) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(4) var<storage, read_write> viz_results: array<f32>;

// ============================================================================
// --- Helpers ---
// ============================================================================
fn get_idx(x: u32, y: u32) -> u32 { return y * L + x; }
fn get_link_idx(x: u32, y: u32, dir: u32) -> u32 { return (y * L + x) * 2u + dir; }

// 角度を [-π, π) に安全に丸める
fn wrap(a: f32) -> f32 { return atan2(sin(a), cos(a)); }

// --- PCG 乱数ジェネレータ ---
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


// ============================================================================
// Initialization Kernels
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn init_cold(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);

    // U(1)における単位元（位相ゼロ）は角度 0.0
    links[get_link_idx(x, y, 0u)] = 0.0;
    links[get_link_idx(x, y, 1u)] = 0.0;
    higgs[site_idx] = 0.0;

    // 乱数シードの空回し（初期化のお作法）
    var seed = rng_state[site_idx];
    let dummy = random_u32(&seed);
    rng_state[site_idx] = seed;
}

@compute @workgroup_size(8, 8, 1)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];

    // -π から π までの完全にランダムな角度（超高温のノイズ宇宙）
    links[get_link_idx(x, y, 0u)] = (random_f32(&seed) * 2.0 - 1.0) * PI;
    links[get_link_idx(x, y, 1u)] = (random_f32(&seed) * 2.0 - 1.0) * PI;
    higgs[site_idx] = (random_f32(&seed) * 2.0 - 1.0) * PI;

    rng_state[site_idx] = seed;
}

// ============================================================================
// 1. ヒッグス場の更新 (サイト)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn update_higgs(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    if ((x + y) % 2u != params.update_subset % 2u) { return; }

    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];
    let old_phi = higgs[site_idx];

    let xp1 = (x + 1u) % L; let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L; let ym1 = (y + L - 1u) % L;

    // 隣のヒッグス場と、間のゲージ場の角度
    let p_R = higgs[get_idx(xp1, y)]; let t_R = links[get_link_idx(x, y, 0u)];
    let p_U = higgs[get_idx(x, yp1)]; let t_U = links[get_link_idx(x, y, 1u)];
    let p_L = higgs[get_idx(xm1, y)]; let t_L = links[get_link_idx(xm1, y, 0u)];
    let p_D = higgs[get_idx(x, ym1)]; let t_D = links[get_link_idx(x, ym1, 1u)];

    // メトロポリス提案 (ランダムに角度をずらす)

    // 【変更前】
    // let new_phi = wrap(old_phi + (random_f32(&seed) * 2.0 - 1.0) * PI * 0.5);

    // 【変更後】適応的ステップサイズ
    let step_size = 2.0 / sqrt(params.kappa + 1.0);
    let new_phi = wrap(old_phi + (random_f32(&seed) * 2.0 - 1.0) * step_size);



    // 局所作用の計算: S = -kappa * cos(phi(x) + theta_mu(x) - phi(x+mu))
    let S_old = -params.kappa * (
        cos(old_phi + t_R - p_R) + cos(old_phi + t_U - p_U) +
        cos(p_L + t_L - old_phi) + cos(p_D + t_D - old_phi)
    );
    
    let S_new = -params.kappa * (
        cos(new_phi + t_R - p_R) + cos(new_phi + t_U - p_U) +
        cos(p_L + t_L - new_phi) + cos(p_D + t_D - new_phi)
    );

    let dS = S_new - S_old;
    if (dS < 0.0 || exp(-dS) > random_f32(&seed)) {
        higgs[site_idx] = new_phi;
    }
    rng_state[site_idx] = seed;
}

// ============================================================================
// 2. ゲージ場の更新 (リンク)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn update_gauge(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    if ((x + y) % 2u != params.update_subset % 2u) { return; }

    let dir = params.update_subset / 2u;
    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];

    let link_idx = get_link_idx(x, y, dir);
    let old_theta = links[link_idx];

    // 【変更前】
    // let new_theta = wrap(old_theta + (random_f32(&seed) * 2.0 - 1.0) * PI * 0.5);

    // 【変更後】適応的ステップサイズ
    let step_size = 2.0 / sqrt(params.beta + params.kappa + 1.0);
    let new_theta = wrap(old_theta + (random_f32(&seed) * 2.0 - 1.0) * step_size);

    let xp1 = (x + 1u) % L; let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L; let ym1 = (y + L - 1u) % L;

    var staple = 0.0;
    var S_old_h = 0.0; var S_new_h = 0.0;

    let phi_x = higgs[get_idx(x, y)];
    
    // 1. プラケット(磁場)と 2. ヒッグス(超伝導電流) のエネルギー
    if (dir == 0u) { // x方向
        let phi_xp1 = higgs[get_idx(xp1, y)];
        S_old_h = -params.kappa * cos(phi_x + old_theta - phi_xp1);
        S_new_h = -params.kappa * cos(phi_x + new_theta - phi_xp1);
        
        let up_plaq_old = old_theta + links[get_link_idx(xp1, y, 1u)] - links[get_link_idx(x, yp1, 0u)] - links[get_link_idx(x, y, 1u)];
        let dn_plaq_old = old_theta - links[get_link_idx(xp1, ym1, 1u)] - links[get_link_idx(x, ym1, 0u)] + links[get_link_idx(x, ym1, 1u)];
        let up_plaq_new = new_theta + links[get_link_idx(xp1, y, 1u)] - links[get_link_idx(x, yp1, 0u)] - links[get_link_idx(x, y, 1u)];
        let dn_plaq_new = new_theta - links[get_link_idx(xp1, ym1, 1u)] - links[get_link_idx(x, ym1, 0u)] + links[get_link_idx(x, ym1, 1u)];
        
        let S_old_g = -params.beta * (cos(up_plaq_old) + cos(dn_plaq_old));
        let S_new_g = -params.beta * (cos(up_plaq_new) + cos(dn_plaq_new));
        let dS = (S_new_h + S_new_g) - (S_old_h + S_old_g);

        if (dS < 0.0 || exp(-dS) > random_f32(&seed)) { links[link_idx] = new_theta; }
        
    } else { // y方向
        // (y方向も同様に xとyを入れ替えて計算。コード長節約のため理論は同じです)
        let phi_yp1 = higgs[get_idx(x, yp1)];
        S_old_h = -params.kappa * cos(phi_x + old_theta - phi_yp1);
        S_new_h = -params.kappa * cos(phi_x + new_theta - phi_yp1);
        
        let rt_plaq_old = old_theta - links[get_link_idx(xp1, y, 1u)] - links[get_link_idx(x, y, 0u)] + links[get_link_idx(x, yp1, 0u)];
        let lf_plaq_old = old_theta + links[get_link_idx(xm1, y, 0u)] - links[get_link_idx(xm1, yp1, 0u)] - links[get_link_idx(xm1, y, 1u)];
        let rt_plaq_new = new_theta - links[get_link_idx(xp1, y, 1u)] - links[get_link_idx(x, y, 0u)] + links[get_link_idx(x, yp1, 0u)];
        let lf_plaq_new = new_theta + links[get_link_idx(xm1, y, 0u)] - links[get_link_idx(xm1, yp1, 0u)] - links[get_link_idx(xm1, y, 1u)];
        
        let S_old_g = -params.beta * (cos(rt_plaq_old) + cos(lf_plaq_old));
        let S_new_g = -params.beta * (cos(rt_plaq_new) + cos(lf_plaq_new));
        let dS = (S_new_h + S_new_g) - (S_old_h + S_old_g);

        if (dS < 0.0 || exp(-dS) > random_f32(&seed)) { links[link_idx] = new_theta; }
    }
    rng_state[site_idx] = seed;
}

// ============================================================================
// 3. 物理量の測定
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn measure_observables_E(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);

    // ゲージ不変なヒッグス運動エネルギーを測定
    // cos(phi(x) + theta_x(x) - phi(x+1))
    let phi_x = higgs[site_idx];
    let theta_x = links[get_link_idx(x, y, 0u)];
    let phi_xp1 = higgs[get_idx((x + 1u) % L, y)];

    // 結果を保存（今回は u1_higgs_render.wgsl が higgs を直接読むため
    // 画面の「色」には影響しませんが、TS側のエラーを防ぎ、
    // 将来的にエネルギー分布を可視化する際に使えます）
    viz_results[site_idx] = cos(phi_x + theta_x - phi_xp1);
}

// ============================================================================
// 3. 物理量の測定 (DeGrand-Toussaint トポロジカル電荷)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn measure_observables_C(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);

    let xp1 = (x + 1u) % L;
    let yp1 = (y + 1u) % L;

    // 1. 各リンクのゲージ不変な位相差 (Covariant Derivative)
    let dx_bottom = wrap(higgs[get_idx(xp1, y)] - higgs[get_idx(x, y)] - links[get_link_idx(x, y, 0u)]);
    let dy_right  = wrap(higgs[get_idx(xp1, yp1)] - higgs[get_idx(xp1, y)] - links[get_link_idx(xp1, y, 1u)]);
    let dx_top    = wrap(higgs[get_idx(xp1, yp1)] - higgs[get_idx(x, yp1)] - links[get_link_idx(x, yp1, 0u)]);
    let dy_left   = wrap(higgs[get_idx(x, yp1)] - higgs[get_idx(x, y)] - links[get_link_idx(x, y, 1u)]);

    let curl = dx_bottom + dy_right - dx_top - dy_left;

    // 2. プラケット (磁場)
    let plaq = wrap(
        links[get_link_idx(x, y, 0u)] + 
        links[get_link_idx(xp1, y, 1u)] - 
        links[get_link_idx(x, yp1, 0u)] - 
        links[get_link_idx(x, y, 1u)]
    );

    // 3. トポロジカル電荷の計算
    // curl と plaq を足すことで、熱ノイズが完全に相殺される！
    // 浮動小数点の微小な誤差を消すために round() で確実に整数化します。
    let Q = round((curl + plaq) / (2.0 * PI)); 

    viz_results[site_idx] = Q; 
}