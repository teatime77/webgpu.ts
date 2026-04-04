// ============================================================================
// SU(2) Gauge-Higgs Model (Full Scale)
// Frozen Higgs limit: Gauge field (links) and Higgs field (sites) are SU(2).
// ============================================================================

const PI: f32 = 3.1415926535;
const L: u32 = 128u;

alias SU2Mat = vec4<f32>;

// --- SU(2) Math Helpers ---
fn su2_mult(a: SU2Mat, b: SU2Mat) -> SU2Mat {
    return SU2Mat(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

fn su2_inv(a: SU2Mat) -> SU2Mat {
    return SU2Mat(a.x, -a.y, -a.z, -a.w);
}

fn su2_trace(a: SU2Mat) -> f32 {
    return 2.0 * a.x;
}

// --- PCG RNG & Heat Bath Helpers ---
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

fn generate_su2_heatbath(alpha: f32, seed: ptr<function, u32>) -> SU2Mat {
    let x0 = sample_su2_x0(alpha, seed);
    let rho = sqrt(max(0.0, 1.0 - x0 * x0));
    let r5 = random_f32(seed);
    let r6 = random_f32(seed);
    let cos_theta = 2.0 * r5 - 1.0;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * r6;
    return SU2Mat(x0, rho * sin_theta * cos(phi), rho * sin_theta * sin(phi), rho * cos_theta);
}

// --- Bindings & Params ---
struct SimParams {
    beta: f32,
    kappa: f32,
    update_subset: u32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> links: array<SU2Mat>; // ゲージ場
@group(0) @binding(2) var<storage, read_write> higgs: array<SU2Mat>; // ヒッグス場
@group(0) @binding(3) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(4) var<storage, read_write> viz_results: array<f32>;

fn get_idx(x: u32, y: u32) -> u32 { return y * L + x; }
fn get_link_idx(x: u32, y: u32, dir: u32) -> u32 { return (y * L + x) * 2u + dir; }

// --- Initialization Kernels ---
@compute @workgroup_size(8, 8, 1)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];

    let ident = SU2Mat(1.0, 0.0, 0.0, 0.0); // 簡略化のためhot startも一旦identからヒートバスで乱す運用を推奨
    links[get_link_idx(x, y, 0u)] = generate_su2_heatbath(0.0, &seed);
    links[get_link_idx(x, y, 1u)] = generate_su2_heatbath(0.0, &seed);
    higgs[site_idx] = generate_su2_heatbath(0.0, &seed);
    
    rng_state[site_idx] = seed;
}

// ============================================================================
// 1. ヒッグス場の更新 (サイトの更新)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn update_higgs(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    if ((x + y) % 2u != params.update_subset % 2u) { return; }

    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];

    let xp1 = (x + 1u) % L; let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L; let ym1 = (y + L - 1u) % L;

    // 隣接するヒッグス場を、間のゲージ場(リンク)を使って平行移動して集める（共変微分）
    let f_right = su2_mult(links[get_link_idx(x, y, 0u)], higgs[get_idx(xp1, y)]);
    let f_up    = su2_mult(links[get_link_idx(x, y, 1u)], higgs[get_idx(x, yp1)]);
    let f_left  = su2_mult(su2_inv(links[get_link_idx(xm1, y, 0u)]), higgs[get_idx(xm1, y)]);
    let f_down  = su2_mult(su2_inv(links[get_link_idx(x, ym1, 1u)]), higgs[get_idx(x, ym1)]);

    // これがヒッグス場が感じる「周囲の力(ステープル)」
    let V_H = f_right + f_up + f_left + f_down;
    let k = length(V_H);

    var new_phi: SU2Mat;
    if (k > 0.0001) {
        let W = V_H / k;
        let alpha = 2.0 * params.kappa * k; // ヒッグスの結合強度
        let X = generate_su2_heatbath(alpha, &seed);
        new_phi = su2_mult(X, W);
    } else {
        new_phi = generate_su2_heatbath(0.0, &seed);
    }

    higgs[site_idx] = new_phi;
    rng_state[site_idx] = seed;
}

// ============================================================================
// 2. ゲージ場の更新 (リンクの更新)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn update_gauge(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    if ((x + y) % 2u != params.update_subset % 2u) { return; }

    let dir_to_update = params.update_subset / 2u;
    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];

    let xp1 = (x + 1u) % L; let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L; let ym1 = (y + L - 1u) % L;

    var staple_gauge: SU2Mat;
    var phi_x = higgs[get_idx(x, y)];
    var phi_x_mu: SU2Mat;

    // 1. 純粋ゲージ理論のステープル（四角形の残り3辺）を計算
    if (dir_to_update == 0u) {
        let staple_up = su2_mult(links[get_link_idx(xp1, y, 1u)], su2_mult(su2_inv(links[get_link_idx(x, yp1, 0u)]), su2_inv(links[get_link_idx(x, y, 1u)])));
        let staple_down = su2_mult(su2_inv(links[get_link_idx(xp1, ym1, 1u)]), su2_mult(su2_inv(links[get_link_idx(x, ym1, 0u)]), links[get_link_idx(x, ym1, 1u)]));
        staple_gauge = staple_up + staple_down;
        phi_x_mu = higgs[get_idx(xp1, y)];
    } else {
        let staple_right = su2_mult(links[get_link_idx(x, yp1, 0u)], su2_mult(su2_inv(links[get_link_idx(xp1, y, 1u)]), su2_inv(links[get_link_idx(x, y, 0u)])));
        let staple_left = su2_mult(su2_inv(links[get_link_idx(xm1, yp1, 0u)]), su2_mult(su2_inv(links[get_link_idx(xm1, y, 1u)]), links[get_link_idx(xm1, y, 0u)]));
        staple_gauge = staple_right + staple_left;
        phi_x_mu = higgs[get_idx(x, yp1)];
    }

    // 2. ヒッグス場からのステープル（両端のサイトが引き合う力）
    let staple_higgs = su2_mult(phi_x_mu, su2_inv(phi_x));

    // ゲージ場が受けるすべての力 = (磁場の力) + (ヒッグス電流の力)
    let V_total = params.beta * staple_gauge + 2.0 * params.kappa * staple_higgs;
    let k = length(V_total);

    var new_link: SU2Mat;
    if (k > 0.0001) {
        let W = V_total / k;
        let alpha = k;
        let X = generate_su2_heatbath(alpha, &seed);
        new_link = su2_mult(X, su2_inv(W));
    } else {
        new_link = generate_su2_heatbath(0.0, &seed);
    }

    links[get_link_idx(x, y, dir_to_update)] = new_link;
    rng_state[site_idx] = seed;
}

// ============================================================================
// 3. 物理量の測定 (ゲージ不変なヒッグス運動エネルギー)
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn measure_observables(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    let site_idx = get_idx(x, y);
    
    // ヒッグス場とゲージ場がどれだけ「揃っているか」を測定
    // ReTr( phi(x)^dagger * U_x(x) * phi(x+1) )
    let phi_x = higgs[site_idx];
    let U_x = links[get_link_idx(x, y, 0u)];
    let phi_xp1 = higgs[get_idx((x + 1u) % L, y)];

    let higgs_link = su2_mult(su2_inv(phi_x), su2_mult(U_x, phi_xp1));

    // 0.5 * Tr = ReTr
    viz_results[site_idx] = su2_trace(higgs_link) * 0.5;
}