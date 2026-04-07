const PI: f32 = 3.1415926535;

// SU(2) matrix represented as a unit quaternion (a0, a1, a2, a3)
// a = a0*I + i*a1*sigma1 + i*a2*sigma2 + i*a3*sigma3
// where I is identity and sigma_k are Pauli matrices.
// Stored as vec4<f32>(a0, a1, a2, a3)
alias SU2Mat = vec4<f32>;

// --- SU(2) Math Helpers ---

// Multiply two SU(2) matrices (quaternion multiplication)
fn su2_mult(a: SU2Mat, b: SU2Mat) -> SU2Mat {
    return SU2Mat(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w, // a0*b0 - a.b
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z, // a0*b_vec + b0*a_vec + a_vec x b_vec
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

// Inverse of an SU(2) matrix (quaternion conjugate)
fn su2_inv(a: SU2Mat) -> SU2Mat {
    return SU2Mat(a.x, -a.y, -a.z, -a.w);
}

// Trace of an SU(2) matrix. For a = a0*I + i*a_k*sigma_k, Tr(a) = 2*a0.
fn su2_trace(a: SU2Mat) -> f32 {
    return 2.0 * a.x;
}

// 状態更新とハッシュ出力を分離した正しいPCG
fn random_u32(state: ptr<function, u32>) -> u32 {
    let old_state = *state;
    
    // 1. 状態の更新（LCGステップ：これにより 2^32 の周期が完全に保証される）
    *state = old_state * 747796405u + 2891336453u;

    // 2. 出力の計算（古い状態を使ってハッシュ化する）
    var word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    word = (word >> 22u) ^ word;

    return word;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
    return f32(random_u32(state)) / 4294967295.0; // 0.0 ~ 1.0
}

// --- SU(2) Heat Bath Helpers ---

// Kennedy-Pendletonアルゴリズムによる第0成分(x0)のサンプリング
// 目標分布: P(x0) ∝ sqrt(1 - x0^2) * exp(alpha * x0)
fn sample_su2_x0(alpha: f32, seed: ptr<function, u32>) -> f32 {
    if (alpha < 0.0001) {
        return 2.0 * random_f32(seed) - 1.0;
    }

    var x0: f32 = 0.0;
    for (var i = 0u; i < 1000u; i++) {
        // log(0) を防ぐために微小値を足す
        let r1 = random_f32(seed) + 1e-7;
        let r2 = random_f32(seed) + 1e-7;
        let r3 = random_f32(seed);
        let r4 = random_f32(seed);

        let x = -log(r1) / alpha;
        let y = -log(r2) / alpha;
        let z = cos(2.0 * PI * r3);
        let lambda = x + y * z * z;

        x0 = 1.0 - lambda;

        // 物理的にあり得ない範囲はリジェクト
        if (x0 < -1.0) { continue; }

        // 受容判定
        if (r4 * r4 <= 0.5 * (1.0 + x0)) {
            break;
        }
    }
    return x0;
}

// 与えられたステープルの強さ(alpha)に基づくランダムなSU(2)行列の生成
fn generate_su2_heatbath(alpha: f32, seed: ptr<function, u32>) -> SU2Mat {
    let x0 = sample_su2_x0(alpha, seed);
    let rho = sqrt(max(0.0, 1.0 - x0 * x0));

    // 3次元球面上のランダムな方向を決定
    let r5 = random_f32(seed);
    let r6 = random_f32(seed);
    let cos_theta = 2.0 * r5 - 1.0;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * r6;

    let x1 = rho * sin_theta * cos(phi);
    let x2 = rho * sin_theta * sin(phi);
    let x3 = rho * cos_theta;

    return SU2Mat(x0, x1, x2, x3);
}

// --- Simulation Constants & Bindings ---

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> links: array<SU2Mat>;
@group(0) @binding(2) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(3) var<storage, read_write> viz_results: array<f32>;

struct SimParams {
    beta: f32,
    update_subset: u32,
    filler1:f32,
    filler2:f32,
};

const L = 32;

// Helper to get link index
fn get_link_idx(x: u32, y: u32, dir: u32) -> u32 {
    return (y * L + x) * 2u + dir;
}

// --- Compute Kernels ---

@compute @workgroup_size(8, 8, 1)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    let site_idx = id.y * L + id.x;
    var rand_seed = rng_state[site_idx];

    // Generate a random SU(2) matrix (random unit quaternion)
    let u1 = random_f32(&rand_seed);
    let u2 = random_f32(&rand_seed);
    let u3 = random_f32(&rand_seed);

    let a0 = sqrt(1.0 - u1) * sin(2.0 * 3.14159265 * u2);
    let a1 = sqrt(1.0 - u1) * cos(2.0 * 3.14159265 * u2);
    let a2 = sqrt(u1) * sin(2.0 * 3.14159265 * u3);
    let a3 = sqrt(u1) * cos(2.0 * 3.14159265 * u3);

    let random_mat = SU2Mat(a0, a1, a2, a3);

    // Initialize links in both directions
    links[get_link_idx(id.x, id.y, 0u)] = random_mat;
    links[get_link_idx(id.x, id.y, 1u)] = random_mat;

    rng_state[site_idx] = rand_seed;
}

@compute @workgroup_size(8, 8, 1)
fn init_cold(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    // 境界チェック
    if (x >= L || y >= L) { return; }

    let site_idx = y * L + x;

    // SU(2)の単位行列（Cold Start用の真空状態）
    let identity_mat = SU2Mat(1.0, 0.0, 0.0, 0.0);

    // x方向、y方向ともに単位行列で初期化
    links[get_link_idx(x, y, 0u)] = identity_mat;
    links[get_link_idx(x, y, 1u)] = identity_mat;

    // 乱数シードの初期化（一回空回ししておく）
    var rand_seed = rng_state[site_idx];
    let dummy = random_u32(&rand_seed); 
    rng_state[site_idx] = rand_seed;
}

@compute @workgroup_size(8, 8, 1)
fn metropolis_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= L || y >= L) { return; } // 追加

    // Checkerboard update to avoid race conditions
    if ((x + y) % 2u != params.update_subset % 2u) {
        return;
    }
    
    let dir_to_update = params.update_subset / 2u; // 0 for x-dir, 1 for y-dir
    let site_idx = y * L + x;
    var rand_seed = rng_state[site_idx];

    // --- Calculate the staple V ---
    let xp1 = (x + 1u) % L;
    let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L;
    let ym1 = (y + L - 1u) % L;

    var staple: SU2Mat;
    if (dir_to_update == 0u) { // Updating an x-link at (x, y)
        let staple_up = su2_mult(links[get_link_idx(xp1, y, 1u)], su2_mult(su2_inv(links[get_link_idx(x, yp1, 0u)]), su2_inv(links[get_link_idx(x, y, 1u)])));
        let staple_down = su2_mult(su2_inv(links[get_link_idx(xp1, ym1, 1u)]), su2_mult(su2_inv(links[get_link_idx(x, ym1, 0u)]), links[get_link_idx(x, ym1, 1u)]));
        staple = staple_up + staple_down;
    } else { // Updating a y-link at (x, y)
        let staple_right = su2_mult(links[get_link_idx(x, yp1, 0u)], su2_mult(su2_inv(links[get_link_idx(xp1, y, 1u)]), su2_inv(links[get_link_idx(x, y, 0u)])));
        let staple_left = su2_mult(su2_inv(links[get_link_idx(xm1, yp1, 0u)]), su2_mult(su2_inv(links[get_link_idx(xm1, y, 1u)]), links[get_link_idx(xm1, y, 0u)]));
        staple = staple_right + staple_left;
    }

    // --- Heat Bath Step ---
    let link_idx = get_link_idx(x, y, dir_to_update);
    
    // ステープル(4次元ベクトル)の長さを計算
    let k = length(staple);
    var new_link: SU2Mat;

    if (k > 0.0001) {
        // 正規化してSU(2)行列 W を作る
        let W = staple / k;
        let alpha = params.beta * k;
        
        // 目標の確率分布に従う行列 X を生成
        let X = generate_su2_heatbath(alpha, &rand_seed);
        
        // U = X * W^\dagger (ステープルの方向に向ける)
        new_link = su2_mult(X, su2_inv(W));
    } else {
        // ステープルが0の場合（初期など）は完全ランダム
        new_link = generate_su2_heatbath(0.0, &rand_seed);
    }

    // メトロポリス判定の式（exp(-delta_S)...）は完全に不要になります。
    // 無条件で受容してメモリに書き込む。
    links[link_idx] = new_link;
    rng_state[site_idx] = rand_seed;
}


@compute @workgroup_size(8, 8, 1)
fn measure_plaquette(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let site_idx = y * L + x;

    let xp1 = (x + 1u) % L;
    let yp1 = (y + 1u) % L;

    // Plaquette = U_1 U_2 U_3^dag U_4^dag
    let U1 = links[get_link_idx(x, y, 0u)];
    let U2 = links[get_link_idx(xp1, y, 1u)];
    let U3_inv = su2_inv(links[get_link_idx(x, yp1, 0u)]);
    let U4_inv = su2_inv(links[get_link_idx(x, y, 1u)]);

    let plaquette_matrix = su2_mult(U1, su2_mult(U2, su2_mult(U3_inv, U4_inv)));

    // Store 1/2 * Tr(U_p) for visualization and measurement
    viz_results[site_idx] = 0.5 * su2_trace(plaquette_matrix);
}