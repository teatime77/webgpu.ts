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

// (x,y)からx方向(dx)またはy方向(dy)に進む際のリンクを取得する。
// ※ dx, dy はマイナスをとるため i32
fn get_link_dir(x: u32, y: u32, dx: i32, dy: i32) -> SU2Mat {
    if (dx == 1) {
        // 右に進む場合（正方向）
        return get_link(vec2<u32>(x, y), 0u);
    } else if (dx == -1) {
        // 左に進む場合（逆走なので一つ左のリンクをエルミート共役/逆行列にする）
        let prev_x = (x + L - 1u) % L;
        return su2_inv(get_link(vec2<u32>(prev_x, y), 0u));
    } else if (dy == 1) {
        // 上に進む場合（正方向）
        return get_link(vec2<u32>(x, y), 1u);
    } else if (dy == -1) {
        // 下に進む場合（逆走なので一つ下のリンクをエルミート共役/逆行列にする）
        let prev_y = (y + L - 1u) % L;
        return su2_inv(get_link(vec2<u32>(x, prev_y), 1u));
    }
    
    // 原則ここには来ないが、念のためSU(2)の単位元(1, 0, 0, 0)を返す
    return SU2Mat(1.0, 0.0, 0.0, 0.0);
}

// u32とi32の数をmod L で加算する。（変更なし）
fn addModL(x : u32, dx : i32) -> u32 {
    return u32(i32(x) + i32(L) + dx) % L;
}

// 3個のリンクのステープルを計算する。
fn get_staple(start_x: u32, start_y: u32, dx1: i32, dy1: i32, dx2: i32, dy2: i32, dx3: i32, dy3: i32) -> SU2Mat {
    // WGSLでは引数は書き換えられないので、内部変数を用意する
    var cx = start_x;
    var cy = start_y;

    // 1番目のリンク
    let l1 = get_link_dir(cx, cy, dx1, dy1);
    // 座標を更新（マイナスになっても L を足して % L することで安全にラップアラウンド）
    cx = addModL(cx, dx1);
    cy = addModL(cy, dy1);

    // 2番目のリンク
    let l2 = get_link_dir(cx, cy, dx2, dy2);
    cx = addModL(cx, dx2);
    cy = addModL(cy, dy2);

    // 3番目のリンク
    let l3 = get_link_dir(cx, cy, dx3, dy3);

    // 3個のリンクを乗算する。
    // ※ SU(2)は非可換群なので、乗算の順序 (l1 * l2 * l3) が物理的に極めて重要です。
    return su2_mult(su2_mult(l1, l2), l3);
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
    let mx = x % L;
    let my = y % L;
    return (my * L + mx) * 2u + dir;
}

// --- (必要であれば追加) get_link_dir が依存する補助関数 ---
fn get_link(pos: vec2<u32>, dir: u32) -> SU2Mat {
    return links[get_link_idx(pos.x, pos.y, dir)];
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

// ============================================================================
// メトロポリス（ヒートバス）更新カーネル
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn metropolis_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= L || y >= L) { return; } 

    // Checkerboard update to avoid race conditions
    if ((x + y) % 2u != params.update_subset % 2u) {
        return;
    }
    
    let dir_to_update = params.update_subset / 2u; // 0 for x-dir, 1 for y-dir
    let site_idx = y * L + x;
    var rand_seed = rng_state[site_idx];

    // --- Calculate the staple V ---
    var staple: SU2Mat;
    
    if (dir_to_update == 0u) { 
        // x方向リンクの更新
        // 更新対象のリンクの終点 (x+1, y) から出発して、(x, y) に戻る3辺
        
        // 上側のコの字 (上 -> 左 -> 下)
        let staple_up = get_staple(x + 1u, y, 0, 1, -1, 0, 0, -1);
        // 下側のコの字 (下 -> 左 -> 上)
        let staple_down = get_staple(x + 1u, y, 0, -1, -1, 0, 0, 1);
        
        staple = staple_up + staple_down;
        
    } else { 
        // y方向リンクの更新
        // 更新対象のリンクの終点 (x, y+1) から出発して、(x, y) に戻る3辺
        
        // 右側のコの字 (右 -> 下 -> 左)
        let staple_right = get_staple(x, y + 1u, 1, 0, 0, -1, -1, 0);
        // 左側のコの字 (左 -> 下 -> 右)
        let staple_left = get_staple(x, y + 1u, -1, 0, 0, -1, 1, 0);
        
        staple = staple_right + staple_left;
    }

    // --- Heat Bath Step ---
    let link_idx = get_link_idx(x, y, dir_to_update);
    let k = length(staple);
    var new_link: SU2Mat;

    if (k > 0.0001) {
        let W = staple / k;
        let alpha = params.beta * k;
        let X = generate_su2_heatbath(alpha, &rand_seed);
        new_link = su2_mult(X, su2_inv(W));
    } else {
        new_link = generate_su2_heatbath(0.0, &rand_seed);
    }

    links[link_idx] = new_link;
    rng_state[site_idx] = rand_seed;
}

// ============================================================================
// プラケット測定カーネル
// ============================================================================
@compute @workgroup_size(8, 8, 1)
fn measure_plaquette(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= L || y >= L) { return; } // 安全のための境界チェックを追加
    let site_idx = y * L + x;

    // Plaquette = 正方形の閉路を一周する (右 -> 上 -> 左 -> 下)
    // 逆方向のエルミート共役(逆行列)処理などは get_link_dir が自動で処理します。
    
    // 1. (x, y) から右へ
    let U1 = get_link_dir(x, y, 1, 0);
    // 2. (x+1, y) から上へ
    let U2 = get_link_dir(x + 1u, y, 0, 1);
    // 3. (x+1, y+1) から左へ
    let U3 = get_link_dir(x + 1u, y + 1u, -1, 0);
    // 4. (x, y+1) から下へ
    let U4 = get_link_dir(x, y + 1u, 0, -1);

    // 4つのリンクを順番に乗算
    let plaquette_matrix = su2_mult(su2_mult(U1, U2), su2_mult(U3, U4));

    // Store 1/2 * Tr(U_p) for visualization and measurement
    viz_results[site_idx] = 0.5 * su2_trace(plaquette_matrix);
}