// File: ./wgsl/lgt_u1.wgsl

// --- Constants ---
// These must match the values in the TypeScript file.
const L: u32 = 32u;
const L_squared: u32 = L * L;
const WORKGROUP_SIZE: u32 = 8u;

const PI: f32 = 3.1415926535;
const TWO_PI: f32 = 2.0 * PI;

// --- Uniforms & Storage Buffers ---
struct SimParams {
    beta: f32,
    update_subset: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> links: array<vec2<f32>>; // Stored as (cos, sin)
@group(0) @binding(2) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(3) var<storage, read_write> viz_results: array<f32>; // Can be bitcast from i32

// --- Helper Functions ---
fn get_site_idx(coord: vec2<u32>) -> u32 {
    return coord.y * L + coord.x;
}

fn get_link_idx(site_idx: u32, dir: u32) -> u32 {
    return 2u * site_idx + dir;
}

fn get_link(coord: vec2<u32>, dir: u32) -> vec2<f32> {
    let site_idx = get_site_idx(coord);
    let link_idx = get_link_idx(site_idx, dir);
    return links[link_idx];
}

fn set_link(coord: vec2<u32>, dir: u32, val: vec2<f32>) {
    let site_idx = get_site_idx(coord);
    let link_idx = get_link_idx(site_idx, dir);
    links[link_idx] = val;
}

// (x,y)からx方向(dx)またはy方向(dy)に進む際のリンクを取得する。
// ※ dx, dy はマイナスをとるため i32 に変更
fn get_link_dir(x: u32, y: u32, dx: i32, dy: i32) -> vec2<f32> {
    if (dx == 1) {
        // 右に進む場合（正方向）
        return get_link(vec2<u32>(x, y), 0u);
    } else if (dx == -1) {
        // 左に進む場合（逆走なので一つ左のリンクを複素共役）
        let prev_x = (x + L - 1u) % L;
        return complex_conj(get_link(vec2<u32>(prev_x, y), 0u));
    } else if (dy == 1) {
        // 上に進む場合（正方向）
        return get_link(vec2<u32>(x, y), 1u);
    } else if (dy == -1) {
        // 下に進む場合（逆走なので一つ下のリンクを複素共役）
        let prev_y = (y + L - 1u) % L;
        return complex_conj(get_link(vec2<u32>(x, prev_y), 1u));
    }
    
    // 原則ここには来ないが、念のため単位元を返す
    return vec2<f32>(1.0, 0.0);
}

// u32とi32の数をmod L で加算する。
fn addModL(x : u32, dx : i32) -> u32 {
    return u32(i32(x) + i32(L) + dx) % L;
}

// 3個のリンクのステープルを計算する。
fn get_staple(start_x: u32, start_y: u32, dx1: i32, dy1: i32, dx2: i32, dy2: i32, dx3: i32, dy3: i32) -> vec2<f32> {
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

    // 3番目のリンク（typo修正: l3に）
    let l3 = get_link_dir(cx, cy, dx3, dy3);

    // 3個のリンクを乗算する。
    return complex_mul(complex_mul(l1, l2), l3);
}

// 正しい32-bit PCG random number generator
fn pcg(state: ptr<function, u32>) -> f32 {
    let old_state = *state;
    // LCG step
    *state = old_state * 747796405u + 2891336453u;
    
    // PCG-RXS-M-XS (32-bit output)
    var word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
    word = (word >> 22u) ^ word;
    
    // 0.0 ~ 1.0 の範囲に正規化
    return f32(word) / f32(0xFFFFFFFFu);
}

// Complex multiplication: (a+ib) * (c+id)
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex conjugate: (a+ib)* = (a-ib)
fn complex_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2(a.x, -a.y);
}

// --- Compute Kernels ---

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn init_hot(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let site_idx = get_site_idx(global_id.xy);
    if (global_id.x >= L || global_id.y >= L) { return; }
    
    var site_rng_state = rng_state[site_idx];

    let angle1 = pcg(&site_rng_state) * TWO_PI;
    set_link(global_id.xy, 0u, vec2(cos(angle1), sin(angle1)));

    let angle2 = pcg(&site_rng_state) * TWO_PI;
    set_link(global_id.xy, 1u, vec2(cos(angle2), sin(angle2)));

    rng_state[site_idx] = site_rng_state;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn init_cold(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let site_idx = get_site_idx(global_id.xy);
    if (global_id.x >= L || global_id.y >= L) { return; }
    
    // 全てのリンクを角度0 (cos(0)=1, sin(0)=0) に揃える（真の真空状態）
    set_link(global_id.xy, 0u, vec2(1.0, 0.0));
    set_link(global_id.xy, 1u, vec2(1.0, 0.0));

    // 乱数のシードは初期化しておく
    var site_rng_state = rng_state[site_idx];
    let dummy = pcg(&site_rng_state); 
    rng_state[site_idx] = site_rng_state;
}

// フォン・ミーゼス分布 P(x) ∝ exp(kappa * cos(x)) から角度をサンプリング
fn sample_von_mises(kappa: f32, state: ptr<function, u32>) -> f32 {
    // 【特例処理】kappaが0（温度が無限大）のときは、完全にランダムな一様分布になるため、
    // 棄却法を使わずに一発で -PI から PI の角度を返して終了します。
    if (kappa == 0.0) {
        return (pcg(state) - 0.5) * TWO_PI;
    }
    
    var x: f32 = 0.0;
    
    // 【安全ループ】通常は while(true) で当たりが出るまで繰り返しますが、
    // GPUシェーダー（WGSL）で無限ループを書くとクラッシュ判定されるため、
    // 念のため「最大1000回まで」という上限をつけています。
    for (var i = 0u; i < 1000u; i++) {
        
        // 1. 候補となる角度 x を -PI から PI の間でランダムに提案する
        x = (pcg(state) - 0.5) * TWO_PI;
        
        // 2. 判定用の高さ u を 0.0 から 1.0 の間でランダムに引く
        let u = pcg(state);
        
        // 3. 候補 x のときの山の高さ（最大1.0に規格化済み）を計算し、u と比較する
        if (u <= exp(kappa * (cos(x) - 1.0))) {
            // u が山より低ければ「当たり」！ループを抜けてこの x を採用する。
            break;
        }
    }
    return x;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn metropolis_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= L || y >= L) { return; }

    let subset = (x + y) % 2u + 2u * (x % 2u);
    if (subset != params.update_subset) { return; }

    let site_idx = get_site_idx(global_id.xy);
    let site_coord = global_id.xy;
    var site_rng_state = rng_state[site_idx];

    // --- 修正点1: 常に最新のリンクをローカル変数で追跡 ---
    var current_u0 = get_link(site_coord, 0u);
    var current_u1 = get_link(site_coord, 1u);

// ==========================================================
    // Update U_0(n) link (Heat Bath)
    // ==========================================================
    let n_xp1 = vec2<u32>((x + 1u) % L, y);
    let n_yp1 = vec2<u32>(x, (y + 1u) % L);
    let n_ym1 = vec2<u32>(x, (y + L - 1u) % L);
    let n_xp1_ym1 = vec2<u32>((x + 1u) % L, (y + L - 1u) % L);

    let staple_up   = get_staple((x + 1u) % L, y,  0, 1, -1, 0,  0, -1);
    let staple_down = get_staple((x + 1u) % L, y,  0,-1, -1, 0,  0, 1);
    let staple_sum_1 = staple_up + staple_down;

    // --- Heat Bath ロジック ---
    let norm_1 = length(staple_sum_1);
    if (norm_1 > 0.0001) {
        let kappa_1 = params.beta * norm_1;
        let phi_1 = sample_von_mises(kappa_1, &site_rng_state);
        
        let rot_1 = vec2(cos(phi_1), sin(phi_1));
        // 中心となる方向 (V*)
        let v_star_1 = vec2(staple_sum_1.x, -staple_sum_1.y) / norm_1;
        current_u0 = complex_mul(rot_1, v_star_1);
    } else {
        // ステープルが0の場合は完全にランダム
        let angle = pcg(&site_rng_state) * TWO_PI;
        current_u0 = vec2(cos(angle), sin(angle));
    }
    set_link(site_coord, 0u, current_u0);

    // ==========================================================
    // Update U_1(n) link (Heat Bath)
    // ==========================================================
    let n_xm1 = vec2<u32>((x + L - 1u) % L, y);
    let n_xm1_yp1 = vec2<u32>((x + L - 1u) % L, (y + 1u) % L);

    let staple_right = get_staple(x, (y + 1u) % L,  1, 0,  0,-1, -1, 0);
    let staple_left  = get_staple(x, (y + 1u) % L, -1, 0,  0,-1,  1, 0);
    let staple_sum_2 = staple_right + staple_left;

    // --- Heat Bath ロジック ---
    let norm_2 = length(staple_sum_2);
    if (norm_2 > 0.0001) {
        let kappa_2 = params.beta * norm_2;
        let phi_2 = sample_von_mises(kappa_2, &site_rng_state);
        
        let rot_2 = vec2(cos(phi_2), sin(phi_2));
        // 中心となる方向 (V*)
        let v_star_2 = vec2(staple_sum_2.x, -staple_sum_2.y) / norm_2;
        current_u1 = complex_mul(rot_2, v_star_2);
    } else {
        let angle = pcg(&site_rng_state) * TWO_PI;
        current_u1 = vec2(cos(angle), sin(angle));
    }
    set_link(site_coord, 1u, current_u1);

    rng_state[site_idx] = site_rng_state;
}

// =============================================================================
// NEW KERNEL: Measure Vortices
// This kernel calculates the integer vortex charge for each plaquette.
// =============================================================================
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn measure_vortices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let site_idx = get_site_idx(global_id.xy);
    if (x >= L || y >= L) { return; }

    // A plaquette is defined by its bottom-left corner (x,y).
    // The angle sum is theta_1(n) + theta_2(n+x) - theta_1(n+y) - theta_2(n)
    let n = vec2<u32>(x, y);
    let n_xp1 = vec2<u32>((x + 1u) % L, y);
    let n_yp1 = vec2<u32>(x, (y + 1u) % L);

    // Get link variables (cos, sin) and convert to angles using atan2
    let theta1_n = atan2(get_link(n, 0u).y, get_link(n, 0u).x);
    let theta2_n = atan2(get_link(n, 1u).y, get_link(n, 1u).x);
    let theta1_n_yp1 = atan2(get_link(n_yp1, 0u).y, get_link(n_yp1, 0u).x);
    let theta2_n_xp1 = atan2(get_link(n_xp1, 1u).y, get_link(n_xp1, 1u).x);

    // Sum of angles around the plaquette. This can be outside [-PI, PI].
    let plaquette_angle_sum = theta1_n + theta2_n_xp1 - theta1_n_yp1 - theta2_n;

    // The vortex charge is the integer winding number.
    // We find it by seeing how many times 2*PI fits into the angle sum.
    let vortex_charge = i32(round(plaquette_angle_sum / TWO_PI));

    viz_results[site_idx] = bitcast<f32>(vortex_charge);
}

// =============================================================================
// KERNEL: Measure Plaquette Energy
// This kernel calculates the real part of the plaquette operator, Tr Re U_p.
// =============================================================================
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn measure_plaquette(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let site_idx = get_site_idx(global_id.xy);
    if (x >= L || y >= L) { return; }

    // Plaquette operator U_p = U_1(n) U_2(n+x) U_1(n+y)* U_2(n)*
    let n = vec2<u32>(x, y);
    let n_xp1 = vec2<u32>((x + 1u) % L, y);
    let n_yp1 = vec2<u32>(x, (y + 1u) % L);

    let p1 = get_link(n, 0u);
    let p2 = get_link(n_xp1, 1u);
    let p3 = complex_conj(get_link(n_yp1, 0u));
    let p4 = complex_conj(get_link(n, 1u));

    let plaquette_val = complex_mul(complex_mul(complex_mul(p1, p2), p3), p4);
    viz_results[site_idx] = plaquette_val.x; // Store the real part (cosine of the angle)
}
