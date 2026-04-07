// ============================================================================
// Higgs Field (Complex Scalar Field) Simulation
// ============================================================================

const L: u32 = 128u; // 視覚的な迫力を出すため 128x128 など少し大きめがおすすめ
const L_squared: u32 = L * L;
const WORKGROUP_SIZE: u32 = 8u;

// 複素スカラー場 (2成分の実数ベクトル)
alias ScalarField = vec2<f32>;

struct SimParams {
    kappa: f32,       // 隣り合う格子点との結合の強さ（温度の逆数に相当）
    lambda_c: f32,    // メキシカンハットの深さ (0.1 〜 1.0 程度)
    update_subset: u32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> field: array<ScalarField>;
@group(0) @binding(2) var<storage, read_write> rng_state: array<u32>;

// --- PCG乱数 ---
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

fn get_idx(x: u32, y: u32) -> u32 {
    return y * L + x;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }
    
    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];
    
    // 最初はランダムな微小ゆらぎ (対称性が保たれた高温状態)
    let rx = (random_f32(&seed) * 2.0 - 1.0) * 0.1;
    let ry = (random_f32(&seed) * 2.0 - 1.0) * 0.1;
    
    field[site_idx] = vec2(rx, ry);
    rng_state[site_idx] = seed;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn metropolis_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= L || y >= L) { return; }

    if ((x + y) % 2u != params.update_subset % 2u) { return; }
    
    let site_idx = get_idx(x, y);
    var seed = rng_state[site_idx];

    let xp1 = (x + 1u) % L; let yp1 = (y + 1u) % L;
    let xm1 = (x + L - 1u) % L; let ym1 = (y + L - 1u) % L;

    // 隣接サイトの和 (Stapleに相当)
    let neighbors = field[get_idx(xp1, y)] + field[get_idx(xm1, y)] + 
                    field[get_idx(x, yp1)] + field[get_idx(x, ym1)];

    let old_phi = field[site_idx];
    
    // MCMCの提案: 現在の値にランダムな微小ベクトルを足す
    let step_size = 0.5;
    let jump = vec2((random_f32(&seed) * 2.0 - 1.0), (random_f32(&seed) * 2.0 - 1.0)) * step_size;
    let new_phi = old_phi + jump;

    // 局所エネルギー (作用 S) の計算
    // S = -kappa * (phi・neighbors) + |phi|^2 + lambda * (|phi|^2 - 1)^2
    let old_r2 = dot(old_phi, old_phi);
    let new_r2 = dot(new_phi, new_phi);

    let S_old = -params.kappa * dot(old_phi, neighbors) + old_r2 + params.lambda_c * (old_r2 - 1.0) * (old_r2 - 1.0);
    let S_new = -params.kappa * dot(new_phi, neighbors) + new_r2 + params.lambda_c * (new_r2 - 1.0) * (new_r2 - 1.0);

    let dS = S_new - S_old;

    // メトロポリス判定
    if (dS < 0.0 || exp(-dS) > random_f32(&seed)) {
        field[site_idx] = new_phi;
    }

    rng_state[site_idx] = seed;
}
