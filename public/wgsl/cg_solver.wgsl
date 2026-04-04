// ============================================================================
// WebGPU Conjugate Gradient Solver (Test: 2D Laplacian + Mass)
// ============================================================================

const L: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> scalars: array<f32>; 

fn get_idx(x_c: u32, y_c: u32) -> u32 {
    return y_c * L + x_c;
}

// --- Kernel 1: 行列ベクトル積 q = A * p ---
@compute @workgroup_size(64)
fn apply_A(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= L * L) { return; }

    let cx = idx % L;
    let cy = idx / L;

    let xp1 = (cx + 1u) % L;
    let yp1 = (cy + 1u) % L;
    let xm1 = (cx + L - 1u) % L;
    let ym1 = (cy + L - 1u) % L;

    let p_center = p[idx];
    let p_right  = p[get_idx(xp1, cy)];
    let p_left   = p[get_idx(xm1, cy)];
    let p_up     = p[get_idx(cx, yp1)];
    let p_down   = p[get_idx(cx, ym1)];

    // 変更点: 質量項(mass^2 = 0.05)を追加して発散を防ぐ！
    q[idx] = (4.0 + 0.05) * p_center - p_right - p_left - p_up - p_down;
}

// --- Kernel 2: 内積 p_q = p * q と alpha の計算 ---
@compute @workgroup_size(1)
fn calc_p_q() {
    var sum: f32 = 0.0;
    for(var i = 0u; i < L * L; i++) {
        sum += p[i] * q[i];
    }
    scalars[1] = sum;
    
    // 【重要】ここで alpha の計算を済ませておく！(同期ズレが起きない)
    // ゼロ割りを防ぐための微小値(1e-12)を足す
    scalars[2] = scalars[0] / (sum + 1e-12); 
}

// --- Kernel 3: x と r の更新 ---
@compute @workgroup_size(64)
fn update_x_r(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= L * L) { return; } // workgroupBarrier が消えたのでシンプルに

    let alpha = scalars[2];
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * q[i];
}

// --- Kernel 4: new_rho = r * r と beta の計算 ---
@compute @workgroup_size(1)
fn calc_new_rho() {
    var sum: f32 = 0.0;
    for(var i = 0u; i < L * L; i++) {
        sum += r[i] * r[i];
    }
    scalars[4] = sum;
    
    // 【重要】ここで beta と rho の更新を済ませておく！
    scalars[3] = sum / (scalars[0] + 1e-12); // beta = new_rho / rho
    scalars[0] = sum;                        // rho = new_rho
}

// --- Kernel 5: p の更新 ---
@compute @workgroup_size(64)
fn update_p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= L * L) { return; }

    let beta = scalars[3];
    p[i] = r[i] + beta * p[i];
}