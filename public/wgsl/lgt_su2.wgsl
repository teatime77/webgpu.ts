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

// Generate a random SU(2) matrix close to the identity.
// `epsilon` controls how far from identity the new matrix is.
fn random_su2_near_identity(epsilon: f32, seed: ptr<function, u32>) -> SU2Mat {
    // Generate a random 3D vector
    let r1 = (2.0 * random_f32(seed) - 1.0) * epsilon;
    let r2 = (2.0 * random_f32(seed) - 1.0) * epsilon;
    let r3 = (2.0 * random_f32(seed) - 1.0) * epsilon;

    // Project to a quaternion close to identity (1, 0, 0, 0)
    let norm_sq = r1 * r1 + r2 * r2 + r3 * r3;
    if (norm_sq > 1.0) { // Should be rare with small epsilon
        return SU2Mat(1.0, 0.0, 0.0, 0.0);
    }
    let a0 = sqrt(1.0 - norm_sq);

    return normalize(SU2Mat(a0, r1, r2, r3));
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

    // --- Metropolis Step ---
    let link_idx = get_link_idx(x, y, dir_to_update);
    let old_link = links[link_idx];

    // Propose a new link by multiplying with a random matrix near identity
    let new_link = normalize(su2_mult(random_su2_near_identity(0.5, &rand_seed), old_link));

    // delta_S = -beta/2 * Tr((new_link - old_link) * staple)
    let delta_S = -params.beta / 2.0 * (su2_trace(su2_mult(new_link, staple)) - su2_trace(su2_mult(old_link, staple)));

    // Accept/Reject
    if (exp(-delta_S) > random_f32(&rand_seed)) {
        links[link_idx] = new_link;
    }

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