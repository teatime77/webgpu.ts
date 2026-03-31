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

// PCG random number generator
fn pcg(state: ptr<function, u32>) -> f32 {
    let old_state = *state;
    *state = old_state * 747796405u + 2891336453u;
    let xorshifted = ((old_state >> 18u) ^ old_state) >> 27u;
    let rot = old_state >> 27u; // For a 32-bit state, the rotation amount is taken from the high bits of the state.
    let result = (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
    return f32(result) / f32(0xFFFFFFFFu);
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
fn metropolis_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= L || y >= L) { return; }

    // Checkerboard update to avoid race conditions
    let subset = (x + y) % 2u + 2u * (x % 2u);
    if (subset != params.update_subset) { return; }

    let site_idx = get_site_idx(global_id.xy);
    let site_coord = global_id.xy;
    var site_rng_state = rng_state[site_idx];

    // --- Update U_0(n) link (x-direction) ---
    // This link is part of two plaquettes: P(n) and P(n-y)
    let n_xp1 = vec2<u32>((x + 1u) % L, y);
    let n_yp1 = vec2<u32>(x, (y + 1u) % L);
    let n_ym1 = vec2<u32>(x, (y - 1u + L) % L);
    let n_xp1_ym1 = vec2<u32>((x + 1u) % L, (y - 1u + L) % L);

    // Staple from P(n) = U0(n) U1(n+x) U0(n+y)^-1 U1(n)^-1
    // The staple is V_up = U1(n+x) U0(n+y)^-1 U1(n)^-1
    let staple_up = complex_mul(complex_mul(get_link(n_xp1, 1u), complex_conj(get_link(n_yp1, 0u))), complex_conj(get_link(site_coord, 1u)));
    
    // Staple from P(n-y) = U0(n-y) U1(n-y+x) U0(n)^-1 U1(n-y)^-1.
    // The action part is Re(U0(n)^-1 * Staple), which is Re(U0(n) * Staple^dagger)
    // The staple^dagger is V_down_dag = U1(n-y+x)^-1 U0(n-y)^-1 U1(n-y)
    let staple_down = complex_mul(complex_mul(complex_conj(get_link(n_xp1_ym1, 1u)), complex_conj(get_link(n_ym1, 0u))), get_link(n_ym1, 1u));

    let staple_sum_1 = staple_up + staple_down;
    let old_link_1 = get_link(site_coord, 0u);
    let old_action_1 = dot(staple_sum_1, old_link_1);

    let rand_angle_1 = (pcg(&site_rng_state) - 0.5) * 2.0 * PI;
    let new_link_1 = vec2(cos(rand_angle_1), sin(rand_angle_1));
    let new_action_1 = dot(staple_sum_1, new_link_1);

    let dS_1 = new_action_1 - old_action_1;
    if (dS_1 > 0.0 || exp(params.beta * dS_1) > pcg(&site_rng_state)) {
        set_link(site_coord, 0u, new_link_1);
    }

    // --- Update U_1(n) link (y-direction) ---
    // This link is part of two plaquettes: P(n) and P(n-x)
    let n_xm1 = vec2<u32>((x - 1u + L) % L, y);
    let n_xm1_yp1 = vec2<u32>((x - 1u + L) % L, (y + 1u) % L);

    // Staple from P(n) = U0(n) U1(n+x) U0(n+y)^-1 U1(n)^-1
    // Action part is Re(U1(n)^-1 * Staple), so we use Staple^dagger
    // Staple^dagger is V_right_dag = U0(n+y) U1(n+x)^-1 U0(n)^-1
    let staple_right = complex_mul(complex_mul(get_link(n_yp1, 0u), complex_conj(get_link(n_xp1, 1u))), complex_conj(get_link(site_coord, 0u)));
    
    // Staple from P(n-x) = U0(n-x) U1(n) U0(n-x+y)^-1 U1(n-x)^-1
    // The staple is V_left = U0(n-x+y)^-1 U1(n-x)^-1 U0(n-x)
    let staple_left = complex_mul(complex_mul(complex_conj(get_link(n_xm1, 0u)), get_link(n_xm1, 1u)), get_link(n_xm1_yp1, 0u));

    let staple_sum_2 = staple_right + staple_left;
    let old_link_2 = get_link(site_coord, 1u);
    let old_action_2 = dot(staple_sum_2, old_link_2);

    let rand_angle_2 = (pcg(&site_rng_state) - 0.5) * 2.0 * PI;
    let new_link_2 = vec2(cos(rand_angle_2), sin(rand_angle_2));
    let new_action_2 = dot(staple_sum_2, new_link_2);

    let dS_2 = new_action_2 - old_action_2;
    if (dS_2 > 0.0 || exp(params.beta * dS_2) > pcg(&site_rng_state)) {
        set_link(site_coord, 1u, new_link_2);
    }

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
