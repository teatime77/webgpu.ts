const L = 32u; // Lattice size
const L_squared = L * L;

struct SimParams {
    beta: f32,
    // A value that changes each frame to pick a different link subset to update.
    // This avoids race conditions where adjacent threads update neighboring links.
    // Can be 0, 1, 2, 3 for (even/odd x, even/odd y) subsets.
    update_subset: u32, 
};

@group(0) @binding(0) var<uniform> params: SimParams;

// Link variables U are stored as vec2<f32>(cos(theta), sin(theta))
// Layout: 2*L*L array.
// First L*L for horizontal links (mu=0)
// Second L*L for vertical links (mu=1)
@group(0) @binding(1) var<storage, read_write> links: array<vec2<f32>>;

// State for random number generator
@group(0) @binding(2) var<storage, read_write> rng_state: array<u32>;

// Buffer to store results of measurements
@group(0) @binding(3) var<storage, read_write> plaquette_results: array<f32>;

// --- Complex number helpers ---
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

// --- Lattice and Link helpers (with periodic boundary conditions) ---
fn get_link_idx(x: u32, y: u32, mu: u32) -> u32 {
    return mu * L_squared + (y % L) * L + (x % L);
}

fn get_link(x: u32, y: u32, mu: u32) -> vec2<f32> {
    return links[get_link_idx(x, y, mu)];
}

// --- Random number helpers ---
fn pcg(state_in: u32) -> vec2<u32> {
    let state = state_in * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let result = (word >> 22u) ^ word;
    return vec2<u32>(result, state);
}

fn random_float(state: ptr<function, u32>) -> f32 {
    let res = pcg(*state);
    *state = res.y;
    return f32(res.x) / 4294967295.0;
}

// Initialize links to a random configuration (hot start)
@compute @workgroup_size(8, 8, 1)
fn init_hot(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L || id.y >= L) { return; }

    let idx = id.y * L + id.x;
    var state = rng_state[idx];

    let pi = 3.14159265;
    // Init horizontal link
    let angle_h = (random_float(&state) - 0.5) * 2.0 * pi;
    links[get_link_idx(id.x, id.y, 0u)] = vec2<f32>(cos(angle_h), sin(angle_h));
    
    // Init vertical link
    let angle_v = (random_float(&state) - 0.5) * 2.0 * pi;
    links[get_link_idx(id.x, id.y, 1u)] = vec2<f32>(cos(angle_v), sin(angle_v));

    rng_state[idx] = state;
}

// One Metropolis update step
@compute @workgroup_size(8, 8, 1)
fn metropolis_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= L || y >= L) { return; }

    // Update only a subset of links to avoid race conditions.
    if ((x % 2u) != (params.update_subset % 2u) || (y % 2u) != (params.update_subset / 2u)) {
        return;
    }

    let r_idx = y * L + x;
    var state = rng_state[r_idx];

    // --- Update horizontal link U_0(x,y) ---
    // The "staple" is the product of the 3 other links in the two plaquettes sharing U_0(x,y)
    let staple_h1 = cmul(get_link(x + 1u, y, 1u), cconj(get_link(x, y + 1u, 0u)));
    let staple_h1_full = cmul(staple_h1, cconj(get_link(x, y, 1u)));

    let staple_h2 = cmul(cconj(get_link(x, y - 1u, 0u)), get_link(x, y - 1u, 1u));
    let staple_h2_full = cmul(staple_h2, cconj(get_link(x + 1u, y - 1u, 1u)));

    let staple_h = staple_h1_full + staple_h2_full;

    let old_link_h = get_link(x, y, 0u);
    // Propose a new link by rotating the old one by a small random angle.
    // The width of the change (2.0 here) should be tuned for ~50% acceptance rate.
    let d_theta = (random_float(&state) - 0.5) * 2.0;
    let d_link = vec2<f32>(cos(d_theta), sin(d_theta));
    let new_link_h = cmul(old_link_h, d_link);

    // The change in action is proportional to Re( (U_new - U_old) * Staple )
    let delta_S_h = params.beta * (cmul(new_link_h, staple_h).x - cmul(old_link_h, staple_h).x);

    if (exp(delta_S_h) > random_float(&state)) {
        links[get_link_idx(x, y, 0u)] = new_link_h;
    }

    // --- Update vertical link U_1(x,y) ---
    let staple_v1 = cmul(get_link(x, y + 1u, 0u), cconj(get_link(x + 1u, y, 1u)));
    let staple_v1_full = cmul(staple_v1, cconj(get_link(x, y, 0u)));

    let staple_v2 = cmul(cconj(get_link(x - 1u, y, 1u)), get_link(x - 1u, y, 0u));
    let staple_v2_full = cmul(staple_v2, cconj(get_link(x - 1u, y + 1u, 0u)));
    
    let staple_v = staple_v1_full + staple_v2_full;

    let old_link_v = get_link(x, y, 1u);
    let d_theta_v = (random_float(&state) - 0.5) * 2.0;
    let d_link_v = vec2<f32>(cos(d_theta_v), sin(d_theta_v));
    let new_link_v = cmul(old_link_v, d_link_v);

    let delta_S_v = params.beta * (cmul(new_link_v, staple_v).x - cmul(old_link_v, staple_v).x);

    if (exp(delta_S_v) > random_float(&state)) {
        links[get_link_idx(x, y, 1u)] = new_link_v;
    }

    rng_state[r_idx] = state;
}

// --- Observable: Average Plaquette Value ---
@compute @workgroup_size(8, 8, 1)
fn measure_plaquette(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= L || y >= L) { return; }

    let u0 = get_link(x, y, 0u);
    let u1 = get_link(x + 1u, y, 1u);
    let u2 = cconj(get_link(x, y + 1u, 0u));
    let u3 = cconj(get_link(x, y, 1u));

    let plaquette = cmul(cmul(u0, u1), cmul(u2, u3));

    let idx = y * L + x;
    plaquette_results[idx] = plaquette.x; // Store Re(P)
}
