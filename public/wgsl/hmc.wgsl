// Uniforms that are constant for the entire simulation
struct HMCParams {
    // for leapfrog simulation
    epsilon: f32,
    L: u32,
    // for data size
    N: u32,
    num_chains: u32,
    seed: u32,
};

@group(0) @binding(0) var<uniform> params: HMCParams;
@group(0) @binding(1) var<storage, read> data_x1: array<f32>;
@group(0) @binding(2) var<storage, read> data_x2: array<f32>;
@group(0) @binding(3) var<storage, read> data_y: array<f32>;

// A helper function to calculate potential energy for a given theta vector.
// This is the core of the statistical model.
fn potential_energy(theta: vec4<f32>) -> f32 {
    let beta0 = theta[0];
    let beta1 = theta[1];
    let beta2 = theta[2];
    let log_sigma = theta[3];
    let sigma = exp(log_sigma);

    // Likelihood calculation (sum over N data points)
    var ll: f32 = 0.0;
    for (var i: u32 = 0u; i < params.N; i = i + 1u) {
        let mu = beta0 + beta1 * data_x1[i] + beta2 * data_x2[i];
        ll = ll - (log(sigma) + 0.5 * log(2.0 * 3.1415926535) + ((data_y[i] - mu) * (data_y[i] - mu)) / (2.0 * sigma * sigma));
    }

    // Prior calculation
    let prior_b0 = - (beta0 * beta0) / (2.0 * 100.0 * 100.0);
    let prior_b1 = - (beta1 * beta1) / (2.0 * 100.0 * 100.0);
    let prior_b2 = - (beta2 * beta2) / (2.0 * 100.0 * 100.0);
    let prior = prior_b0 + prior_b1 + prior_b2;

    // Potential energy is the negative log posterior
    return -(prior + ll);
}

// A helper function to calculate the gradient of the potential energy
// using the much faster analytical method.
fn get_gradient_analytical(q: vec4<f32>) -> vec4<f32> {
    let beta0 = q[0];
    let beta1 = q[1];
    let beta2 = q[2];
    let log_sigma = q[3];
    let sigma = exp(log_sigma);
    let sigma2 = sigma * sigma;

    var sum_err_div_sigma2: f32 = 0.0;
    var sum_err_x1_div_sigma2: f32 = 0.0;
    var sum_err_x2_div_sigma2: f32 = 0.0;
    var sum_err2_div_sigma2: f32 = 0.0;

    for (var i: u32 = 0u; i < params.N; i = i + 1u) {
        let mu = beta0 + beta1 * data_x1[i] + beta2 * data_x2[i];
        let err = data_y[i] - mu;
        let err_div_sigma2 = err / sigma2;
        sum_err_div_sigma2 = sum_err_div_sigma2 + err_div_sigma2;
        sum_err_x1_div_sigma2 = sum_err_x1_div_sigma2 + err_div_sigma2 * data_x1[i];
        sum_err_x2_div_sigma2 = sum_err_x2_div_sigma2 + err_div_sigma2 * data_x2[i];
        sum_err2_div_sigma2 = sum_err2_div_sigma2 + err * err / sigma2;
    }

    // Gradient of U = -ll - prior is dU = -d(ll) - d(prior)
    // Partial derivatives of the negative log-likelihood (-ll)
    let grad_ll_b0 = -sum_err_div_sigma2;
    let grad_ll_b1 = -sum_err_x1_div_sigma2;
    let grad_ll_b2 = -sum_err_x2_div_sigma2;
    let grad_ll_log_sigma = f32(params.N) - sum_err2_div_sigma2;

    // Partial derivatives of the negative prior (-prior)
    let grad_prior_b0 = beta0 / 10000.0;
    let grad_prior_b1 = beta1 / 10000.0;
    let grad_prior_b2 = beta2 / 10000.0;

    return vec4<f32>(
        grad_ll_b0 + grad_prior_b0,
        grad_ll_b1 + grad_prior_b1,
        grad_ll_b2 + grad_prior_b2,
        grad_ll_log_sigma // prior for log_sigma is 0
    );
}

// --- Pseudo-Random Number Generator (PRNG) ---
// A simple PCG hash-based generator to run on the GPU.
var<private> rand_seed: u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_init(invocation_id: u32, seed_offset: u32) {
    rand_seed = pcg_hash(invocation_id + params.seed + seed_offset);
}

fn rand_u32() -> u32 {
    rand_seed = pcg_hash(rand_seed);
    return rand_seed;
}

fn rand_f32() -> f32 {
    return f32(rand_u32()) / 4294967295.0; // max u32
}

// Box-Muller transform for generating normally distributed random numbers.
fn rand_normal() -> vec2<f32> {
    // Ensure u1 is not zero to avoid log(0)
    let u1 = max(rand_f32(), 1.17549435e-38); // Smallest positive f32 value
    let u2 = rand_f32();
    let r = sqrt(-2.0 * log(u1));
    let a = 2.0 * 3.1415926535 * u2;
    return vec2<f32>(r * cos(a), r * sin(a));
}

// --- HMC Step Kernel ---
// This single kernel calculates the current and new potential energies
// and performs the leapfrog integration in one dispatch.

@group(1) @binding(0) var<storage, read> q_in: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> q_out: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> accepted_out: atomic<u32>;
@group(1) @binding(3) var<storage, read_write> rand_state: array<u32>;

@compute @workgroup_size(64)
fn hmc_step_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chain_index = global_id.x;

    if (chain_index >= params.num_chains) {
        return;
    }

    // Initialize the PRNG for this thread using its unique state.
    rand_init(chain_index, rand_state[chain_index]);

    // --- Generate all random numbers for this step on the GPU ---
    let p_rand1 = rand_normal();
    let p_rand2 = rand_normal();
    let current_p = vec4<f32>(p_rand1.x, p_rand1.y, p_rand2.x, p_rand2.y);
    let acceptance_rand = rand_f32();

    // --- Part 1: Leapfrog Integration ---
    let U_current = potential_energy(q_in[chain_index]);
    var q_new = q_in[chain_index];
    var p_new = current_p;
    
    var grad = get_gradient_analytical(q_new);
    p_new = p_new - 0.5 * params.epsilon * grad;
    for (var i: u32 = 0u; i < params.L; i = i + 1u) {
        q_new = q_new + params.epsilon * p_new;
        if (i != params.L - 1u) {
            grad = get_gradient_analytical(q_new);
            p_new = p_new - params.epsilon * grad;
        }
    }
    grad = get_gradient_analytical(q_new);
    p_new = p_new - 0.5 * params.epsilon * grad;
    p_new = -p_new; // Invert momentum for reversibility

    // --- Part 2: Acceptance Check ---
    let U_new = potential_energy(q_new);
    let K_current = 0.5 * dot(current_p, current_p);
    let K_new = 0.5 * dot(p_new, p_new);
    let log_alpha = (U_current + K_current) - (U_new + K_new);

    // Use the random number for this chain to decide acceptance
    if (log(acceptance_rand) < log_alpha) {
        q_out[chain_index] = q_new;
        atomicAdd(&accepted_out, 1u); // Atomically increment total acceptance count
    } else {
        q_out[chain_index] = q_in[chain_index];
    }

    // Store the updated PRNG state for the next iteration.
    rand_state[chain_index] = rand_seed;
}