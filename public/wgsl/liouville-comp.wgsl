struct Env {
    time   : f32,
    tick   : f32,
    filler1: f32,
    filler2: f32,
};

struct Uniforms {
    env: Env,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input_particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_particles: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_particles = arrayLength(&input_particles);
    if (global_id.x >= num_particles) {
        return;
    }
    let idx = global_id.x;

    if (u.env.tick == 0.0) {
        // --- On the first frame (tick=0), calculate initial positions ---
        // This value must match the configuration in liouville.ts
        let points_side = 23.0;

        let i = f32(idx);
        let x_idx = i % points_side;
        let y_idx = floor(i / points_side);

        // Place a 1x1 square centered at (q=1.0, p=0.0)
        let q = 1.0 + (x_idx / (points_side - 1.0) - 0.5) * 1.0;
        let p = 0.0 + (y_idx / (points_side - 1.0) - 0.5) * 1.0;

        output_particles[idx] = vec4<f32>(q, p, 0.0, 1.0);
    } else {
        let p_in = input_particles[idx];
        var q = p_in.x;
        var p = p_in.y;

        // Symplectic Euler integration for H = 1/2 p^2 - cos(q)
        // dq/dt = p
        // dp/dt = -sin(q)
        let dt = 0.01; // Fixed time step for stability
        
        let p_new = p - dt * sin(q);
        let q_new = q + dt * p_new;

        // Wrap q to be in [-PI, PI]
        let pi = 3.14159265;
        var q_wrapped = q_new % (2.0 * pi);
        if (q_wrapped > pi) { q_wrapped = q_wrapped - 2.0 * pi; }
        else if (q_wrapped < -pi) { q_wrapped = q_wrapped + 2.0 * pi; }
        
        output_particles[idx] = vec4<f32>(q_wrapped, p_new, 0.0, 1.0);
    }
}