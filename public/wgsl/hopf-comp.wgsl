struct Particle {
    position : vec4<f32>,
    scale    : vec4<f32>,
    color    : vec4<f32>,
};

// The framework provides this standard environment uniform block.
struct Env {
    time   : f32,
    tick   : f32,
    filler1: f32,
    filler2: f32,
}
struct Uniforms {
    env: Env,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

// The compute pipeline framework provides these two ping-pong buffers.
@group(0) @binding(1) var<storage, read> particlesA: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB: array<Particle>;

// Helper to create nice colors
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let m = i % 6.0;
    if (m == 0.0) { return vec3<f32>(v, t, p); }
    if (m == 1.0) { return vec3<f32>(q, v, p); }
    if (m == 2.0) { return vec3<f32>(p, v, t); }
    if (m == 3.0) { return vec3<f32>(p, q, v); }
    if (m == 4.0) { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let instance_idx = global_id.x;
    let num_particles = arrayLength(&particlesB);
    if (instance_idx >= num_particles) {
        return;
    }

    // On subsequent frames, just copy the data. The positions are static.
    if (u.env.tick > 0.0) {
        particlesB[instance_idx] = particlesA[instance_idx];
        return;
    }

    // --- On the first frame (tick == 0), calculate sphere positions ---
    // These values must match the configuration in hopf.ts
    let num_theta = 30.0;
    let num_phi = 40.0;
    let pi = 3.1415926535;

    // 1. Map instance_index to (theta, phi) on the S2 sphere
    let phi_idx = f32(instance_idx % u32(num_phi));
    let theta_idx = f32(instance_idx / u32(num_phi));
    let theta = (theta_idx + 0.5) * pi / num_theta; // (0..pi)
    let phi = phi_idx * 2.0 * pi / num_phi;         // (0..2pi)

    // 2. Calculate Cartesian coordinates for the point on the S2 sphere
    let radius = 2.1; // Slightly larger than the fiber projection to look nice
    let x = radius * sin(theta) * cos(phi);
    let y = radius * sin(theta) * sin(phi);
    let z = radius * cos(theta);

    // 3. Calculate color from the same parameter as the fiber
    let hue = phi / (2.0 * pi);
    let color = vec4<f32>(hsv2rgb(hue, 0.8, 1.0), 1.0);

    // 4. Write to the output buffer
    particlesB[instance_idx].position = vec4<f32>(x, y, z, 1.0);
    particlesB[instance_idx].scale = vec4<f32>(0.1, 0.1, 0.1, 1.0); // Small spheres
    particlesB[instance_idx].color = color;
}