struct Uniforms {
    viewMatrix: mat4x4<f32>,
    normMatrix: mat4x4<f32>,
    materialColor: vec4<f32>,
    ambientColor: vec4<f32>,
    directionalColor: vec4<f32>,
    lightingDirection: vec4<f32>,
    lightPosition: vec4<f32>,
    cameraPosition: vec4<f32>,
    env: vec4<f32>, // env.x is time in ms
    shapeInfo: vec4<f32>,
    gridSize: vec4<f32>, // [num_circles_theta, num_circles_phi, 0, 0]
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

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

@vertex
fn main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32
) -> VertexOutput {
    let num_segments = 64.0;
    let pi = 3.1415926535;

    // 1. Map instance_index to (theta, phi) on the S2 sphere
    let num_theta = 30.0;
    let num_phi = 40.0;
    let phi_idx = f32(instance_idx % u32(num_phi));
    let theta_idx = f32(instance_idx / u32(num_phi));

    let theta = (theta_idx + 0.5) * pi / num_theta; // (0..pi)
    let phi = phi_idx * 2.0 * pi / num_phi;         // (0..2pi)

    // 2. Map vertex_index to psi (parameter along the circle fiber)
    let psi = f32(vertex_idx) * 2.0 * pi / num_segments;

    // 3. Inverse Hopf map to get a point on S3 in R4
    let p0 = vec2<f32>(cos((psi + phi) / 2.0), sin((psi + phi) / 2.0));
    let p1 = vec2<f32>(cos((psi - phi) / 2.0), sin((psi - phi) / 2.0));
    var q = vec4<f32>(
        cos(theta / 2.0) * p0.x,
        cos(theta / 2.0) * p0.y,
        sin(theta / 2.0) * p1.x,
        sin(theta / 2.0) * p1.y
    );

    // 4. Animate by rotating in 4D
    let time = u.env.x * 0.0005;
    let cos_t = cos(time);
    let sin_t = sin(time);
    let y0 = q.y;
    let x1 = q.z;
    q.y = y0 * cos_t - x1 * sin_t;
    q.z = y0 * sin_t + x1 * cos_t;

    // 5. Stereographic projection from S3 to R3
    let w = 1.0 - q.w;
    let world_pos = vec3<f32>(q.x / w, q.y / w, q.z / w) * 2.0;

    // 6. Transform to clip space and set color
    let hue = phi / (2.0 * pi);
    return VertexOutput(u.viewMatrix * vec4<f32>(world_pos, 1.0), vec4<f32>(hsv2rgb(hue, 0.8, 1.0), 1.0));
}