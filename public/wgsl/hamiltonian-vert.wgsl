struct Uniforms {
    viewMatrix: mat4x4<f32>,
    normMatrix: mat4x4<f32>,
    materialColor: vec4<f32>,
    ambientColor: vec4<f32>,
    directionalColor: vec4<f32>,
    lightingDirection: vec3<f32>,
    lightPosition: vec3<f32>,
    cameraPosition: vec3<f32>,
    gridSize: vec2<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) fragColor: vec4<f32>,
};

fn rgba(theta_arg : f32) -> vec4<f32>{
    const pi = 3.14159265359;

    var theta = theta_arg;
    if(theta < 0.0){
        theta += 2.0 * pi;
    }

    var r : f32 = 0.0;
    var g : f32 = 0.0;
    var b : f32 = 0.0;

    // Map theta (0 to 2*pi) to t (0 to 3)
    var t = theta * 3.0 / (2.0 * pi);
    if(t <= 1.0){
        // Red to Green
        r = (1.0 - t);
        g = t;
    } else if(t <= 2.0){
        // Green to Blue
        t -= 1.0;
        g = (1.0 - t);
        b = t;
    } else {
        // Blue to Red
        t -= 2.0;
        b = (1.0 - t);
        r = t;
    }

    return vec4<f32>(r, g, b, 1.0);
}

@vertex
fn main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let grid = uniforms.gridSize;
    let num_cols = grid.x;
    let num_rows = grid.y;

    let col_idx = f32(vertex_index / 2u);
    let row_idx = f32(instance_index);

    let is_top = (vertex_index % 2u) == 0u;
    let q_row_idx = select(row_idx, row_idx + 1.0, is_top);

    // Map indices to q and p coordinates in phase space
    // q in [-PI, PI], p in [-2.5, 2.5]
    let q_range = 2.0 * 3.14159;
    let p_range = 5.0;
    let u = col_idx / num_cols;
    let v = q_row_idx / num_rows;

    let q = -q_range/2.0 + u * q_range;
    let p = -p_range/2.0 + v * p_range;

    // Calculate Hamiltonian H = 1/2 * p^2 - cos(q)
    let h = 0.5 * p * p - cos(q);

    var pos = vec3<f32>(q, p, h);

    // Calculate normal for lighting
    // Normal = (-dH/dq, -dH/dp, 1) = (-sin(q), -p, 1)
    let normal = normalize(vec3<f32>(-sin(q), -p, 1.0));

    // Calculate color based on Hamiltonian height (h)
    // h is in approx [-1, 4.125]. Normalize to [0, 1].
    let h_norm = (h + 1.0) / 5.125;
    // Map to a color spectrum (Red -> Green -> Blue)
    let color = rgba(h_norm * 4.2); // 4.2 is approx 2/3 of 2*PI

    var output: VertexOutput;
    output.Position = uniforms.viewMatrix * vec4<f32>(pos, 1.0);
    output.worldPos = pos;
    output.worldNormal = (uniforms.normMatrix * vec4<f32>(normal, 0.0)).xyz;
    output.fragColor = color;
    return output;
}