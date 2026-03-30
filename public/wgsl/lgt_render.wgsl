// This buffer holds the plaquette values calculated by the compute shader.
// It must have the same structure as the one in lgt_u1.wgsl.
@group(0) @binding(0) var<storage, read> plaquette_results: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) value: f32,
};

const L = 32u; // Must match const in lgt_u1.wgsl

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // --- Generate a quad for each instance ---
    // We define a single quad using 6 vertices.
    let pos = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    );

    // --- Map instance_index to a 2D grid coordinate ---
    let ix = f32(instance_index % L);
    let iy = f32(instance_index / L);

    // --- Calculate quad position in Normalized Device Coordinates (NDC) ---
    // Scale and shift the quad to fit on the screen in a grid.
    let scale = 2.0 / f32(L); // Scale from [0, L] range to [-1, 1] range
    let quad_pos = (pos[vertex_index] + vec2<f32>(ix, iy)) * scale - 1.0;

    var output: VertexOutput;
    // Flip the Y-axis because NDC's +Y is up, but our grid's +Y is down.
    output.position = vec4<f32>(quad_pos.x, -quad_pos.y, 0.0, 1.0);
    
    // Pass the plaquette value for this instance to the fragment shader.
    output.value = plaquette_results[instance_index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Map the plaquette value from its [-1, 1] range to a color.
    // A low value (disordered) will be blue.
    // A high value (ordered) will be red.
    let val = (input.value + 1.0) * 0.5; // Map to [0, 1]
    let color = vec3<f32>(val, 0.1, 1.0 - val); // A simple red-blue colormap
    return vec4<f32>(color, 1.0);
}
