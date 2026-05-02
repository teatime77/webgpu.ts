// File: ./wgsl/lgt_render.wgsl

// This shader takes an array of vortex charges and renders a colored quad for each.
@group(0) @binding(0) var<storage, read> viz_results: array<f32>;

struct RenderParams {
    viz_mode: u32, // 0 for plaquette, 1 for vortex
};
@group(0) @binding(1) var<uniform> render_params: RenderParams;

const L: f32 = 32.0; // Must match TS
const PI: f32 = 3.1415926535;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // --- Calculate quad position from instance_index ---
    let i = f32(instance_index % u32(L));
    let j = f32(floor(f32(instance_index) / L));
    let cell_size = 2.0 / L;
    let origin = vec2<f32>(-1.0 + i * cell_size, -1.0 + j * cell_size);

    // --- Define a unit quad (2 triangles) ---
    let quad_coords = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    );
    let pos = origin + quad_coords[vertex_index] * cell_size;

    // --- Determine color from vortex charge ---
    var color: vec4<f32>;

    if (render_params.viz_mode == 1u) { // Vortex mode
        let charge = bitcast<i32>(viz_results[instance_index]);
        if (charge == 1) {
            color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // Yellow for vortex
        } else if (charge == -1) {
            color = vec4<f32>(0.0, 1.0, 1.0, 1.0); // Cyan for anti-vortex
        } else {
            color = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Black for no vortex
        }
    } else {
        // Plaquette energy mode
        // The value is cos(theta), which ranges from -1 to 1.
        // Map this to a red-blue color scale.
        let plaquette_val = viz_results[instance_index];
        let r = 0.5 * (1.0 + plaquette_val);
        let b = 0.5 * (1.0 - plaquette_val);
        color = vec4<f32>(r, 0.0, b, 1.0);
    }

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
