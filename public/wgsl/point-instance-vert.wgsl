struct Uniforms {
    viewMatrix: mat4x4<f32>,
    materialColor: vec4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// This is the output of the compute shader
@group(0) @binding(1) var<storage, read> instance_positions: array<vec4<f32>>;

// This is the mesh for a single point (e.g. a small sphere)
struct Vertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
};
@group(0) @binding(2) var<storage, read> mesh_vertices: array<Vertex>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let mesh_pos = mesh_vertices[vertex_index].pos.xyz;

    // Get the instance position (q, p) from the compute shader output
    let instance_pos = instance_positions[instance_index].xyz;

    // Final position is mesh vertex, scaled, and translated to instance position
    let final_pos = instance_pos + mesh_pos * 0.03;

    var output: VertexOutput;
    output.position = uniforms.viewMatrix * vec4<f32>(final_pos, 1.0);
    output.color = uniforms.materialColor;
    return output;
}