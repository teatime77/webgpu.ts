struct Uniforms {
    viewMatrix: mat4x4<f32>,
    materialColor: vec4<f32>,
};

struct Instance { 
    position: vec4<f32>, 
    color   : vec4<f32> 
}

struct Vertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> vertices: array<Vertex>;

@vertex
fn main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let mesh_pos = vertices[vertex_index].pos.xyz;

    let instance  = instances[instance_index];

    let final_pos = instance.position.xyz + mesh_pos * 0.03;

    var output: VertexOutput;
    output.position = uniforms.viewMatrix * vec4<f32>(final_pos, 1.0);
    output.color    = instance.color;

    return output;
}