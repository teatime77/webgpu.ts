struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat3x3<f32>,

    ambientColor      : vec3<f32>,
    directionalColor  : vec3<f32>,
    lightingDirection : vec3<f32>
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec4<f32>,
}

@vertex
fn main(
    @location(0) position: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) pos: vec2<f32>
) -> VertexOutput {

    var output : VertexOutput;

    output.Position = uniforms.viewMatrix * (vec4<f32>(position, 1.0) + vec4<f32>(pos.x, pos.y, 0, 1));
    output.fragColor = vec4<f32>(norm, 1.0);
    
    return output;
}
