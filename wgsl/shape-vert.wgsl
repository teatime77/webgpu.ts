struct Uniforms {
    viewMatrix        : mat4x4<f32>,

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
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>
) -> VertexOutput {

    var output : VertexOutput;
    output.Position = uniforms.viewMatrix * position;
    output.fragColor = color;
    
    return output;
}
