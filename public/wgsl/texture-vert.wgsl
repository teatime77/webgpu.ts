struct Uniforms {
    projectionMatrix : mat4x4<f32>,
    viewMatrix : mat4x4<f32>,
    worldMatrix : mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
}

@vertex
fn main(
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>  
) -> VertexOutput {

    var output : VertexOutput;
    output.Position = uniforms.worldMatrix * position;
    output.fragUV = uv;
    
    return output;
}
