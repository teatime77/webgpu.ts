struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat4x4<f32>,

    materialColor     : vec4<f32>,
    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
    lightingDirection : vec4<f32>,
    env               : vec4<f32>,
    shapeInfo         : vec4<f32>
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec4<f32>,
}

@vertex
fn main(
    @location(0) position: vec3<f32>,
    @location(1) norm: vec3<f32>
) -> VertexOutput {

    var output : VertexOutput;
    output.Position = uniforms.viewMatrix * vec4<f32>(position, 1.0);

    var transformedNormal = uniforms.normMatrix * vec4<f32>(norm, 1.0);
    var directionalLightWeighting = max(dot(transformedNormal.xyz, uniforms.lightingDirection.xyz), 0.0);
    output.fragColor = uniforms.materialColor * (uniforms.ambientColor + directionalLightWeighting * uniforms.directionalColor);
    
    return output;
}
