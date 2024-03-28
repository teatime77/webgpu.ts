struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat4x4<f32>,

    materialColor     : vec4<f32>,
    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
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
    @location(2) a_particlePos : vec4<f32>,
    @location(3) a_particleVel : vec4<f32>
) -> VertexOutput {

    var output : VertexOutput;

    output.Position = uniforms.viewMatrix * (vec4<f32>(position, 1.0) + vec4<f32>(a_particlePos.xyz, 1));

    var transformedNormal = uniforms.normMatrix * vec4<f32>(norm, 1.0);
    var directionalLightWeighting = max(dot(transformedNormal.xyz, uniforms.lightingDirection), 0.0);
    output.fragColor = uniforms.materialColor * (uniforms.ambientColor + directionalLightWeighting * uniforms.directionalColor);
    
    return output;
}
