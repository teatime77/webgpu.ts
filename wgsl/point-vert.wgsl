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
    @location(0) vertPos:  vec3<f32>,
    @location(1) vertNorm: vec3<f32>,
    @location(2) meshPos : vec4<f32>,
    @location(3) meshVec : vec4<f32>,
    @builtin(vertex_index) vertex_index : u32
) -> VertexOutput {


    var pos = vec4<f32>(meshPos.xyz + meshVec.xyz, 1.0);

    var output : VertexOutput;

    output.Position = uniforms.viewMatrix * vec4<f32>(pos.xyz, 1.0);

    output.fragColor = uniforms.materialColor * uniforms.ambientColor;
    
    return output;
}
