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
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec4<f32>,
}

@vertex
fn main(
  @location(0) a_particlePos : vec4<f32>,
  @location(1) a_particleVel : vec4<f32>,
  @location(2) a_pos  : vec3<f32>,
  @location(3) a_norm : vec3<f32>
) -> VertexOutput {
  var output : VertexOutput;

  output.position = uniforms.viewMatrix * vec4(0.2 * a_pos + a_particlePos.xyz, 1.0);

  let brightness = max(dot(a_norm, uniforms.lightingDirection), 0.0);

  output.color = brightness * uniforms.materialColor;

  return output;
}

