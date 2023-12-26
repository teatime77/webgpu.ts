struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat3x3<f32>,

    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
    lightingDirection : vec3<f32>
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(3) color : vec4<f32>,
}

@vertex
fn vert_main(
  @location(0) a_particlePos : vec4<f32>,
  @location(1) a_particleVel : vec4<f32>,
  @location(2) a_pos  : vec3<f32>,
  @location(3) a_norm : vec3<f32>
) -> VertexOutput {
  var output : VertexOutput;

  output.position = uniforms.viewMatrix * vec4(0.2 * a_pos + a_particlePos.xyz, 1.0);

  // let material  = clamp(a_particleVel, vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0) );
  let material  = vec3(1.0, 1.0, 1.0);
  let brightness = max(dot(a_norm, uniforms.lightingDirection), 0.0);

  output.color = vec4(brightness * material, 1.0);

  return output;
}

@fragment
fn frag_main(@location(3) color : vec4<f32>) -> @location(0) vec4<f32> {
  return color;
}