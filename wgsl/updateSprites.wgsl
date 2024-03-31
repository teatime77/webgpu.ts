struct Particle {
  meshPos : vec4<f32>,
  meshVec : vec4<f32>,
}
struct SimParams {
    env : vec4<f32>
}

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> particlesA : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB : array<Particle>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  const pi = 3.14159265359;

  var index = GlobalInvocationID.x;

  var pos = particlesA[index].meshPos;

  var theta : f32 = 2.0 * pi * f32(i32(params.env.x) % 4096) / 4096.0;

  // Write back
  particlesB[index].meshPos = pos;
  particlesB[index].meshVec = sin(theta) * vec4<f32>(0.0, 0.0, 1.0, 0.0);
}
