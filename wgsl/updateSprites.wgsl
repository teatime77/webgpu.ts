struct Particle {
  pos : vec4<f32>,
  vel : vec4<f32>,
}
struct SimParams {
  deltaT : f32,
  rule1Distance : f32,
  rule2Distance : f32,
  rule3Distance : f32,
  rule1Scale : f32,
  rule2Scale : f32,
  rule3Scale : f32,
}

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> particlesA : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB : array<Particle>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var index = GlobalInvocationID.x;

  var vPos = particlesA[index].pos;
  var vVel = particlesA[index].vel;

  vPos = vPos + (vVel * params.deltaT);
  vVel = vVel * params.deltaT;

  // Write back
  particlesB[index].pos = vPos;
  particlesB[index].vel = vVel;
}
