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
struct Particles {
  particles : array<Particle>,
}
@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> particlesA : Particles;
@binding(2) @group(0) var<storage, read_write> particlesB : Particles;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var index = GlobalInvocationID.x;

  var vPos = particlesA.particles[index].pos;
  var vVel = particlesA.particles[index].vel;

  // Wrap around boundary
  if (3.0 < abs(vPos.x)) {
    vVel.x  = - sign(vPos.x) * abs(vVel.x);
  }
  if (3.0 < abs(vPos.y)) {
    vVel.y  = - sign(vPos.y) * abs(vVel.y);
  }
  if (vPos.z < 3.0) {
    vVel.z  = abs(vVel.z);
  }
  else if (6.0 < vPos.z) {
    vVel.z  = - abs(vVel.z);
  }

  // kinematic update
  vPos = vPos + (vVel * params.deltaT);

  // Write back
  particlesB.particles[index].pos = vPos;
  particlesB.particles[index].vel = vVel;
}
