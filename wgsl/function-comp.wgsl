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
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    const pi = 3.14159265359;

    var index = GlobalInvocationID.x;
    var tick  = i32(params.env.y);
    if(tick == 0){
      var theta_i = index / 20;
      var phi_i   = index % 20;

      var theta = pi * f32(theta_i) / 20.0;
      var z     = cos(theta);
      var r     = sin(theta);

      var phi = 2.0 * pi * f32(phi_i) / 20.0;
      var x   = r * cos(phi);
      var y   = r * sin(phi);

      particlesB[index].meshPos = 6.0 * vec4<f32>(x, y, z, 0.0);
      particlesB[index].meshVec = vec4<f32>(x, y, z, 0.0);
    }
    else{


      var pos = particlesA[index].meshPos;
      var v1 : vec3<f32> = normalize(pos.xyz);

      // var alpha : f32 = 2.0 * pi * f32(i32(params.env.x) % 3000) / 3000.0;
      // var dist = sin(alpha);

      // Write back
      particlesB[index].meshPos = pos;
      particlesB[index].meshVec = vec4<f32>(v1, 0.0);
      // particlesB[index].meshVec = dist * vec4<f32>(v1, 0.0);
    }
}
