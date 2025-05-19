struct Particle {
  meshPos : vec4<f32>,
  meshVec : vec4<f32>,
}
struct SimParams {
    env : vec4<f32>
}

struct Complex {
    real: f32,
    imag: f32,
};

fn comp(a: f32, b: f32) -> Complex {
    return Complex(a, b);
}

fn add(a: Complex, b: Complex) -> Complex {
    return Complex(a.real + b.real, a.imag + b.imag);
}

fn multiply(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

fn absC(c: Complex) -> f32 {
    return sqrt(c.real * c.real + c.imag * c.imag);
}

fn pow2(a: Complex) -> Complex {
    return multiply(a, a);
}

fn pow3(a: Complex) -> Complex {
    return multiply(pow2(a), a);
}

fn conjugate(a: Complex) -> Complex {
    return Complex(a.real, -a.imag);
}

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> particlesA : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB : array<Particle>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    const pi      = 3.14159265359;
    const sz      = 100;
    const max_xy  = 2.0;
    const step    = (2.0 * max_xy) / f32(sz);

    var index = GlobalInvocationID.x;
    var tick  = i32(params.env.y);
    if(tick == 0){
      var yi  = index / sz;
      var xi  = index % sz;
      var y   = -max_xy + step * f32(yi);
      var x   = -max_xy + step * f32(xi);

      var theta = x + y;
      var z     = sin(theta);

      particlesB[index].meshPos = vec4<f32>(x, y, z, 0.0);
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
