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
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    const pi = 3.14159265359;
    const sz = 16;
    const r1 = 5.0;


    var ix = i32(GlobalInvocationID.x);
    var iy = i32(GlobalInvocationID.y);
    var iz = i32(GlobalInvocationID.z);
    var index = sz * sz * iz + sz * iy + ix;

    var tick  = i32(params.env.y);

    var theta = pi * f32(iy) / f32(sz);
    var z     = r1 * cos(theta);
    var r     = r1 * sin(theta);

    r += f32(iz);

    var phi = 2.0 * pi * f32(ix) / f32(sz);
    var x   = r * cos(phi);
    var y   = r * sin(phi);



    var p : vec3<f32>;

    if(tick == 0){
        p = vec3<f32>(x, y, z);
    }
    else{

        p = particlesA[index].meshPos.xyz;
    }

    var d = 2.0;
    var o1 = vec3<f32>( 0.0, 0.0, -d);
    var o2 = vec3<f32>( 0.0, 0.0,  d);

    var v1 = p - o1;
    var v2 = p - o2;

    var l1 = length(v1);
    var l2 = length(v2);

    var v = 10.0 * v1 / (l1 * l1 * l1) - 10.0 * v2 / (l2 * l2 * l2);

    if(tick == 0){
      

        particlesB[index].meshPos = vec4<f32>(p, 0.0);
        particlesB[index].meshVec = vec4<f32>(v, 0.0);
    }
    else{


        // Write back
        particlesB[index].meshPos = vec4<f32>(p, 0.0);
        particlesB[index].meshVec = vec4<f32>(v, 0.0);
    }
}
