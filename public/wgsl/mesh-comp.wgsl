struct Particle {
    position : vec4<f32>,
    scale    : vec4<f32>,
    color    : vec4<f32>,
}

struct Env {
    time   : f32,
    tick   : f32,
    filler1: f32,
    filler2: f32,
}

struct Uniforms {
    env: Env,
}

fn rgba(theta_arg : f32) -> vec4<f32>{
    const pi      = 3.14159265359;

    var theta = theta_arg;
    while(theta < 0){
        theta += 2 * pi;
    }
    while(2 * pi < theta){
        theta -= 2 * pi;
    }

    var r : f32 = 0;
    var g : f32 = 0;
    var b : f32 = 0;

    var t = theta * 3.0 / (2.0 * pi);
    if(t <= 1.0){
        r = (1.0 - t);
        g = t;
    }
    else{
        t -= 1.0;
        if(t <= 1.0){
            g = (1.0 - t);
            b = t;
        }
        else{
            t -= 1.0;

            b = (1.0 - t);
            r = t;
        }
    }

    return vec4<f32>(r, g, b, 1.0);
}


@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> particlesA : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB : array<Particle>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    const pi = 3.14159265359;

    var index = GlobalInvocationID.x;
    var tick  = i32(uniforms.env.tick);
    if(tick == 0){
        var theta_i = index / 20;
        var phi_i   = index % 20;

        var theta = pi * f32(theta_i) / 10.0;
        var z     = cos(theta);
        var r     = sin(theta);

        var phi = 2.0 * pi * f32(phi_i) / 20.0;
        var x   = r * cos(phi);
        var y   = r * sin(phi);

        particlesB[index].position = 24.0 * vec4<f32>(x, y, z, 1.0);
        particlesB[index].scale    = vec4<f32>(4.0, 4.0, 4.0, 1.0);
        particlesB[index].color    = rgba(theta);
    }
    else{

        particlesB[index] = particlesA[index];
    }
}
