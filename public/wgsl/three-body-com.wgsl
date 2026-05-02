struct Particle {
    position : vec4<f32>,
    scale    : vec4<f32>,
    color    : vec4<f32>,
}

struct Env {
    time   : f32,
    tick   : f32,
    random : f32,
    filler2: f32,
}

struct Uniforms {
    env: Env,
}

const PI: f32 = 3.14159265359;

// 符号なし整数のハッシュ関数
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 0.0 ~ 1.0 の f32 を返す関数
fn rand(seed: u32) -> f32 {
    let h = pcg_hash(seed + u32(uniforms.env.random * 1000));
    // 32ビットの最大値で割って 0.0 ~ 1.0 に変換
    return f32(h) / f32(0xffffffffu);
}

fn calcAcceleration(pos1: vec3<f32>, pos2: vec3<f32>) -> vec3<f32> {
    const G = 0.0001;

    let diff     = pos1 - pos2;
    let distSq   = dot(diff, diff);
    let forceMag = G / (distSq + 1e-6);
    let force    = forceMag * normalize(diff);

    return force;
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> particlesA : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesB : array<Particle>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    const body_scale = 0.1;

    var global_idx = GlobalInvocationID.x;

    let array_length : u32 = arrayLength(&particlesA);
    if(array_length <= global_idx){
        return;
    }

    let body_trail_length = array_length / 3;
    let body_idx          = global_idx / body_trail_length;
    let local_idx         = global_idx % body_trail_length;

    var pos   = vec3<f32>(0.0, 0.0, 0.0);
    var scale = vec3<f32>(0.0, 0.0, 0.0);
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let tick = u32(uniforms.env.tick);

    if(3 <= body_idx){

    }
    else if(tick == 0){

        if (local_idx == 0) {
            return;
        }
        
        if(local_idx == 1){

            pos.x = mix(-4.0, 4.0, rand(global_idx));
            pos.y = mix(-4.0, 4.0, rand(global_idx + 1));
            pos.z = mix(-1.0, 1.0, rand(global_idx + 2));

            scale = body_scale * vec3<f32>(1.0, 1.0, 1.0);
            color = vec4<f32>(0.5, 0.5, 1.0, 1.0);

            let c = vec3<f32>(rand(global_idx + 3), rand(global_idx + 4), rand(global_idx + 5)) - 0.5;

            // let th = atan2(pos.y, pos.x) + 0.5 * PI;
            // let vel = 0.005 * vec3<f32>(cos(th), sin(th), 0.0);
            let vel = -0.005 * c * pos.xyz;

            particlesB[global_idx - 1].position = vec4<f32>(vel, 1.0);
            particlesB[global_idx - 1].scale    = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            particlesB[global_idx - 1].color    = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        else if(2 <= local_idx){

            color = vec4<f32>(0.7, 0.7, 0.7, 1.0);
        }
    }
    else if(tick == 1){
        // if this is the 1st or 2nd time

        let particle = particlesA[global_idx];
        pos   = particle.position.xyz;
        scale = particle.scale.xyz;
        color = particle.color;
    }
    else{
        // if this is not the 1st and 2nd time

        if(3 <= body_idx){
            return;
        }

        if (local_idx == 0) {
            return;
        }
        else if(local_idx == 1){
            // if this is a body.

            var prev_pos = particlesA[global_idx].position.xyz;
            var prev_vel = particlesA[global_idx - 1].position.xyz;

            // Carry over scale and color from the previous frame to prevent the body from disappearing.
            scale = particlesA[global_idx].scale.xyz;
            color = particlesA[global_idx].color;

            let i1 = (body_idx + 1) % 3;
            let i2 = (body_idx + 2) % 3;

            let pos1 = particlesA[i1 * body_trail_length + 1].position.xyz;
            let pos2 = particlesA[i2 * body_trail_length + 1].position.xyz;

            let acc1 = calcAcceleration(pos1, prev_pos);
            let acc2 = calcAcceleration(pos2, prev_pos);

            let velocity = prev_vel + acc1 + acc2;

            particlesB[global_idx - 1].position = vec4<f32>(velocity, 1.0);

            pos = prev_pos + velocity;
        }
        else{
            // if this is a trail.

            if(tick % 5 == 0){
                // copy the previous data.
                let particle = particlesA[global_idx - 1];
                pos   = particle.position.xyz;

                // The scale becomes smaller.
                scale = 0.99 * particle.scale.xyz;

                // The color becomes dimmer.
                // color = particle.color;
                // color.w *= 0.95;
            }
            else{
                let particle = particlesA[global_idx];
                pos   = particle.position.xyz;
                scale = particle.scale.xyz;
                color = particle.color;
            }
            color = vec4<f32>(0.7, 0.7, 0.7, 1.0);
        }
    }

    particlesB[global_idx].position = vec4<f32>(pos, 1.0);
    particlesB[global_idx].scale    = vec4<f32>(scale, 1.0);
    particlesB[global_idx].color    = color;
}
