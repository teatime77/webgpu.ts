struct Particle {
    meshPos : vec4<f32>,
    meshVec : vec4<f32>,
}

struct SimParams {
    env : vec4<f32>
}

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> inEH : array<Particle>;
@group(0) @binding(2) var<storage, read_write> outEH : array<Particle>;
    
const PI : f32 = 3.14159265359;

const mu0 : f32 = 1.25663706212e-06;
const epsilon0 : f32 = 8.854187812799999e-12;

const sx = 16;
const sy = 16;
const sz = 1;
const K : f32 = 1.0 / 16.0;

const iE = 0;
const iH = 1;

fn getidx(eh : i32, i : i32, j : i32, k : i32) -> i32 {
    return eh + i * 2 + j * 2 * sx + k * (2 * sx * sy);
}

fn getvec(eh : i32, i : i32, j : i32, k : i32) -> vec3<f32> {
    return inEH[getidx(eh, i, j, k)].meshVec.xyz;
}

fn calcRot(flag : i32, E : vec3<f32>, H : vec3<f32>, i : i32, j : i32, k : i32) -> vec3<f32>{
    if(flag == 1){
        var Ei = E;
        var Ej = E;
        var Ek = E;

        if(i + 1 < sx){

            Ei = getvec(iE, i + 1, j, k);
        }

        if(j + 1 < sy){

            Ej = getvec(iE, i, j + 1, k);
        }

        if(k + 1 < sz){

            Ek = getvec(iE, i, j, k + 1);
        }

        var rx = (Ej.z - E.z) - (Ek.y - E.y);
        var ry = (Ek.x - E.x) - (Ei.z - E.z);
        var rz = (Ei.y - E.y) - (Ej.x - E.x);

        return vec3<f32>(rx, ry, rz);
    }
    else{

        var Hi = H; 
        var Hj = H; 
        var Hk = H; 

        if(0 <= i - 1){

            Hi = getvec(iH, i - 1, j, k);
        }

        if(0 <= j - 1){

            Hj = getvec(iH, i, j - 1, k);
        }

        if(0 <= k - 1){

            Hk = getvec(iH, i, j, k - 1);
        }

        var rx = (H.z - Hj.z) - (H.y - Hk.y);
        var ry = (H.x - Hk.x) - (H.z - Hi.z);
        var rz = (H.y - Hi.y) - (H.x - Hj.x);

        return vec3<f32>(rx, ry, rz);
    }
}

@compute @workgroup_size(8,8,2)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    // var L = 3.2 / f32(max(sx, max(sy,sz)));
    var L = 10.0 / f32(max(sx, max(sy,sz)));
    // var K = f32(@{K});

    var col  = i32(GlobalInvocationID.x);
    var row  = i32(GlobalInvocationID.y);
    var eh   = i32(GlobalInvocationID.z);
    var dep : i32 = 0;

    var x = f32(col - sx/2) * L;
    var y = f32(row - sy/2) * L;
    var z = f32(dep - sz/2) * L;
    var E : vec3<f32>;
    var H : vec3<f32>;

    var tick = i32(params.env.y);

    if(tick == 0){
        E = vec3<f32>(0.0, 0.0, 0.0);
        H = vec3<f32>(0.0, 0.0, 0.0);
    }
    else{    
        E = getvec(iE, col, row, dep);
        H = getvec(iH, col, row, dep);
    
        if(tick % 2 == 0){

            if(eh == iE){

                var rotH = calcRot(0, E, H, col, row, dep);
                E = E + K * rotH;
            }
        }
        else{
            if(eh == iH){

                var rotE = calcRot(1, E, H, col, row, dep);
                H = H - rotE;
            }
        }
    }

    if(eh == iE){

        if(col == sx / 2 && row == sy / 2 && dep == sz / 2){
            E.z += 0.01 * sin(2.0 * PI * f32(tick) / 200.0);
        }
    }

    var index = getidx(eh, col, row, dep);

    outEH[index].meshPos = vec4<f32>(x, y, z, 0.0);

    if(eh == iE){

        outEH[index].meshVec   = vec4<f32>(E, 0.0);
    }
    else{

        outEH[index].meshVec   = vec4<f32>(H, 0.0);
    }
}