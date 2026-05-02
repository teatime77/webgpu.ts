
struct Env {
    time   : f32,   // elapsed time in milli-seconds
    tick   : f32,   // execution counter.
    filler1: f32,
    filler2: f32,
}

struct SimParams {
    env : Env
}

struct EH {
    position : vec3<f32>,
    filler   : f32,
    vector   : vec3<f32>,
    radius   : f32,
    color    : vec4<f32>,
}

const grid_width  = 32;
const grid_height = 32;
const half_width    = 16;
const half_height   = 16;

const params_speed  = 0.2;

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> gridIn : array<EH>;
@group(0) @binding(2) var<storage, read_write> gridOut : array<EH>;

fn getIdx(ix : u32, iy : u32) -> u32 {
    let cx = clamp(ix, 0u, grid_width - 1u);
    let cy = clamp(iy, 0u, grid_height - 1u);

    return cy * grid_width + cx;
}

// Generates a pseudocolor (Jet colormap approximation: Blue->Green->Red) based on vector magnitude
fn get_pseudocolor(mag: f32) -> vec4<f32> {
    // Scale magnitude to a 0.0 - 1.0 range (adjust multiplier as needed for brightness)
    let v = clamp(mag * 2.0, 0.0, 1.0); 
    
    let r = clamp(1.5 - abs(4.0 * v - 3.0), 0.0, 1.0);
    let g = clamp(1.5 - abs(4.0 * v - 2.0), 0.0, 1.0);
    let b = clamp(1.5 - abs(4.0 * v - 1.0), 0.0, 1.0);
    
    return vec4<f32>(r, g, b, 1.0);
}

// ---------------------------------------------------------
// PASS 1: Update Magnetic Field (H)
// ---------------------------------------------------------
fn update_h(id : vec3<u32>){
    var idx = getIdx(id.x, id.y);

    // TMz mode: E-field oscillates in Z, H-field oscillates in X and Y
    let ez = gridIn[idx].vector.z;

    // Fetch neighboring E-fields to calculate spatial derivatives (Yee Grid)
    let ez_top   = gridIn[getIdx(id.x, id.y + 1u)].vector.z;
    let ez_right = gridIn[getIdx(id.x + 1u, id.y)].vector.z;

    var h_vec = gridOut[idx].vector;

    // Faraday's Law curl update
    h_vec.x -= params_speed * (ez_top - ez);
    h_vec.y += params_speed * (ez_right - ez);

    gridOut[idx].vector = h_vec;
    gridOut[idx].color  = vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------
// PASS 2: Update Electric Field (E) & Apply Excitation
// ---------------------------------------------------------
fn update_e(id : vec3<u32>){
    let idx = getIdx(id.x, id.y);

    let hx = gridIn[idx].vector.x;
    let hy = gridIn[idx].vector.y;

    // Fetch neighboring H-fields
    let hx_bottom = gridIn[getIdx(id.x, id.y - 1u)].vector.x;
    let hy_left   = gridIn[getIdx(id.x - 1u, id.y)].vector.y;

    var e_vec = gridOut[idx].vector;

    // Ampere's Law curl update
    e_vec.z += params_speed * ((hy - hy_left) - (hx - hx_bottom));

    if (id.x == half_width && id.y == half_height) {
        e_vec.z += 1.0 * sin((params.env.time / 1000.0) * 5.0);
    }

    gridOut[idx].vector = e_vec;
    gridOut[idx].color  = vec4<f32>(0.5, 0.5, 1.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (grid_width <= id.x || grid_height <= id.y) { 
        return; 
    }

    var tick = u32(params.env.tick);

    if(tick <= 1){
        let idx = getIdx(id.x, id.y);

        var x = mix(-5.0, 5.0, f32(id.x) / f32(grid_width));
        var y = mix(-5.0, 5.0, f32(id.y) / f32(grid_height));

        gridOut[idx].position = vec3<f32>(x, y, 0.0);
        gridOut[idx].vector   = vec3<f32>(0.0, 0.0, 0.0);
        gridOut[idx].radius   = 0.1;
        gridOut[idx].color    = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    else{
        if(tick % 2 == 0){
            update_h(id);
        }
        else{
            update_e(id);
        }
    }
}
