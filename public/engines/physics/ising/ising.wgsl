fn hash_u32(n: u32) -> u32 {
    var x = n;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}

// ==========================================
// @shader: init_spins
// ==========================================
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32(params.gridWidth);
    let h = u32(params.gridHeight);
    if (id.x >= w || id.y >= h) { return; }
    let idx = id.y * w + id.x;

    let r = hash_u32(idx + 13u);
    spinOut[idx] = select(0u, 1u, (r & 1u) == 1u);
}

// ==========================================
// @shader: update_spins
// ==========================================
fn spin_pm(v: u32) -> f32 {
    return select(-1.0, 1.0, v == 1u);
}

fn read_spin(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let nx = (x + w) % w;
    let ny = (y + h) % h;
    let idx = u32(ny * w + nx);
    return spin_pm(spinIn[idx]);
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = i32(params.gridWidth);
    let h = i32(params.gridHeight);
    let x = i32(id.x);
    let y = i32(id.y);
    if (x >= w || y >= h) { return; }

    let idx = u32(y * w + x);
    let s = read_spin(x, y, w, h);

    let nn = read_spin(x - 1, y, w, h)
           + read_spin(x + 1, y, w, h)
           + read_spin(x, y - 1, w, h)
           + read_spin(x, y + 1, w, h);

    let dE = 2.0 * s * (params.couplingJ * nn + params.fieldH);
    let t = max(params.temperature, 0.001);

    let frameSeed = u32(params.time * 1000.0);
    let r = rand01(idx * 1664525u + frameSeed * 1013904223u);
    let accept = (dE <= 0.0) || (r < exp(-dE / t));

    let sOut = select(s, -s, accept);
    spinOut[idx] = select(0u, 1u, sOut > 0.0);
}

// ==========================================
// @shader: render_spins
// ==========================================
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vertex_idx], 0.0, 1.0);
    out.uv = pos[vertex_idx] * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let w = u32(params.gridWidth);
    let h = u32(params.gridHeight);
    let x = min(u32(in.uv.x * f32(w)), w - 1u);
    let y = min(u32(in.uv.y * f32(h)), h - 1u);
    let idx = y * w + x;

    let s = SpinBuffer[idx];
    if (s == 1u) {
        return vec4<f32>(0.92, 0.25, 0.25, 1.0);
    }
    return vec4<f32>(0.20, 0.35, 0.95, 1.0);
}
