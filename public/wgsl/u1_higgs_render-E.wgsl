@group(0) @binding(0) var<storage, read> higgs: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) phi: f32,
};

const L: f32 = 64.0;

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    let x = f32(i_idx % u32(L)); let y = f32(i_idx / u32(L));
    var pos = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    let p = pos[v_idx];
    let screen_pos = vec2((x + p.x) / L * 2.0 - 1.0, (y + p.y) / L * 2.0 - 1.0);
    
    var out: VertexOutput;
    out.clip_position = vec4(screen_pos, 0.0, 1.0);
    out.phi = higgs[i_idx]; 
    return out;
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3(0.0), vec3(1.0)), c.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // phi: -PI to PI
    let hue = (in.phi / (2.0 * 3.14159265)) + 0.5;
    return vec4(hsv2rgb(vec3(hue, 1.0, 1.0)), 1.0);
}