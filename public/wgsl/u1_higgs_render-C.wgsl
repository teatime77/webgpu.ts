@group(0) @binding(0) var<storage, read> viz_results: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) q_val: f32, // トポロジカル電荷 (-1, 0, 1)
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
    out.q_val = viz_results[i_idx]; 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Q は完全に -1.0, 0.0, 1.0 のどれか
    let q = in.q_val;
    
    // 渦(+1)なら赤、反渦(-1)なら青、それ以外(0)は真っ黒
    let r = max(0.0, q);
    let b = max(0.0, -q);
    
    // ドットが目立つように少し明るくする
    return vec4(r * 2.0, 0.0, b * 2.0, 1.0);
}
