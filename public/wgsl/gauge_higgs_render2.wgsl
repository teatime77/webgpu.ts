@group(0) @binding(0) var<storage, read> higgs: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) field_val: vec4<f32>,
};

// typescript側の L に合わせる (128を推奨)
const L: f32 = 128.0;

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    let x = f32(i_idx % u32(L));
    let y = f32(i_idx / u32(L));
    
    var pos = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    
    let p = pos[v_idx];
    let screen_pos = vec2((x + p.x) / L * 2.0 - 1.0, (y + p.y) / L * 2.0 - 1.0);
    
    var out: VertexOutput;
    out.clip_position = vec4(screen_pos, 0.0, 1.0);
    out.field_val = higgs[i_idx]; 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // phi は SU(2)行列 (a0, a1, a2, a3)
    let phi = in.field_val; 
    
    // a1, a2, a3 は -1.0 ~ 1.0 の値をとる3次元ベクトル
    // これらを 0.0 ~ 1.0 にスケールして RGB に直接マッピングする
    let r = phi.y * 0.5 + 0.5; // a1 -> Red
    let g = phi.z * 0.5 + 0.5; // a2 -> Green
    let b = phi.w * 0.5 + 0.5; // a3 -> Blue

    // a0成分(実部)が大きいほど、色が鮮やかに出るようにコントラストをつける
    let intensity = abs(phi.x) * 1.5; 
    
    let rgb = vec3(r, g, b) * intensity;

    return vec4(rgb, 1.0);
}