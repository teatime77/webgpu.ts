@group(0) @binding(0) var<storage, read> higgs: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) field_val: vec4<f32>,
};

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
    // 画面いっぱいに描画
    let screen_pos = vec2((x + p.x) / L * 2.0 - 1.0, (y + p.y) / L * 2.0 - 1.0);
    
    var out: VertexOutput;
    out.clip_position = vec4(screen_pos, 0.0, 1.0);
    out.field_val = higgs[i_idx]; // SU(2)行列 (a0, a1, a2, a3) をフラグメントへ渡す
    return out;
}

// HSVからRGBへの変換
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3(0.0), vec3(1.0)), c.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let phi = in.field_val; 
    
    // SU(2)のベクトル成分(a1, a2)を使って角度(位相)を計算
    let theta = atan2(phi.z, phi.y); 
    
    // 角度を色相(0.0 ~ 1.0)にマッピング
    let hue = (theta / (2.0 * 3.14159265)) + 0.5;
    
    let sat = 1.0;
    
    // a0成分(実部)の大きさを明度に反映させると、SU(2)特有の立体感が出ます
    let val = 0.5 + 0.5 * abs(phi.x); 
    
    let rgb = hsv2rgb(vec3(hue, sat, val));
    return vec4(rgb, 1.0);
}