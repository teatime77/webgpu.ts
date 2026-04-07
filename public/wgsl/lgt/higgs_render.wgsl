@group(0) @binding(0) var<storage, read> field: array<vec2<f32>>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) field_val: vec2<f32>,
};

const L: f32 = 128.0; // higgs_field.wgslのLに合わせる

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    // インスタンス描画で各格子点を四角形で描画する処理 (LGTと同じ手法)
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
    out.uv = p;
    out.field_val = field[i_idx];
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
    
    // 場の大きさ (真空期待値 VEV)
    let r = length(phi);
    // 場の方向 (位相)
    let theta = atan2(phi.y, phi.x);
    
    // 角度を色相(0.0 ~ 1.0)にマッピング
    let hue = (theta / (2.0 * 3.14159265)) + 0.5;
    let sat = 1.0;
    
    // 場の大きさが1.0(メキシカンハットの谷底)に近いほど明るくする
    let val = smoothstep(0.0, 1.2, r); 
    
    let rgb = hsv2rgb(vec3(hue, sat, val));
    return vec4(rgb, 1.0);
}