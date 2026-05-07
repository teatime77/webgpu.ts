// @shader: init_temperature
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ix = global_id.x;
    let iy = global_id.y;
    let gw = u32(Params.gridWidth);
    let gh = u32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let idx = iy * gw + ix;
    var t = Params.ambientTemperature;

    let cx = i32(gw) / 2;
    let cy = i32(gh) / 2;
    let dx = i32(ix) - cx;
    let dy = i32(iy) - cy;
    if (dx * dx + dy * dy <= 16) {
        t = Params.sourceTemperature;
    }

    tempOut[idx] = t;
}

// @shader: fem_step
fn cell(temp: ptr<storage, array<f32>, read_write>, x: u32, y: u32, gw: u32) -> f32 {
    return (*temp)[y * gw + x];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ix = global_id.x;
    let iy = global_id.y;
    let gw = u32(Params.gridWidth);
    let gh = u32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let idx = iy * gw + ix;
    let onBoundary = ix == 0u || iy == 0u || ix == gw - 1u || iy == gh - 1u;

    if (onBoundary) {
        tempOut[idx] = Params.ambientTemperature;
        return;
    }

    // Lumped-mass linear FEM on a regular grid reduces to a Laplacian stencil.
    let tc = cell(&tempIn, ix, iy, gw);
    let tl = cell(&tempIn, ix - 1u, iy, gw);
    let tr = cell(&tempIn, ix + 1u, iy, gw);
    let td = cell(&tempIn, ix, iy - 1u, gw);
    let tu = cell(&tempIn, ix, iy + 1u, gw);
    let lap = tl + tr + td + tu - 4.0 * tc;

    var tNew = tc + Params.dt * Params.thermalDiffusivity * lap;
    tNew = tNew + Params.cooling * (Params.ambientTemperature - tNew);

    let cx = i32(gw) / 2;
    let cy = i32(gh) / 2;
    let dx = i32(ix) - cx;
    let dy = i32(iy) - cy;
    if (dx * dx + dy * dy <= 9) {
        tNew = Params.sourceTemperature;
    }

    tempOut[idx] = tNew;
}

// @shader: build_surface
fn sample_temp_clamped(x: i32, y: i32) -> f32 {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return tempIn[idx];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ix = i32(global_id.x);
    let iy = i32(global_id.y);
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let u = f32(ix) / (Params.gridWidth - 1.0);
    let v = f32(iy) / (Params.gridHeight - 1.0);
    let x = u * 2.0 - 1.0;
    let z = v * 2.0 - 1.0;
    let t = sample_temp_clamped(ix, iy);
    let y = (t - Params.ambientTemperature) * Params.heightScale;

    let txp = sample_temp_clamped(ix + 1, iy);
    let txm = sample_temp_clamped(ix - 1, iy);
    let typ = sample_temp_clamped(ix, iy + 1);
    let tym = sample_temp_clamped(ix, iy - 1);
    let dtdx = txp - txm;
    let dtdy = typ - tym;
    let normal = normalize(vec3<f32>(-dtdx, 0.2, -dtdy));

    let idx = u32((iy * gw + ix) * 2);
    SurfaceData[idx] = vec4<f32>(x, y, z, t);
    SurfaceData[idx + 1u] = vec4<f32>(normal, 0.0);
}

// @shader: render_surface
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) temperature: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let gw = u32(Params.gridWidth);
    let quads_x = gw - 1u;
    let quad_id = vertex_index / 6u;
    let qx = quad_id % quads_x;
    let qy = quad_id / quads_x;

    var offsets = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u),
        vec2<u32>(0u, 1u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u)
    );
    let o = offsets[vertex_index % 6u];
    let vx = qx + o.x;
    let vy = qy + o.y;

    let idx = (vy * gw + vx) * 2u;
    let posTemp = SurfaceData[idx];
    let n = SurfaceData[idx + 1u].xyz;

    var out: VertexOutput;
    out.position = Camera.viewProjection * vec4<f32>(posTemp.xyz, 1.0);
    out.normal = n;
    out.temperature = posTemp.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = in.temperature;
    let cold = vec3<f32>(0.10, 0.25, 0.95);
    let hot = vec3<f32>(1.00, 0.25, 0.05);
    let k = clamp((t - Params.ambientTemperature) / max(Params.sourceTemperature - Params.ambientTemperature, 0.0001), 0.0, 1.0);
    let base = cold * (1.0 - k) + hot * k;
    let lightDir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let lambert = 0.2 + 0.8 * max(dot(normalize(in.normal), lightDir), 0.0);
    return vec4<f32>(base * lambert, 1.0);
}
