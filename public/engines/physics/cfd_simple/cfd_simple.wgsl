// @shader: init_fields
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    let gw = u32(Params.gridWidth);
    let gh = u32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let idx = iy * gw + ix;
    // Channel-style base field: mean flow to the +x direction; walls at top/bottom.
    if (iy == 0u || iy == gh - 1u) {
        velOut[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if (ix == 0u) {
        let inletCenterY = Params.gridHeight * (0.5 + 0.22 * sin(Params.time * 1.4));
        let vyIn = Params.force * Params.dt * 0.8 * sin(Params.time * 2.1) *
            exp(-pow((f32(iy) - inletCenterY) / (Params.gridHeight * 0.14), 2.0));
        velOut[idx] = vec4<f32>(Params.bulkVx, vyIn, 0.0, 0.0);
    } else if (ix == gw - 1u) {
        velOut[idx] = vec4<f32>(Params.bulkVx, 0.0, 0.0, 0.0);
    } else {
        velOut[idx] = vec4<f32>(Params.bulkVx, 0.0, 0.0, 0.0);
    }
    dyeOut[idx] = 0.0;
}

// @shader: step_velocity
fn vel_at(x: i32, y: i32) -> vec2<f32> {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return velIn[idx].xy;
}

fn vel_bilerp(x: f32, y: f32) -> vec2<f32> {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = fract(x);
    let fy = fract(y);

    let v00 = vel_at(x0, y0);
    let v10 = vel_at(x0 + 1, y0);
    let v01 = vel_at(x0, y0 + 1);
    let v11 = vel_at(x0 + 1, y0 + 1);
    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;
    return v0 * (1.0 - fy) + v1 * fy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let idx = u32(iy * gw + ix);
    if (iy == 0 || iy == gh - 1) {
        velOut[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    if (ix == 0) {
        let inletCenterY = Params.gridHeight * (0.5 + 0.22 * sin(Params.time * 1.4));
        let vyIn = Params.force * Params.dt * 0.8 * sin(Params.time * 2.1) *
            exp(-pow((f32(iy) - inletCenterY) / (Params.gridHeight * 0.14), 2.0));
        velOut[idx] = vec4<f32>(Params.bulkVx, vyIn, 0.0, 0.0);
        return;
    }
    if (ix == gw - 1) {
        let vL = vel_at(ix - 1, iy);
        velOut[idx] = vec4<f32>(vL.x, vL.y, 0.0, 0.0);
        return;
    }

    let vc = vel_at(ix, iy);
    let advectScale = Params.dt * 28.0;
    let xPrev = f32(ix) - vc.x * advectScale;
    let yPrev = f32(iy) - vc.y * advectScale;
    let vAdv = vel_bilerp(xPrev, yPrev);
    let vl = vel_at(ix - 1, iy);
    let vr = vel_at(ix + 1, iy);
    let vd = vel_at(ix, iy - 1);
    let vu = vel_at(ix, iy + 1);

    let lap = vl + vr + vd + vu - 4.0 * vAdv;
    var v = vAdv + Params.viscosity * lap;

    // Optional extra inlet pulse (bulk flow already carries the field downstream).
    let inletCenterY = Params.gridHeight * (0.5 + 0.22 * sin(Params.time * 1.4));
    if (ix < 5 && abs(f32(iy) - inletCenterY) < Params.gridHeight * 0.14) {
        v = v + vec2<f32>(Params.force * Params.dt * 0.35, 0.0);
    }

    // Mild damping for stability (light so mean flow survives).
    v = v * (1.0 - 0.04 * Params.dt);
    velOut[idx] = vec4<f32>(v, 0.0, 0.0);
}

// @shader: step_dye
fn dye_at(x: i32, y: i32) -> f32 {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return dyeIn[idx];
}

fn vel_rw_at(x: i32, y: i32) -> vec2<f32> {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return velIn[idx].xy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let idx = u32(iy * gw + ix);
    if (ix == 0 || iy == 0 || ix == gw - 1 || iy == gh - 1) {
        dyeOut[idx] = 0.0;
        return;
    }

    let v = vel_rw_at(ix, iy);
    let advectScale = Params.dt * 28.0;
    let xPrev = f32(ix) - v.x * advectScale;
    let yPrev = f32(iy) - v.y * advectScale;

    let x0 = i32(floor(xPrev));
    let y0 = i32(floor(yPrev));
    let fx = fract(xPrev);
    let fy = fract(yPrev);

    let d00 = dye_at(x0, y0);
    let d10 = dye_at(x0 + 1, y0);
    let d01 = dye_at(x0, y0 + 1);
    let d11 = dye_at(x0 + 1, y0 + 1);
    let d0 = d00 * (1.0 - fx) + d10 * fx;
    let d1 = d01 * (1.0 - fx) + d11 * fx;
    var d = d0 * (1.0 - fy) + d1 * fy;

    // Add dye at inlet.
    let inletCenterY = Params.gridHeight * (0.5 + 0.22 * sin(Params.time * 1.4));
    if (ix < 4 && abs(f32(iy) - inletCenterY) < Params.gridHeight * 0.12) {
        d = 1.0;
    }
    d = max(d - Params.dyeDissipation, 0.0);
    dyeOut[idx] = d;
}

// @shader: build_surface
fn dye_sample(x: i32, y: i32) -> f32 {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return dyeIn[idx];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    if (ix >= gw || iy >= gh) {
        return;
    }

    let u = f32(ix) / (Params.gridWidth - 1.0);
    let v = f32(iy) / (Params.gridHeight - 1.0);
    let x = u * 2.0 - 1.0;
    let z = v * 2.0 - 1.0;

    let d = dye_sample(ix, iy);
    let y = d * 0.35;

    let dx = dye_sample(ix + 1, iy) - dye_sample(ix - 1, iy);
    let dy = dye_sample(ix, iy + 1) - dye_sample(ix, iy - 1);
    let n = normalize(vec3<f32>(-dx, 0.2, -dy));

    let idx = u32((iy * gw + ix) * 2);
    SurfaceData[idx] = vec4<f32>(x, y, z, d);
    SurfaceData[idx + 1u] = vec4<f32>(n, 0.0);
}

// @shader: render_surface
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) dye: f32,
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
    let posD = SurfaceData[idx];
    let n = SurfaceData[idx + 1u].xyz;

    var out: VertexOutput;
    out.position = Camera.viewProjection * vec4<f32>(posD.xyz, 1.0);
    out.normal = n;
    out.dye = posD.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let low = vec3<f32>(0.02, 0.08, 0.28);
    let high = vec3<f32>(0.05, 0.75, 1.0);
    let base = low * (1.0 - in.dye) + high * in.dye;
    let lightDir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let lambert = 0.2 + 0.8 * max(dot(normalize(in.normal), lightDir), 0.0);
    return vec4<f32>(base * lambert, 1.0);
}
