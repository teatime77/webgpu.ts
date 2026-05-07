// @shader: init_potential
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
    phiOut[idx] = 0.0;
}

// @shader: solve_potential
fn charge_density(ix: i32, iy: i32) -> f32 {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let cx = gw / 2;
    let cy = gh / 2;

    let dx = f32(ix - cx);
    let dy = f32(iy - cy);
    let r2 = dx * dx + dy * dy;
    let sigma = max(Params.sourceRadius * 0.6, 1.0);

    // Smooth source term for a stable electrostatic Poisson solve.
    let g = exp(-r2 / (2.0 * sigma * sigma));
    return Params.chargeStrength * g;
}

fn phi_at(x: i32, y: i32) -> f32 {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return phiIn[idx];
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
        // Grounded boundary (Dirichlet).
        phiOut[idx] = 0.0;
        return;
    }

    // Linear FEM on regular quads (with lumped mass) yields a 5-point stencil.
    let pL = phi_at(ix - 1, iy);
    let pR = phi_at(ix + 1, iy);
    let pD = phi_at(ix, iy - 1);
    let pU = phi_at(ix, iy + 1);
    let rhs = -charge_density(ix, iy) / max(Params.permittivity, 1e-4);
    let phiJacobi = (pL + pR + pD + pU - rhs) * 0.25;

    let old = phiIn[idx];
    let relaxed = old + Params.relaxation * (phiJacobi - old);
    phiOut[idx] = relaxed;
}

// @shader: build_surface
fn sample_phi(x: i32, y: i32) -> f32 {
    let gw = i32(Params.gridWidth);
    let gh = i32(Params.gridHeight);
    let sx = clamp(x, 0, gw - 1);
    let sy = clamp(y, 0, gh - 1);
    let idx = u32(sy * gw + sx);
    return phiIn[idx];
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

    let phi = sample_phi(ix, iy);
    let y = phi * Params.heightScale;

    let pxp = sample_phi(ix + 1, iy);
    let pxm = sample_phi(ix - 1, iy);
    let pyp = sample_phi(ix, iy + 1);
    let pym = sample_phi(ix, iy - 1);
    let dphidx = pxp - pxm;
    let dphidy = pyp - pym;
    let normal = normalize(vec3<f32>(-dphidx, 0.2, -dphidy));

    let idx = u32((iy * gw + ix) * 2);
    SurfaceData[idx] = vec4<f32>(x, y, z, phi);
    SurfaceData[idx + 1u] = vec4<f32>(normal, 0.0);
}

// @shader: render_surface
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) phi: f32,
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
    let posPhi = SurfaceData[idx];
    let n = SurfaceData[idx + 1u].xyz;

    var out: VertexOutput;
    out.position = Camera.viewProjection * vec4<f32>(posPhi.xyz, 1.0);
    out.normal = n;
    out.phi = posPhi.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cNeg = vec3<f32>(0.1, 0.35, 1.0);
    let cPos = vec3<f32>(1.0, 0.25, 0.05);
    let phiScaled = clamp(in.phi * 4.0, -1.0, 1.0);
    let t = phiScaled * 0.5 + 0.5;
    let base = cNeg * (1.0 - t) + cPos * t;

    let lightDir = normalize(vec3<f32>(0.7, 1.0, 0.4));
    let lambert = 0.2 + 0.8 * max(dot(normalize(in.normal), lightDir), 0.0);
    return vec4<f32>(base * lambert, 1.0);
}
