// Mean-field 2D Gross–Pitaevskii (dimensionless). See build/sims/bec.ts.

fn hash_u32(n: u32) -> u32 {
    var x = n;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}

fn idx_xy(x: i32, y: i32, w: i32, h: i32) -> u32 {
    let sx = clamp(x, 0, w - 1);
    let sy = clamp(y, 0, h - 1);
    return u32(sy * w + sx);
}

fn c_mul_s(v: vec2<f32>, s: f32) -> vec2<f32> {
    return vec2<f32>(v.x * s, v.y * s);
}

fn mul_neg_i(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.y, -z.x);
}

fn trap_v(xy: vec2<f32>, omega: f32) -> f32 {
    return 0.5 * omega * omega * dot(xy, xy);
}

fn temp_factor(t: f32) -> f32 {
    return clamp((t - 0.15) / 2.85, 0.0, 1.0);
}

fn g_eff_from_ui(g: f32, temperature: f32) -> f32 {
    let u = temp_factor(temperature);
    return g * mix(1.0, 0.08, u);
}

fn phys_dx() -> f32 {
    return (2.0 * params.domainHalf) / params.gridWidth;
}

fn cell_xy_px(x: i32, y: i32) -> vec2<f32> {
    let L = params.domainHalf;
    let fx = (f32(x) + 0.5) / params.gridWidth;
    let fy = (f32(y) + 0.5) / params.gridHeight;
    return vec2<f32>(fx * 2.0 * L - L, fy * 2.0 * L - L);
}

// @shader: init_psi
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.gridWidth);
    let h = i32(params.gridHeight);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }

    let xy = cell_xy_px(x, y);
    let sigma = params.domainHalf * 0.52;
    let r2 = dot(xy, xy);
    let amp = exp(-r2 / (sigma * sigma));

    let seed = u32(y * w + x);
    let phase = 6.2831853 * (f32(hash_u32(seed ^ 0x9e3779b9u)) / 4294967295.0);
    psiOut[u32(y * w + x)] = vec2<f32>(amp * cos(phase), amp * sin(phase));
}

// @shader: rk2_mid
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.gridWidth);
    let h = i32(params.gridHeight);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }

    let dx = phys_dx();
    let inv_dx2 = 1.0 / (dx * dx);
    let geff = g_eff_from_ui(params.g, params.temperature);

    let psi_c = psiIn[u32(y * w + x)];
    let lap = psiIn[idx_xy(x + 1, y, w, h)] + psiIn[idx_xy(x - 1, y, w, h)]
            + psiIn[idx_xy(x, y + 1, w, h)] + psiIn[idx_xy(x, y - 1, w, h)]
            - 4.0 * psi_c;

    let xy = cell_xy_px(x, y);
    let rho = dot(psi_c, psi_c);
    let pot = trap_v(xy, params.omega) + geff * rho;
    let hpsi = c_mul_s(lap, -0.5 * inv_dx2) + c_mul_s(psi_c, pot);
    let rhs = mul_neg_i(hpsi);

    PsiRK[u32(y * w + x)] = psi_c + c_mul_s(rhs, 0.5 * params.dt);
}

// @shader: rk2_finish
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.gridWidth);
    let h = i32(params.gridHeight);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }

    let dx = phys_dx();
    let inv_dx2 = 1.0 / (dx * dx);
    let geff = g_eff_from_ui(params.g, params.temperature);

    let psi_mid = PsiRK[u32(y * w + x)];
    let lap = PsiRK[idx_xy(x + 1, y, w, h)] + PsiRK[idx_xy(x - 1, y, w, h)]
            + PsiRK[idx_xy(x, y + 1, w, h)] + PsiRK[idx_xy(x, y - 1, w, h)]
            - 4.0 * psi_mid;

    let xy = cell_xy_px(x, y);
    let rho = dot(psi_mid, psi_mid);
    let pot = trap_v(xy, params.omega) + geff * rho;
    let hpsi = c_mul_s(lap, -0.5 * inv_dx2) + c_mul_s(psi_mid, pot);
    let rhs = mul_neg_i(hpsi);

    let psi_n = psi0[u32(y * w + x)];
    psiOut[u32(y * w + x)] = psi_n + c_mul_s(rhs, params.dt);
}

// @shader: norm_partial
var<workgroup> norm_reduce_buf: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let w = i32(params.gridWidth);
    let h = i32(params.gridHeight);
    let gx = i32(gid.x);
    let gy = i32(gid.y);
    let dx = phys_dx();
    let cell_mass_scale = dx * dx;

    let tid = lid.y * 8u + lid.x;

    var cell = 0.0;
    if (gx < w && gy < h) {
        let p = psiR[u32(gy * w + gx)];
        cell = dot(p, p) * cell_mass_scale;
    }
    norm_reduce_buf[tid] = cell;
    workgroupBarrier();

    var s: u32 = 32u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            norm_reduce_buf[tid] += norm_reduce_buf[tid + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }

    if (tid == 0u) {
        let wx = u32(params.gridWidth / 8.0);
        PartialNorm[wid.y * wx + wid.x] = norm_reduce_buf[0];
    }
}

// @shader: norm_total
@compute @workgroup_size(1, 1, 1)
fn main() {
    let n = u32(params.partialNormCount);
    var sum = 0.0;
    for (var i = 0u; i < n; i++) {
        sum += PartialNorm[i];
    }
    NormScalar[0] = sum;
}

// @shader: norm_apply
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.gridWidth);
    let h = i32(params.gridHeight);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }

    let s = max(NormScalar[0], 1e-30);
    let scale = sqrt(params.particleNumber / s);
    let idx = u32(y * w + x);
    psiN[idx] = c_mul_s(psiN[idx], scale);
}

// @shader: render_psi
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
    let xi = i32(x);
    let yi = i32(y);
    let ww = i32(params.gridWidth);
    let hh = i32(params.gridHeight);

    let psi = Psi[y * w + x];
    let rho = dot(psi, psi);
    let u = temp_factor(params.temperature);

    // Local contrast (3×3 max |ψ|²): avoids “single bright pixel + black wings” when ρ is sharply peaked.
    var rho_mx = rho;
    let qxp = Psi[idx_xy(xi + 1, yi, ww, hh)];
    rho_mx = max(rho_mx, dot(qxp, qxp));
    let qxm = Psi[idx_xy(xi - 1, yi, ww, hh)];
    rho_mx = max(rho_mx, dot(qxm, qxm));
    let qyp = Psi[idx_xy(xi, yi + 1, ww, hh)];
    rho_mx = max(rho_mx, dot(qyp, qyp));
    let qym = Psi[idx_xy(xi, yi - 1, ww, hh)];
    rho_mx = max(rho_mx, dot(qym, qym));
    let qpp = Psi[idx_xy(xi + 1, yi + 1, ww, hh)];
    rho_mx = max(rho_mx, dot(qpp, qpp));
    let qmm = Psi[idx_xy(xi - 1, yi - 1, ww, hh)];
    rho_mx = max(rho_mx, dot(qmm, qmm));
    let qpm = Psi[idx_xy(xi + 1, yi - 1, ww, hh)];
    rho_mx = max(rho_mx, dot(qpm, qpm));
    let qmp = Psi[idx_xy(xi - 1, yi + 1, ww, hh)];
    rho_mx = max(rho_mx, dot(qmp, qmp));

    let knee = max(rho_mx * 0.55, 3e-5);
    let rho_rel = rho / (rho + knee);
    let rho_sqrt = sqrt(max(rho, 0.0));
    let rho_vis = clamp(mix(rho_rel * 1.15, rho_sqrt * 12.0, 0.35), 0.0, 1.0);

    let edge = smoothstep(0.04, 0.97, rho_vis);

    let phase = atan2(psi.y, psi.x);
    let hue_core = vec3<f32>(
        0.35 + 0.55 * cos(phase),
        0.55 + 0.35 * sin(phase + 1.3),
        0.75 + 0.25 * cos(phase - 0.7)
    );
    let hue_hot = vec3<f32>(1.0, 0.55 + 0.15 * sin(phase), 0.18);

    let tint = mix(hue_core, hue_hot, u);
    let tint_clamped = clamp(tint, vec3<f32>(0.0), vec3<f32>(1.0));

    let bg = vec3<f32>(0.02, 0.025, 0.04);
    let col = mix(bg, tint_clamped, edge * rho_vis);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
