// Volumetric rendering of hydrogen orbital probability density.
// Ported into JSON + WGSL + DSL framework (no legacy electrons code changes).

// @shader: render_orbital

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = pos[vi] * 0.5 + 0.5;
    return out;
}

// Loose bounding sphere for ray extent only (ρ(|ψ|²) already falls off exponentially).
fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return vec2<f32>(1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

fn orbital_density(mode: i32, p: vec3<f32>) -> f32 {
    let r = max(length(p), 1e-5);
    let ct = p.z / r;
    let cos_theta = clamp(ct, -1.0, 1.0);
    let pi = 3.14159265359;
    let inv_sqrt_pi = 1.0 / sqrt(pi);
    let c_2p = 1.0 / (4.0 * sqrt(2.0 * pi));
    let c_3dz2 = 1.0 / (81.0 * sqrt(6.0 * pi));

    var psi = 0.0;
    if (mode == 0) {
        // 1s (atomic units, Z=1, a₀=1): ψ = π^{-1/2} exp(−r).
        psi = inv_sqrt_pi * exp(-r);
    } else if (mode == 1) {
        // 2p_z: ψ₂₁₀ = (4√(2π))^{-1} r exp(−r/2) cos θ.
        psi = c_2p * r * exp(-0.5 * r) * cos_theta;
    } else {
        // 3d_z²: ψ₃₂₀ = (81√(6π))^{-1} r² exp(−r/3) (3cos²θ − 1).
        psi = c_3dz2 * r * r * exp(-r / 3.0) * (3.0 * cos_theta * cos_theta - 1.0);
    }

    return psi * psi;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let bg = vec3<f32>(0.06, 0.065, 0.09);

    // Fullscreen ray in clip-like coordinates.
    let uv = in.uv * 2.0 - 1.0;
    let base_rd = normalize(vec3<f32>(uv.x, -uv.y, -1.0));

    // Framework `camera.view` is a standard view matrix (world -> camera).
    // For raymarching in world space we need camera position and camera->world rotation.
    let Rcw = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let Rwc = transpose(Rcw);
    let cam_trans = camera.view[3].xyz;
    let ro = -(Rwc * cam_trans);
    let rd = normalize(Rwc * base_rd);

    let mode = i32(round(params.orbitalMode));

    // d states extend farther in r; larger bounds avoid a hard spherical “rim”.
    let march_sphere_r = select(5.5, 6.5, mode == 2);
    let hit = intersect_sphere(ro, rd, march_sphere_r);
    let t_near = hit.x;
    let t_far = hit.y;
    if (t_near > t_far || t_far < 0.0) {
        return vec4<f32>(bg, 1.0);
    }

    let step_size = 0.016;
    let max_steps = select(720, 840, mode == 2);
    var ray_t = max(t_near, 0.0);
    var acc = vec4<f32>(0.0);

    let base_col = vec3<f32>(0.12, 0.62, 1.0);
    let warm_col = vec3<f32>(1.0, 0.62, 0.16);
    let tint = mix(base_col, warm_col, clamp(f32(mode) / 2.0, 0.0, 1.0));

    for (var i = 0; i < max_steps; i++) {
        if (ray_t >= t_far || acc.a > 0.99) { break; }
        let pos = ro + rd * ray_t;
        let p = pos * 2.5;

        let rho = orbital_density(mode, p);
        // Normalized hydrogen ρ is tiny in atomic units; photographic γ compresses dynamic range.
        if (rho > 1e-16) {
            let rho_gamma = pow(max(rho, 0.0), 0.42);
            let vis = 1.0 - exp(-rho_gamma * params.densityScale);
            let voxel_col = tint * (0.42 + 1.35 * vis);
            let voxel_alpha =
                clamp(vis * params.opacityScale * step_size * 11.0, 0.0, 0.96);
            let premul = vec4<f32>(voxel_col * voxel_alpha, voxel_alpha);
            acc += premul * (1.0 - acc.a);
        }

        ray_t += step_size;
    }

    var out_col = acc.rgb + bg * (1.0 - acc.a);
    // Mild output lift so mid-tones read on typical displays.
    out_col = pow(clamp(out_col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.82));
    return vec4<f32>(out_col, 1.0);
}
