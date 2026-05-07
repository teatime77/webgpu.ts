// @shader: init_positions
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let numNodes = u32(Params.numNodes);
    if (idx >= numNodes) { return; }

    let segX = u32(Params.segmentsX);
    let segY = u32(Params.segmentsY);
    let w = Params.gridWidth;
    let h = Params.gridHeight;

    let x = idx % (segX + 1u);
    let y = idx / (segX + 1u);

    let px = (f32(x) / f32(segX) - 0.5) * w;
    let py = (f32(y) / f32(segY) - 0.5) * h;
    let pz = 0.0;

    RestPositions[idx * 3u + 0u] = px;
    RestPositions[idx * 3u + 1u] = py;
    RestPositions[idx * 3u + 2u] = pz;

    X[idx * 3u + 0u] = 0.0;
    X[idx * 3u + 1u] = 0.0;
    X[idx * 3u + 2u] = 0.0;
}

// @shader: init_elements
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    let segX = u32(Params.segmentsX);
    let segY = u32(Params.segmentsY);
    if (cell_idx >= segX * segY) { return; }

    let cx = cell_idx % segX;
    let cy = cell_idx / segX;

    let i = cx + cy * (segX + 1u);

    let eIdx = cell_idx * 6u; // 2 triangles * 3 indices

    // Triangle 1
    Elements[eIdx + 0u] = i;
    Elements[eIdx + 1u] = i + 1u;
    Elements[eIdx + 2u] = i + (segX + 1u);

    // Triangle 2
    Elements[eIdx + 3u] = i + 1u;
    Elements[eIdx + 4u] = i + (segX + 1u) + 1u;
    Elements[eIdx + 5u] = i + (segX + 1u);
}

// @shader: init_b
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let node_idx = id.x;
    let numNodes = u32(Params.numNodes);
    if (node_idx >= numNodes) { return; }

    // Gravity: apply a uniform downward load to every node.
    // The top row is clamped in cg_init/update_x_r/update_p, so the load
    // there is harmlessly zeroed out by the solver.
    let force = -Params.gravity;

    B[node_idx * 3u + 0u] = 0.0;
    B[node_idx * 3u + 1u] = force;
    B[node_idx * 3u + 2u] = 0.0;
}

// @shader: cg_init
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let node_idx = id.x;
    let numNodes = u32(Params.numNodes);
    if (node_idx >= numNodes) { return; }

    let segX = u32(Params.segmentsX);
    let segY = u32(Params.segmentsY);
    let y = node_idx / (segX + 1u);
    let is_fixed = (y == segY);

    for (var c = 0u; c < 3u; c++) {
        let idx = node_idx * 3u + c;
        X[idx] = 0.0;
        
        var b_val = B[idx];
        if (is_fixed) {
            b_val = 0.0;
        }
        R[idx] = b_val;
        P[idx] = b_val;
    }
}

// @shader: calc_rho
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    if (local_id.x == 0u) {
        var sum = 0.0;
        let len = arrayLength(&R);
        for (var i = 0u; i < len; i++) {
            let r_val = R[i];
            sum += r_val * r_val;
        }
        // Scalars: [0: rho, 1: p_q, 2: alpha, 3: beta, 4: new_rho]
        Scalars[0] = sum;
    }
}

// @shader: clear_q
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let len = arrayLength(&Q);
    if (idx >= len) { return; }
    atomicStore(&Q[idx], 0);
}

// @shader: apply_A
const SCALE: f32 = 100000.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let elem_idx = id.x;
    let numElements = arrayLength(&Elements) / 3u;
    if (elem_idx >= numElements) { return; }

    let i0 = Elements[elem_idx * 3u + 0u];
    let i1 = Elements[elem_idx * 3u + 1u];
    let i2 = Elements[elem_idx * 3u + 2u];

    let p0 = vec3<f32>(P[i0*3u], P[i0*3u+1u], P[i0*3u+2u]);
    let p1 = vec3<f32>(P[i1*3u], P[i1*3u+1u], P[i1*3u+2u]);
    let p2 = vec3<f32>(P[i2*3u], P[i2*3u+1u], P[i2*3u+2u]);

    // Simple Laplacian-like stiffness for demonstration
    // In a real FEM, you'd compute B^T * D * B * p
    let k = Params.stiffness;
    
    let diff01 = p0 - p1;
    let diff12 = p1 - p2;
    let diff20 = p2 - p0;

    let q0 = k * (diff01 - diff20);
    let q1 = k * (diff12 - diff01);
    let q2 = k * (diff20 - diff12);

    atomicAdd(&Q[i0*3u + 0u], i32(q0.x * SCALE));
    atomicAdd(&Q[i0*3u + 1u], i32(q0.y * SCALE));
    atomicAdd(&Q[i0*3u + 2u], i32(q0.z * SCALE));

    atomicAdd(&Q[i1*3u + 0u], i32(q1.x * SCALE));
    atomicAdd(&Q[i1*3u + 1u], i32(q1.y * SCALE));
    atomicAdd(&Q[i1*3u + 2u], i32(q1.z * SCALE));

    atomicAdd(&Q[i2*3u + 0u], i32(q2.x * SCALE));
    atomicAdd(&Q[i2*3u + 1u], i32(q2.y * SCALE));
    atomicAdd(&Q[i2*3u + 2u], i32(q2.z * SCALE));
}

// @shader: calc_p_q
const SCALE: f32 = 100000.0;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    if (local_id.x == 0u) {
        var sum = 0.0;
        let len = arrayLength(&P);
        for (var i = 0u; i < len; i++) {
            let q_val = f32(atomicLoad(&Q[i])) / SCALE;
            sum += P[i] * q_val;
        }
        Scalars[1] = sum; // p_q
        let rho = Scalars[0];
        Scalars[2] = rho / sum; // alpha
    }
}

// @shader: update_x_r
const SCALE: f32 = 100000.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let node_idx = id.x;
    let numNodes = u32(Params.numNodes);
    if (node_idx >= numNodes) { return; }

    let segX = u32(Params.segmentsX);
    let segY = u32(Params.segmentsY);
    let y = node_idx / (segX + 1u);
    let is_fixed = (y == segY);

    let alpha = Scalars[2];

    for (var c = 0u; c < 3u; c++) {
        let idx = node_idx * 3u + c;
        if (is_fixed) {
            X[idx] = 0.0;
            R[idx] = 0.0;
        } else {
            X[idx] = X[idx] + alpha * P[idx];
            let q_val = f32(atomicLoad(&Q[idx])) / SCALE;
            R[idx] = R[idx] - alpha * q_val;
        }
    }
}

// @shader: calc_new_rho
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    if (local_id.x == 0u) {
        var sum = 0.0;
        let len = arrayLength(&R);
        for (var i = 0u; i < len; i++) {
            let r_val = R[i];
            sum += r_val * r_val;
        }
        Scalars[4] = sum; // new_rho
        let rho = Scalars[0];
        Scalars[3] = sum / rho; // beta
        Scalars[0] = sum; // rho = new_rho
    }
}

// @shader: update_p
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let node_idx = id.x;
    let numNodes = u32(Params.numNodes);
    if (node_idx >= numNodes) { return; }

    let segX = u32(Params.segmentsX);
    let segY = u32(Params.segmentsY);
    let y = node_idx / (segX + 1u);
    let is_fixed = (y == segY);

    let beta = Scalars[3];

    for (var c = 0u; c < 3u; c++) {
        let idx = node_idx * 3u + c;
        if (is_fixed) {
            P[idx] = 0.0;
        } else {
            P[idx] = R[idx] + beta * P[idx];
        }
    }
}

// @shader: render_mesh
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOutput {
    let elem_idx = v_idx / 3u;
    let local_idx = v_idx % 3u;
    let node_idx = Elements[elem_idx * 3u + local_idx];
    
    let rest = vec3<f32>(RestPositions[node_idx*3u], RestPositions[node_idx*3u+1u], RestPositions[node_idx*3u+2u]);
    let disp = vec3<f32>(X[node_idx*3u], X[node_idx*3u+1u], X[node_idx*3u+2u]);
    let pos = rest + disp;
    
    var out: VertexOutput;
    // Simple scaling for visibility
    out.position = vec4<f32>(pos.x * 0.5, pos.y * 0.5, 0.0, 1.0);
    
    // UV for checkerboard pattern
    out.uv = vec2<f32>(rest.x, rest.y);
    
    // Color based on displacement magnitude (blue -> red)
    let stretch = clamp(length(disp) * 0.5, 0.0, 1.0);
    out.color = mix(vec3<f32>(0.2, 0.6, 1.0), vec3<f32>(1.0, 0.2, 0.2), stretch);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Checkerboard pattern to make deformation visible
    let grid = step(0.0, sin(in.uv.x * 40.0) * sin(in.uv.y * 40.0));
    let c = mix(in.color, in.color * 0.5, grid);
    return vec4<f32>(c, 1.0);
}
