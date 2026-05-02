// ==========================================
// @shader: init_cells
// ==========================================
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32(globalUniforms.gridWidth);
    let h = u32(globalUniforms.gridHeight);
    if (id.x >= w || id.y >= h) { return; }

    let idx = id.y * w + id.x;
    
    let hash = (id.x * 374761393u + id.y * 668265263u) ^ 1013904223u;
    if (hash % 100u < 20u) {
        cellBuffer[idx] = 1u;
    } else {
        cellBuffer[idx] = 0u;
    }
}

// ==========================================
// @shader: update_cells
// ==========================================
fn get_cell(x: i32, y: i32, w: i32, h: i32) -> u32 {
    let nx = (x + w) % w;
    let ny = (y + h) % h;
    return cellBufferIn[u32(ny * w + nx)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = i32(globalUniforms.gridWidth);
    let h = i32(globalUniforms.gridHeight);
    let x = i32(id.x);
    let y = i32(id.y);
    
    if (x >= w || y >= h) { return; }

    var alive_neighbors = 0u;
    alive_neighbors += get_cell(x - 1, y - 1, w, h);
    alive_neighbors += get_cell(x,     y - 1, w, h);
    alive_neighbors += get_cell(x + 1, y - 1, w, h);
    alive_neighbors += get_cell(x - 1, y,     w, h);
    alive_neighbors += get_cell(x + 1, y,     w, h);
    alive_neighbors += get_cell(x - 1, y + 1, w, h);
    alive_neighbors += get_cell(x,     y + 1, w, h);
    alive_neighbors += get_cell(x + 1, y + 1, w, h);

    let idx = u32(y * w + x);
    let current_state = cellBufferIn[idx];

    if (current_state == 1u) {
        if (alive_neighbors == 2u || alive_neighbors == 3u) {
            cellBufferOut[idx] = 1u;
        } else {
            cellBufferOut[idx] = 0u;
        }
    } else {
        if (alive_neighbors == 3u) {
            cellBufferOut[idx] = 1u;
        } else {
            cellBufferOut[idx] = 0u;
        }
    }
}

// ==========================================
// @shader: render_cells
// ==========================================
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
    let w = globalUniforms.gridWidth;
    let h = globalUniforms.gridHeight;
    
    let max_x = u32(w) - 1u;
    let max_y = u32(h) - 1u;
    let x = min(u32(in.uv.x * w), max_x);
    let y = min(u32(in.uv.y * h), max_y);
    
    let idx = y * u32(w) + x;
    let cell_state = cellBuffer[idx];

    if (cell_state == 1u) {
        return vec4<f32>(0.0, 0.8, 0.2, 1.0); 
    }
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
}