// ==========================================
// COMPUTE SHADER (Calculates Vertices)
// ==========================================
struct ComputeParams {
    gridSize : f32,
    extent : f32,
    time : f32,
    padding : f32,
}

@group(0) @binding(0) var<storage, read_write> vertices : array<f32>;
@group(0) @binding(1) var<uniform> params : ComputeParams;

fn f(x: f32, y: f32, t: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    return sin(r - t * 3.0) / (r + 0.1) * 2.0; 
}

@compute @workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let gridSizeU = u32(params.gridSize);
    
    if (id.x > gridSizeU || id.y > gridSizeU) {
        return;
    }

    let step = (params.extent * 2.0) / params.gridSize;
    let x = -params.extent + f32(id.x) * step;
    let y = -params.extent + f32(id.y) * step;
    let z = f(x, y, params.time);

    let eps = 0.001;
    let dzdx = (f(x + eps, y, params.time) - f(x - eps, y, params.time)) / (2.0 * eps);
    let dzdy = (f(x, y + eps, params.time) - f(x, y - eps, params.time)) / (2.0 * eps);
    let normal = normalize(vec3<f32>(-dzdx, -dzdy, 1.0));

    // Calculate flat array index (6 floats per vertex)
    let idx = (id.y * (gridSizeU + 1u) + id.x) * 6u;
    
    vertices[idx + 0u] = x;
    vertices[idx + 1u] = y;
    vertices[idx + 2u] = z;
    vertices[idx + 3u] = normal.x;
    vertices[idx + 4u] = normal.y;
    vertices[idx + 5u] = normal.z;
}

// ==========================================
// RENDER SHADER (Draws the Mesh)
// ==========================================
struct RenderUniforms {
    mvpMatrix : mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> renderUniforms : RenderUniforms;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
}

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) height : f32,
    @location(1) normal : vec3<f32>,
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.Position = renderUniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.height = input.position.z;
    output.normal = input.normal; 
    return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0)); 
    let N = normalize(input.normal);
    let diffuse = max(dot(N, lightDir), 0.0);
    let ambient = 0.2;
    let lighting = ambient + diffuse;

    // Color gradient based on height
    let r = (input.height * 0.5) + 0.5;
    let b = 1.0 - (input.height * 0.5);
    let baseColor = vec3<f32>(r, 0.4, b);

    return vec4<f32>(baseColor * lighting, 1.0);
}