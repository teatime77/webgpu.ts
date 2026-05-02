struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat4x4<f32>,

    materialColor     : vec4<f32>,
    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
    lightingDirection : vec4<f32>,
    lightPosition     : vec4<f32>,
    cameraPosition    : vec4<f32>,
    env               : vec4<f32>,
    shapeInfo         : vec4<f32>,
    gridSize          : vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>, // Clip space position (internal use)
    @location(0) worldPos : vec3<f32>,       // Position in world space
    @location(1) worldNormal : vec3<f32>,    // Normal in world space
    @location(2) fragColor : vec4<f32>,
}

// --- 1. Define your Surface Function ---
fn get_z2(x: f32, y: f32) -> f32 {
    // Example: A ripple function
    let d = sqrt(x*x + y*y);
    return sin(d * 10.0) * 0.5;
}

fn get_z(x: f32, y: f32) -> f32 {
    let t = uniforms.env.x / 1000.0;
    let r = sqrt(x * x + y * y);
    return sin(r - t * 3.0) / (r + 0.1) * 2.0; 
}


// --- 2. Compute Normal Analytically ---
// Normal = normalize(cross( tangent_x, tangent_y ))
// tangent_x = (1, 0, dz/dx), tangent_y = (0, 1, dz/dy)
// This simplifies to: normalize(-dz/dx, -dz/dy, 1)
fn get_normal(x: f32, y: f32) -> vec3<f32> {
    let delta = 0.001;
    // Calculate derivatives (Central Difference)
    let dz_dx = (get_z(x + delta, y) - get_z(x - delta, y)) / (2.0 * delta);
    let dz_dy = (get_z(x, y + delta) - get_z(x, y - delta)) / (2.0 * delta);
    
    return normalize(vec3<f32>(-dz_dx, -1.0, -dz_dy)); // Note: Y is usually up in 3D, Z is depth
}

fn rgba(theta_arg : f32) -> vec4<f32>{
    const pi      = 3.14159265359;

    var theta = theta_arg;
    if(theta < 0){
        theta += 2 * pi;
    }

    var r : f32 = 0;
    var g : f32 = 0;
    var b : f32 = 0;

    var t = theta * 3.0 / (2.0 * pi);
    if(t <= 1.0){
        r = (1.0 - t);
        g = t;
    }
    else{
        t -= 1.0;
        if(t <= 1.0){
            g = (1.0 - t);
            b = t;
        }
        else{
            t -= 1.0;

            b = (1.0 - t);
            r = t;
        }
    }

    return vec4<f32>(r, g, b, 1.0);
}

@vertex
fn main(
    @builtin(vertex_index) vIndex : u32,
    @builtin(instance_index) iIndex : u32
) -> VertexOutput {
    
    // --- 3. Grid Logic (Strip Decoding) ---
    // iIndex is the Row (0 to N-1)
    let row_index = f32(iIndex);
    
    // vIndex encodes the Column. 
    // Even indices are top of strip, Odd indices are bottom.
    // vIndex: 0, 1, 2, 3 -> Column: 0, 0, 1, 1 ...
    let col_index = floor(f32(vIndex) / 2.0);
    
    // Determine if we are at the "current" row or "next" row to form the strip triangle
    let row_offset = f32(vIndex % 2u); // 0.0 or 1.0
    
    // Final Integer Grid Coordinates
    let grid_x = col_index;
    let grid_y = row_index + row_offset;

    // Normalize to UV space (0.0 to 1.0) or World Space (-1.0 to 1.0)
    let u = grid_x / uniforms.gridSize[0];       // 0 to 1
    let v = grid_y / uniforms.gridSize[1];       // 0 to 1
    
    // Map to desired X/Z world plane range (e.g., -2 to 2)
    let x_world = (u - 0.5) * 8.0;
    let z_world = (v - 0.5) * 8.0; // Using Z for "depth" in grid, Y for "height"
    
    // --- 4. Apply Surface Function ---
    // let y_height = get_z(x_world, z_world);
    let y_height = exp(x_world);
    let position = vec3<f32>(x_world, y_height, z_world);
    
    // Compute Normal
    // Note: If you use Y as up, you might swap y/z in the helper function
    let normal = get_normal(x_world, z_world); 

    // let r = (y_height * 0.5) + 0.5;
    // let b = 1.0 - (y_height * 0.5);
    // let color = vec4<f32>(r, 0.4, b, 1);
    // let color = uniforms.materialColor;
    let color = rgba(z_world);


    var output : VertexOutput;
    output.Position = uniforms.viewMatrix * vec4<f32>(position, 1.0);
    output.worldPos = position;
    output.worldNormal = normal;
    output.fragColor   = color;
   
    return output;
}