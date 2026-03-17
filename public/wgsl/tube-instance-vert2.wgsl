struct Uniforms {
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat4x4<f32>,
    materialColor     : vec4<f32>,
    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
    lightingDirection : vec4<f32>,
    lightPosition     : vec4<f32>,
    cameraPosition    : vec4<f32>,
    env               : vec4<f32>,
    shapeInfo         : vec4<f32>
}


// 2. Instance Data (Output from your Compute Shader)
struct CylinderInstance {
    position : vec3<f32>,
    filler   : f32,
    vector   : vec3<f32>,
    radius   : f32,
    color    : vec4<f32>,
}

struct Vertex {
    vertPos : vec4<f32>,
    vertNorm: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<CylinderInstance>;
@group(0) @binding(2) var<storage, read> vertexs   : array<Vertex>;

// 4. Output to Fragment Shader
struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) fragColor: vec4<f32>,
}

@vertex
fn main(
    @builtin(instance_index) iIdx : u32,
    @builtin(vertex_index) vIdx   : u32
) -> VertexOutput {
    // Fetch the correct instance parameters calculated by the compute shader
    let instance = instances[iIdx];
    let vert     = vertexs[vIdx];
    
    let bottom = instance.position;
    let top    = bottom + instance.vector;
    let radius = instance.radius;
    
    // Calculate axis direction and total height
    let axis = top - bottom;
    let height = length(axis);
    
    // Normalize the axis direction (fallback to +Z to prevent division by zero)
    var dir = vec3<f32>(0.0, 0.0, 1.0);
    if (height > 0.0001) {
        dir = axis / height;
    }
    
    // Construct an orthonormal basis (rotation matrix) around the direction vector
    var up = vec3<f32>(0.0, 1.0, 0.0);
    // If the cylinder axis is heavily aligned with the Y-axis, use X-axis as 'up' to prevent a zero-length cross product
    if (abs(dir.y) > 0.999) { 
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    
    let right = normalize(cross(up, dir));
    let forward = cross(dir, right); // Automatically normalized since dir and right are orthogonal unit vectors
    
    // Rotation matrix from the base cylinder's local space to the instance's orientation
    let rotation = mat3x3<f32>(right, forward, dir);
    
    // --- Transformations ---
    
    // A. Scale local position
    // Since the base cylinder's Z goes from 0 to 1, scaling Z by 'height' sets the correct length.
    // X and Y are scaled by the 'radius'.
    let localPos = vec3<f32>(
        vert.vertPos.x * radius, 
        vert.vertPos.y * radius, 
        vert.vertPos.z * height
    );
    
    // B. Rotate and translate to world space
    let worldPos = rotation * localPos + bottom;
    
    // C. Transform normals
    // Because the base normals are strictly radial (XY) or axial (Z cap), 
    // non-uniform scaling doesn't skew them. We only need to apply the rotation matrix.
    let worldNorm = normalize(rotation * vert.vertNorm.xyz);
    
    // Assemble the final output
    var out: VertexOutput;
    out.Position    = uniforms.viewMatrix * vec4<f32>(worldPos, 1.0);
    out.worldPos    = worldPos;
    out.worldNormal = worldNorm;
    out.fragColor   = instance.color;
    
    return out;
}
