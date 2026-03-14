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

struct Particle { position: vec4<f32>, scale: vec4<f32>, color: vec4<f32> }

struct Vertex {
    vertPos : vec4<f32>,
    vertNorm: vec4<f32>,
}


@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> particles : array<Particle>;
@group(0) @binding(2) var<storage, read> vertexs   : array<Vertex>;

struct VertexOutput {
    @builtin(position) Position    : vec4<f32>, // Clip space position (internal use)
    @location(0)       worldPos    : vec3<f32>, // Position in world space
    @location(1)       worldNormal : vec3<f32>, // Normal in world space
    @location(2)       fragColor   : vec4<f32>,
}

@vertex
fn main(
    @builtin(instance_index) iIdx : u32,
    @builtin(vertex_index) vIdx   : u32
) -> VertexOutput {
    var particle = particles[iIdx];
    var vertex   = vertexs[vIdx];

    var output : VertexOutput;

    // 1. Calculate XYZ independently to prevent 'w' component corruption
    let scaledPos = vertex.vertPos.xyz * particle.scale.xyz;
    let worldXYZ = particle.position.xyz + scaledPos;
    
    // 2. Safely construct the vec4 with a hardcoded w = 1.0
    let worldPos = vec4<f32>(worldXYZ, 1.0);

    output.Position = uniforms.viewMatrix * worldPos;
    output.worldPos = worldPos.xyz;
    output.fragColor = particle.color;
    
    // 3. Keep normals in World Space. 
    // Since particles have no rotation, the world normal is just the local normal.
    // (Note: If particle.scale is non-uniform, e.g. x=2, y=1, you should divide 
    // the normal by the scale instead to keep it perpendicular to the surface).
    output.worldNormal = vertex.vertNorm.xyz / particle.scale.xyz; 
    
    return output;
}
