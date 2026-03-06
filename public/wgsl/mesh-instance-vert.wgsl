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
    shapeInfo         : vec4<f32>
}

struct Particle {
    position : vec4<f32>,
    scale    : vec4<f32>,
    color    : vec4<f32>,
}

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
    @builtin(instance_index) iIdx : u32,   // Counts 0..ParticleCount
    @builtin(vertex_index) vIdx   : u32    // Counts 0..SphereVertexCount
) -> VertexOutput {
    var particle = particles[iIdx];
    var vertex   = vertexs[vIdx];

    var output : VertexOutput;

    let worldPos = particle.position + vec4<f32>(vertex.vertPos.xyz, 1) * particle.scale;

    output.Position    = uniforms.viewMatrix * worldPos;
    output.worldPos    = worldPos.xyz;
    output.worldNormal = (uniforms.normMatrix * vec4<f32>(vertex.vertNorm.xyz, 0.0)).xyz;
    output.fragColor   = particle.color;
    
    return output;
}
