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

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>, // Clip space position (internal use)
    @location(0) worldPos : vec3<f32>,       // Position in world space
    @location(1) worldNormal : vec3<f32>,    // Normal in world space
    @location(2) fragColor : vec4<f32>,
};

@fragment
fn main(input : VertexOutput) -> @location(0) vec4<f32> {
    
    // Constants for material properties (could also be uniforms)
    let ambientStrength = 0.1;
    let specularStrength = 0.5;
    let shininess = 32.0;
    let lightColor = vec3<f32>(1.0, 1.0, 1.0);
    let objectColor = vec3<f32>(0.0, 0.5, 1.0); // Nice blue

    // -----------------------------------------------------
    // A. Renormalize!
    // The rasterizer linearly interpolates the normal, which changes its 
    // length. We must normalize it back to length 1.0.
    let N = normalize(input.worldNormal);
    
    // -----------------------------------------------------
    // B. Calculate Vectors
    let L = normalize(uniforms.lightPosition.xyz - input.worldPos); // Light Dir
    let V = normalize(uniforms.cameraPosition.xyz - input.worldPos); // View Dir
    
    // -----------------------------------------------------
    // C. Ambient Component
    // Base light so shadows aren't pitch black.
    let ambient = ambientStrength * lightColor;

    // -----------------------------------------------------
    // D. Diffuse Component
    // How much does the surface face the light? (Dot Product)
    let diff = max(dot(N, L), 0.0); 
    let diffuse = diff * lightColor;

    // -----------------------------------------------------
    // E. Specular Component (Phong)
    // Calculate reflection direction. 
    // reflect(incident, normal) expects incident vector pointing TO surface.
    // Our L points to light, so we use -L.
    let R = reflect(-L, N); 
    
    // Calculate alignment between Reflection and View
    let spec = pow(max(dot(V, R), 0.0), shininess);
    let specular = specularStrength * spec * lightColor;

    // -----------------------------------------------------
    // F. Combine
    let finalColor = (ambient + diffuse + specular) * objectColor;

    return vec4<f32>(finalColor, 1.0);
}
