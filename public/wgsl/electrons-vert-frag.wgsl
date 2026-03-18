// render.wgsl

struct Camera {
    view_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var vol_tex: texture_3d<f32>;
@group(0) @binding(1) var vol_sampler: sampler;
@group(0) @binding(2) var<uniform> camera: Camera;


// 1. A Simple Full-Screen Vertex Shader
// This generates a large triangle that covers the whole screen without needing vertex buffers.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

// Helper: Intersect a ray with a 3D bounding box
fn intersectAABB(ro: vec3<f32>, rd: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
    let t0 = (boxMin - ro) / rd;
    let t1 = (boxMax - ro) / rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let near = max(max(tmin.x, tmin.y), tmin.z);
    let far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(near, far);
}

// 2. The Raymarching Fragment Shader
@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // Basic camera setup (You would normally pass these via uniforms)
    // We normalize screen coordinates to [-1, 1]
    let resolution = vec2<f32>(800.0, 800.0); // Replace with actual canvas size
    let uv = (fragCoord.xy / resolution.xy) * 2.0 - 1.0;
    
    // 3. Define the base camera setup
    let base_ro = vec3<f32>(0.0, 0.0, 3.0); 
    let base_rd = normalize(vec3<f32>(uv.x, -uv.y, -1.0));
    
    // 4. Apply the rotation matrix from TypeScript
    // Multiply by the matrix to rotate the camera's position and view direction
    let ro = (camera.view_matrix * vec4<f32>(base_ro, 1.0)).xyz;
    let rd = (camera.view_matrix * vec4<f32>(base_rd, 0.0)).xyz;

    // Define our 3D volume boundaries (-1 to +1 in all directions)
    let boxMin = vec3<f32>(-1.0, -1.0, -1.0);
    let boxMax = vec3<f32>( 1.0,  1.0,  1.0);

    // Check if the ray hits our volume bounding box
    let hit = intersectAABB(ro, rd, boxMin, boxMax);
    let t_near = hit.x;
    let t_far = hit.y;

    if (t_near > t_far || t_far < 0.0) {
        // Ray missed the volume, return black (or background color)
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); 
    }

    // --- RAYMARCHING LOOP ---
    var accumulated_color = vec4<f32>(0.0);
    let step_size = 0.02; // Smaller step = higher quality, but slower
    let max_steps = 150;
    
    // Start marching from the front of the box
    var t = max(t_near, 0.0); 

    for (var i = 0; i < max_steps; i++) {
        if (t >= t_far) { break; } // Exited the back of the box

        // Calculate 3D position of our current step
        let pos = ro + rd * t;

        // Map position from [-1, 1] physical box to [0, 1] texture coordinates
        let uvw = pos * 0.5 + 0.5;

        // Sample the probability density from our 3D texture
        // The hardware automatically interpolates this for smooth gradients!
        let density = textureSampleLevel(vol_tex, vol_sampler, uvw, 0.0).r;

        // Only process if there is actual density here
        if (density > 0.001) {
            // Transfer Function: Map density to a glowing color and opacity
            // We'll use a nice cyan/blue color for the orbital
            let base_color = vec3<f32>(0.1, 0.6, 1.0); 
            
            // Brighten the core where density is highest
            let voxel_color = base_color * density * 2.5;
            let voxel_alpha = density * 0.15;
            // let voxel_color = base_color * density * 0.8; 
            // let voxel_alpha = density * 0.02;

            // Front-to-back alpha blending
            let blended_color = vec4<f32>(voxel_color, voxel_alpha);
            accumulated_color += blended_color * (1.0 - accumulated_color.a);

            // Early exit if the cloud becomes fully opaque (saves performance)
            if (accumulated_color.a > 0.99) { break; }
        }

        t += step_size; // Move the ray forward
    }

    // Add a dark background behind the accumulated cloud
    let background = vec4<f32>(0.05, 0.05, 0.08, 1.0);
    return accumulated_color + background * (1.0 - accumulated_color.a);
}
