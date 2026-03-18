// shader.wgsl

// Bind our 3D texture for writing. We use r32float to store a single density value.
@group(0) @binding(0) var volume_tex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = vec3<f32>(textureDimensions(volume_tex));
    let grid_pos = vec3<f32>(id);

    // Ensure we don't write outside the texture boundaries
    if (grid_pos.x >= dimensions.x || grid_pos.y >= dimensions.y || grid_pos.z >= dimensions.z) {
        return;
    }

    // 1. Map voxel to physical space (-15 to +15 Bohr radii)
    let normalized_pos = grid_pos / dimensions;        // Maps to [0.0, 1.0]
    let physical_pos = (normalized_pos - 0.5) * 30.0;  // Maps to [-15.0, 15.0]

    let x = physical_pos.x;
    let y = physical_pos.y;
    let z = physical_pos.z;

    // 2. Cartesian to Spherical (r, theta)
    let r = length(physical_pos);
    // Add a tiny epsilon to r to prevent division by zero at the nucleus
    let theta = acos(z / max(r, 0.00001)); 

    // 3. Evaluate the 2p_z Wave Function
    let a0 = 1.0; // Bohr radius
    
    // Psi = (r / a0) * exp(-r / 2a0) * cos(theta)
    let psi = (r / a0) * exp(-r / (2.0 * a0)) * cos(theta);

    // Probability density
    var density = psi * psi;

    // Optional: Scale the density so the brightest parts map close to 1.0 
    // This makes rendering easier later.
    density = density * 0.5; 

    // 4. Write to the 3D texture
    textureStore(volume_tex, id, vec4<f32>(density, 0.0, 0.0, 0.0));
}