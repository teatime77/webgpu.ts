import { fetchText } from "@i18n";
import { ComputePipeline } from "./compute.js";
import { startAnimation, makeComputeRenderPipelines } from "./instance.js";
import { ShapeInfo } from "./package.js";
import { Line, RenderPipeline } from "./primitive.js";

export async function showHopfFibration() {
    // --- Configuration ---
    // NOTE: If you change these, you must also change the hardcoded values in hopf-comp.wgsl
    const num_segments_per_circle = 64;
    const num_circles_theta = 30; // Corresponds to latitude
    const num_circles_phi = 40;   // Corresponds to longitude
    const total_circles = num_circles_theta * num_circles_phi;

    // --- Pipeline 1: Fibers (procedural lines) ---
    // This remains the same, drawing the animated fibers.
    const fiberShape: ShapeInfo = {
        type: "Line",
        vertName: "hopf-vert",
        fragName: "color-frag",
        // The vertex shader will use this to know how the instances are arranged
        gridSize: [num_circles_theta, num_circles_phi]
    };
    const fibers = new Line(fiberShape);
    fibers.vertexCount = num_segments_per_circle;
    fibers.renderInstanceCount = total_circles;

    // --- Pipeline 2 & 3: S2 Spheres (compute + instanced mesh) ---
    // This is the new part. We use a helper function to create a compute pipeline
    // and a render pipeline that are automatically linked together.
    const sphereCompShaderText = await fetchText('./wgsl/hopf-comp.wgsl');
    const sphereShapes: ShapeInfo[] = [{
        type: "GeodesicPolyhedron",
        vertName: "mesh-instance-vert", // Use the standard instanced mesh shader
        fragName: "phong-frag",         // Use a lit shader for a 3D look
        divideCount: 2                  // Subdivision level for the sphere primitive
    }];

    const [sphereCompute, sphereMeshes] = makeComputeRenderPipelines(
        sphereCompShaderText,
        [total_circles], // The number of spheres to generate
        sphereShapes
    );

    // --- Start Animation ---
    // We provide the compute pipeline and a list of all meshes to render.
    const comps: ComputePipeline[] = [sphereCompute];
    const meshes: RenderPipeline[] = [fibers, ...sphereMeshes];

    await startAnimation(comps, meshes);
}