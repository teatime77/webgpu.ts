import { fetchText } from "@i18n";
import { ComputePipeline } from "./compute.js";
import { startAnimation, makeComputeRenderPipelines } from "./instance.js";
import { ShapeInfo } from "./package.js";
import { Surface, RenderPipeline } from "./primitive.js";

export async function showLiouville() {
    // --- Configuration ---
    const points_side = 23;
    const num_points = points_side * points_side; // Total number of points for the phase space plot
    const hamiltonian_grid_size = 64; // Grid resolution for the Hamiltonian surface

    // --- Pipeline 1: Hamiltonian Surface (procedural mesh) ---
    const hamiltonianShape: ShapeInfo = {
        type: "Surface",
        vertName: "hamiltonian-vert", // Custom vertex shader
        fragName: "phong-frag",       // Standard lit fragment shader
        gridSize: [hamiltonian_grid_size, hamiltonian_grid_size]
    };
    const hamiltonianSurface = new Surface(hamiltonianShape);

    // --- Pipeline 2: Phase Space Points (compute + instanced mesh) ---
    // The initial positions of the points are now calculated in the compute shader
    // on the first frame (when tick/frame == 0).

    const liouvilleCompShaderText = await fetchText('./wgsl/liouville-comp.wgsl');

    // We'll render each point as a small sphere.
    const pointShapes: ShapeInfo[] = [{
        type: "GeodesicPolyhedron",
        vertName: "point-instance-vert",
        fragName: "color-frag", // Simple color, no lighting for points
        divideCount: 0, // Icosahedron is enough for a "point"
    }];

    const [pointsCompute, pointsMeshes] = await makeComputeRenderPipelines(
        liouvilleCompShaderText,
        [num_points],
        pointShapes
    );
    
    // Make the points red for visibility
    pointsMeshes.forEach(m => m.red());

    // --- Start Animation ---
    const comps: ComputePipeline[] = [pointsCompute];
    const meshes: RenderPipeline[] = [hamiltonianSurface, ...pointsMeshes];

    await startAnimation(comps, meshes);
}