import { fetchText } from "@i18n";
import { ComputePipeline } from "./compute.js";
import { startAnimation, makeComputeRenderPipelines } from "./instance.js";
import { ShapeInfo } from "./package.js";
import { Surface, RenderPipeline } from "./primitive.js";

export async function showVectorField() {
    // --- Configuration ---
    const grid_dim_q = 32;
    const grid_dim_p = 32;
    const hamiltonian_grid_size = 64;

    // --- Pipeline 1: Hamiltonian Surface (background) ---
    const hamiltonianShape: ShapeInfo = {
        type: "Surface",
        vertName: "hamiltonian-vert",
        fragName: "phong-frag",
        gridSize: [hamiltonian_grid_size, hamiltonian_grid_size]
    };
    const hamiltonianSurface = new Surface(hamiltonianShape);

    // --- Pipeline 2: Vector Field Arrows (compute + instanced mesh) ---
    const vectorFieldCompShaderText = await fetchText('./wgsl/vector-field-comp.wgsl');

    // We'll render each vector as an arrow.
    const arrowShapes: ShapeInfo[] = [{
        type: "arrow", // This should be a defined primitive shape
        vertName: "arrow-instance-vert",
        fragName: "phong-frag",
    }];

    const [vectorFieldCompute, arrowMeshes] = await makeComputeRenderPipelines(
        vectorFieldCompShaderText,
        [grid_dim_q, grid_dim_p], // Pass 2D grid dimensions
        arrowShapes
    );

    // --- Start Animation ---
    const comps: ComputePipeline[] = [vectorFieldCompute];
    const meshes: RenderPipeline[] = [hamiltonianSurface, ...arrowMeshes];

    await startAnimation(comps, meshes);
}