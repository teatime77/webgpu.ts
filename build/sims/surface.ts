import {
    defineSimulation,
    storage, uniform,
    compute, render,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "surface",
    version: "1.0",

    metadata: {
        gridWidth: 100,
        gridHeight: 100,
        time: 0.0,
    },

    resources: {
        Params: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            time: "f32",
        }),
        Camera: uniform({
            viewProjection: "mat4x4<f32>",
            view: "mat4x4<f32>",
        }),
        SurfaceData: storage({
            access: "read_write",
            format: "vec4<f32>",
            count: "$metadata.gridWidth * $metadata.gridHeight * 2",
        }),
    },

    nodes: {
        compute_surface: compute({
            shader: "compute_surface",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "SurfaceData" },
            ],
        }),
        render_surface: render({
            shader: "render_surface",
            topology: "triangle-list",
            depthTest: true,
            vertexCount: "($metadata.gridWidth - 1) * ($metadata.gridHeight - 1) * 6",
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Camera" },
                { binding: 2, resource: "SurfaceData", access: "read" },
            ],
        }),
    },

    script: ({ call, loop, yieldFrame }) => {
        loop(() => {
            call.compute_surface();
            call.render_surface();
            yieldFrame();
        });
    },
});
