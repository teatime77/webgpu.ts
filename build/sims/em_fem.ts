import {
    defineSimulation,
    storage, uniform,
    compute, render,
    ui,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "em_fem",
    version: "1.0",

    uis: [
        ui.range({ name: "permittivity", label: "Permittivity", min: 0.1, max: 5.0, step: 0.01 }),
        ui.range({ name: "chargeStrength", label: "Charge Strength", min: -2.0, max: 2.0, step: 0.01 }),
        ui.range({ name: "relaxation", label: "Relaxation", min: 0.1, max: 1.0, step: 0.01 }),
        ui.range({ name: "heightScale", label: "Height Scale", min: 0.05, max: 1.5, step: 0.01 }),
        ui.button({ name: "_restart", label: "Restart", action: "restart" }),
    ],

    metadata: {
        gridWidth: 96,
        gridHeight: 96,
        permittivity: 1.0,
        chargeStrength: 1.0,
        relaxation: 0.92,
        heightScale: 0.45,
        sourceRadius: 7.0,
    },

    resources: {
        Params: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            permittivity: "f32",
            chargeStrength: "f32",
            relaxation: "f32",
            heightScale: "f32",
            sourceRadius: "f32",
        }),
        Camera: uniform({
            viewProjection: "mat4x4<f32>",
            view: "mat4x4<f32>",
        }),
        Potential: storage({
            access: "read_write",
            format: "f32",
            count: "$metadata.gridWidth * $metadata.gridHeight",
            bufferCount: 2,
        }),
        SurfaceData: storage({
            access: "read_write",
            format: "vec4<f32>",
            count: "$metadata.gridWidth * $metadata.gridHeight * 2",
        }),
    },

    nodes: {
        init_potential: compute({
            shader: "init_potential",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Potential", historyLevel: 0, varName: "phiOut", access: "read_write" },
            ],
        }),
        solve_potential: compute({
            shader: "solve_potential",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Potential", historyLevel: 1, varName: "phiIn" },
                { binding: 2, resource: "Potential", historyLevel: 0, varName: "phiOut", access: "read_write" },
            ],
        }),
        build_surface: compute({
            shader: "build_surface",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Potential", historyLevel: 0, varName: "phiIn" },
                { binding: 2, resource: "SurfaceData", access: "read_write" },
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

    script: ({ call, loop, swap, yieldFrame }) => {
        call.init_potential();
        loop(() => {
            call.solve_potential();
            call.build_surface();
            call.render_surface();
            swap("Potential");
            yieldFrame();
        });
    },
});
