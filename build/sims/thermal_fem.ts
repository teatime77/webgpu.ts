import {
    defineSimulation,
    storage, uniform,
    compute, render,
    ui,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "thermal_fem",
    version: "1.0",

    uis: [
        ui.range({ name: "thermalDiffusivity", label: "Thermal Diffusivity", min: 0.01, max: 2.0, step: 0.01 }),
        ui.range({ name: "sourceTemperature", label: "Source Temperature", min: 0.1, max: 3.0, step: 0.01 }),
        ui.range({ name: "ambientTemperature", label: "Ambient Temperature", min: 0.0, max: 1.0, step: 0.01 }),
        ui.range({ name: "cooling", label: "Cooling", min: 0.0, max: 0.2, step: 0.001 }),
        ui.button({ name: "_restart", label: "Restart", action: "restart" }),
    ],

    metadata: {
        gridWidth: 96,
        gridHeight: 96,
        dt: 0.12,
        thermalDiffusivity: 0.35,
        sourceTemperature: 1.7,
        ambientTemperature: 0.12,
        cooling: 0.01,
        heightScale: 0.6,
    },

    resources: {
        Params: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            dt: "f32",
            thermalDiffusivity: "f32",
            sourceTemperature: "f32",
            ambientTemperature: "f32",
            cooling: "f32",
            heightScale: "f32",
        }),
        Camera: uniform({
            viewProjection: "mat4x4<f32>",
            view: "mat4x4<f32>",
        }),
        Temperature: storage({
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
        init_temperature: compute({
            shader: "init_temperature",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Temperature", historyLevel: 0, varName: "tempOut", access: "read_write" },
            ],
        }),
        fem_step: compute({
            shader: "fem_step",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Temperature", historyLevel: 1, varName: "tempIn" },
                { binding: 2, resource: "Temperature", historyLevel: 0, varName: "tempOut", access: "read_write" },
            ],
        }),
        build_surface: compute({
            shader: "build_surface",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Temperature", historyLevel: 0, varName: "tempIn" },
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

    script: ({ call, loop, yieldFrame, swap }) => {
        call.init_temperature();
        loop(() => {
            call.fem_step();
            call.build_surface();
            call.render_surface();
            swap("Temperature");
            yieldFrame();
        });
    },
});
