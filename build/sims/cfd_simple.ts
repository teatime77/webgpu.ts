import {
    defineSimulation,
    storage, uniform,
    compute, render,
    ui,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "cfd_simple",
    version: "1.0",

    uis: [
        ui.range({ name: "dt", label: "Time Step", min: 0.005, max: 0.08, step: 0.001 }),
        ui.range({ name: "viscosity", label: "Viscosity", min: 0.0001, max: 0.02, step: 0.0001 }),
        ui.range({ name: "bulkVx", label: "Mean flow (→)", min: 0.0, max: 4.0, step: 0.05 }),
        ui.range({ name: "force", label: "Inlet Force", min: 0.0, max: 4.0, step: 0.01 }),
        ui.range({ name: "dyeDissipation", label: "Dye Dissipation", min: 0.0, max: 0.02, step: 0.0001 }),
        ui.button({ name: "_restart", label: "Restart", action: "restart" }),
    ],

    metadata: {
        gridWidth: 128,
        gridHeight: 96,
        time: 0.0,
        dt: 0.02,
        viscosity: 0.0015,
        bulkVx: 1.5,
        force: 1.8,
        dyeDissipation: 0.002,
    },

    resources: {
        Params: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            time: "f32",
            dt: "f32",
            viscosity: "f32",
            bulkVx: "f32",
            force: "f32",
            dyeDissipation: "f32",
        }),
        Camera: uniform({
            viewProjection: "mat4x4<f32>",
            view: "mat4x4<f32>",
        }),
        Velocity: storage({
            access: "read_write",
            format: "vec4<f32>",
            count: "$metadata.gridWidth * $metadata.gridHeight",
            bufferCount: 2,
        }),
        Dye: storage({
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
        init_fields: compute({
            shader: "init_fields",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Velocity", historyLevel: 0, varName: "velOut", access: "read_write" },
                { binding: 2, resource: "Dye", historyLevel: 0, varName: "dyeOut", access: "read_write" },
            ],
        }),
        step_velocity: compute({
            shader: "step_velocity",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Velocity", historyLevel: 1, varName: "velIn" },
                { binding: 2, resource: "Velocity", historyLevel: 0, varName: "velOut", access: "read_write" },
            ],
        }),
        step_dye: compute({
            shader: "step_dye",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Velocity", historyLevel: 0, varName: "velIn" },
                { binding: 2, resource: "Dye", historyLevel: 1, varName: "dyeIn" },
                { binding: 3, resource: "Dye", historyLevel: 0, varName: "dyeOut", access: "read_write" },
            ],
        }),
        build_surface: compute({
            shader: "build_surface",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Dye", historyLevel: 0, varName: "dyeIn" },
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

    script: ({ call, loop, swap, yieldFrame, blank }) => {
        call.init_fields();
        // Align ping–pong with UpdateParticles-style sims (e.g. ball): first solver
        // step must read the buffer slot that init just wrote (historyLevel 1).
        blank();
        swap("Velocity", "Dye");
        // Required: GraphManager.step() asserts YieldStatement immediately after swapPingPong.
        blank();
        yieldFrame();
        blank();
        loop(() => {
            call.step_velocity();
            call.step_dye();
            call.build_surface();
            call.render_surface();
            swap("Velocity", "Dye");
            yieldFrame();
        });
    },
});
