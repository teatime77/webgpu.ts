import {
    defineSimulation,
    storage, uniform,
    compute, render,
    ui,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "2D Ising model (statistical mechanics)",
    version: "1.0",

    metadata: {
        gridWidth: 192,
        gridHeight: 192,
        temperature: 2.2,
        couplingJ: 1.0,
        fieldH: 0.0,
        time: 0.0,
    },

    uis: [
        ui.range({ name: "temperature", label: "Temperature T", min: 0.2, max: 5.0, step: 0.01, live: true }),
        ui.range({ name: "couplingJ", label: "Coupling J", min: 0.1, max: 2.0, step: 0.01, live: true }),
        ui.range({ name: "fieldH", label: "External field h", min: -1.0, max: 1.0, step: 0.01, live: true }),
    ],

    resources: {
        IsingParams: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            temperature: "f32",
            couplingJ: "f32",
            fieldH: "f32",
            time: "f32",
        }),
        SpinBuffer: storage({
            access: "read_write",
            format: "u32",
            count: "$metadata.gridWidth * $metadata.gridHeight",
            bufferCount: 2,
        }),
    },

    nodes: {
        InitSpins: compute({
            shader: "init_spins",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { group: 0, binding: 0, resource: "IsingParams", varName: "params" },
                { group: 0, binding: 1, resource: "SpinBuffer", historyLevel: 0, varName: "spinOut", access: "read_write" },
            ],
        }),
        UpdateSpins: compute({
            shader: "update_spins",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { group: 0, binding: 0, resource: "IsingParams", varName: "params" },
                { group: 0, binding: 1, resource: "SpinBuffer", historyLevel: 1, varName: "spinIn" },
                { group: 0, binding: 2, resource: "SpinBuffer", historyLevel: 0, varName: "spinOut", access: "read_write" },
            ],
        }),
        RenderSpins: render({
            shader: "render_spins",
            topology: "triangle-list",
            vertexCount: 6,
            bindings: [
                { group: 0, binding: 0, resource: "IsingParams", varName: "params" },
                { group: 0, binding: 1, resource: "SpinBuffer" },
            ],
        }),
    },

    script: ({ call, loop, swap, yieldFrame, blank }) => {
        call.InitSpins();
        call.RenderSpins();
        blank();
        swap("SpinBuffer");
        yieldFrame();
        blank();
        loop(() => {
            call.UpdateSpins();
            call.RenderSpins();
            blank();
            swap("SpinBuffer");
            yieldFrame();
        });
    },
});
