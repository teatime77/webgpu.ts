import {
    defineSimulation,
    storage, uniform,
    compute, render,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "Conway's Game of Life",
    version: "1.0",

    metadata: {
        gridWidth: 256,
        gridHeight: 256,
    },

    resources: {
        GlobalUniforms: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
        }),
        CellBuffer: storage({
            access: "read_write",
            format: "u32",
            count: "$metadata.gridWidth * $metadata.gridHeight",
            bufferCount: 2,
        }),
    },

    nodes: {
        InitCells: compute({
            shader: "init_cells",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { group: 0, binding: 0, resource: "GlobalUniforms", varName: "globalUniforms" },
                { group: 0, binding: 1, resource: "CellBuffer" },
            ],
        }),
        UpdateCells: compute({
            shader: "update_cells",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { group: 0, binding: 0, resource: "GlobalUniforms", varName: "globalUniforms" },
                { group: 0, binding: 1, resource: "CellBuffer", historyLevel: 1, varName: "cellBufferIn" },
                { group: 0, binding: 2, resource: "CellBuffer", historyLevel: 0, varName: "cellBufferOut" },
            ],
        }),
        RenderCells: render({
            shader: "render_cells",
            topology: "triangle-list",
            vertexCount: 6,
            bindings: [
                { group: 0, binding: 0, resource: "GlobalUniforms", varName: "globalUniforms" },
                { group: 0, binding: 1, resource: "CellBuffer" },
            ],
        }),
    },

    script: ({ call, loop, swap, yieldFrame, blank }) => {
        call.InitCells();
        call.RenderCells();
        blank();
        swap("CellBuffer");
        yieldFrame();
        blank();
        loop(() => {
            call.UpdateCells();
            call.RenderCells();
            blank();
            swap("CellBuffer");
            yieldFrame();
        });
    },
});
