import {
    defineSimulation,
    storage, uniform,
    compute, render,
    ui,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "fem_cg2",
    version: "1.0",

    uis: [
        ui.range({ name: "stiffness", label: "Young's modulus E", scale: "log", min: 0.01, max: 100.0, step: 0.01, format: "%.3f" }),
        ui.range({ name: "poisson", label: "Poisson's ratio", min: 0.0, max: 0.49, step: 0.01 }),
        ui.range({ name: "gravity", label: "Gravity", min: 0.0, max: 0.2, step: 0.001, restart: true }),
        ui.int({ name: "cg_iters", label: "CG iterations", min: 1, max: 500, step: 1 }),
        ui.button({ name: "_restart", label: "Restart", action: "restart" }),
        ui.button({ name: "_reset", label: "Reset defaults", action: "reset" }),
    ],

    metadata: {
        segmentsX: 10,
        segmentsY: 10,
        gridWidth: 2.0,
        gridHeight: 2.0,
        cg_iters: 100,
        dt: 0.01,
        stiffness: 1.0,
        poisson: 0.3,
        gravity: 0.05,
        damping: 0.99,
        numNodes: "(segmentsX + 1) * (segmentsY + 1)",
        numElements: "segmentsX * segmentsY * 2",
    },

    resources: {
        Params: uniform({
            dt: "f32",
            stiffness: "f32",
            poisson: "f32",
            gravity: "f32",
            damping: "f32",
            gridWidth: "f32",
            gridHeight: "f32",
            segmentsX: "f32",
            segmentsY: "f32",
            numNodes: "f32",
            numElements: "f32",
        }),
        X: storage({ access: "read_write", format: "f32", count: "$metadata.numNodes * 3" }),
        B: storage({ access: "read_write", format: "f32", count: "$metadata.numNodes * 3" }),
        P: storage({ access: "read_write", format: "f32", count: "$metadata.numNodes * 3" }),
        R: storage({ access: "read_write", format: "f32", count: "$metadata.numNodes * 3" }),
        Q: storage({ access: "read_write", format: "atomic<i32>", count: "$metadata.numNodes * 3" }),
        Scalars: storage({ access: "read_write", format: "f32", count: 8 }),
        Elements: storage({ access: "read_write", format: "u32", count: "$metadata.numElements * 3" }),
        RestPositions: storage({ access: "read_write", format: "f32", count: "$metadata.numNodes * 3" }),
    },

    nodes: {
        init_positions: compute({
            shader: "init_positions",
            workgroups: ["$metadata.numNodes / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "RestPositions" },
                { binding: 1, resource: "X" },
                { binding: 2, resource: "Params" },
            ],
        }),
        init_elements: compute({
            shader: "init_elements",
            workgroups: ["($metadata.segmentsX * $metadata.segmentsY) / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "Elements" },
                { binding: 1, resource: "Params" },
            ],
        }),
        init_b: compute({
            shader: "init_b",
            workgroups: ["$metadata.numNodes / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "B" },
                { binding: 1, resource: "Params" },
            ],
        }),
        cg_init: compute({
            shader: "cg_init",
            workgroups: ["$metadata.numNodes / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "X" },
                { binding: 1, resource: "B" },
                { binding: 2, resource: "R" },
                { binding: 3, resource: "P" },
                { binding: 4, resource: "Params" },
            ],
        }),
        calc_rho: compute({
            shader: "calc_rho",
            workgroups: [1, 1, 1],
            bindings: [
                { binding: 0, resource: "R" },
                { binding: 1, resource: "Scalars" },
            ],
        }),
        clear_q: compute({
            shader: "clear_q",
            workgroups: ["($metadata.numNodes * 3) / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "Q" },
            ],
        }),
        apply_A: compute({
            shader: "apply_A",
            workgroups: ["$metadata.numElements / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "P" },
                { binding: 1, resource: "Q" },
                { binding: 2, resource: "Elements" },
                { binding: 3, resource: "RestPositions" },
                { binding: 4, resource: "Params" },
            ],
        }),
        calc_p_q: compute({
            shader: "calc_p_q",
            workgroups: [1, 1, 1],
            bindings: [
                { binding: 0, resource: "P" },
                { binding: 1, resource: "Q" },
                { binding: 2, resource: "Scalars" },
            ],
        }),
        update_x_r: compute({
            shader: "update_x_r",
            workgroups: ["$metadata.numNodes / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "X" },
                { binding: 1, resource: "R" },
                { binding: 2, resource: "P" },
                { binding: 3, resource: "Q" },
                { binding: 4, resource: "Scalars" },
                { binding: 5, resource: "Params" },
            ],
        }),
        calc_new_rho: compute({
            shader: "calc_new_rho",
            workgroups: [1, 1, 1],
            bindings: [
                { binding: 0, resource: "R" },
                { binding: 1, resource: "Scalars" },
            ],
        }),
        update_p: compute({
            shader: "update_p",
            workgroups: ["$metadata.numNodes / 64 + 1", 1, 1],
            bindings: [
                { binding: 0, resource: "P" },
                { binding: 1, resource: "R" },
                { binding: 2, resource: "Scalars" },
                { binding: 3, resource: "Params" },
            ],
        }),
        render_mesh: render({
            shader: "render_mesh",
            topology: "triangle-list",
            vertexCount: "$metadata.numElements * 3",
            bindings: [
                { binding: 0, resource: "X", access: "read" },
                { binding: 1, resource: "Elements", access: "read" },
                { binding: 2, resource: "RestPositions", access: "read" },
            ],
        }),
    },

    script: ({ call, loop, for_, yieldFrame, blank, metadata }) => {
        call.init_positions();
        call.init_elements();
        call.init_b();
        blank();
        loop(() => {
            call.cg_init();
            call.calc_rho();
            blank();
            for_(metadata.cg_iters, () => {
                call.clear_q();
                call.apply_A();
                call.calc_p_q();
                call.update_x_r();
                call.calc_new_rho();
                call.update_p();
            });
            blank();
            call.render_mesh();
            yieldFrame();
        });
    },
});
