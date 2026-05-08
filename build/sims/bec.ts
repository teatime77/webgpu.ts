// ============================================================================
// Mean-field BEC (2D Gross–Pitaevskii equation), dimensionless harmonic units.
//
// Equation: i ∂ψ/∂t = −½∇²ψ + V ψ + g_eff |ψ|² ψ ,  V = ½ ω² r²
// Domain: x,y ∈ [−L,L] with periodic-ish clamp Laplacian at borders (Dirichlet-style clamp).
//
// Integration: RK2 (midpoint), explicit Laplacian; mass ∫|ψ|² d²x ≈ Σ|ψ|² dx²
// renormalized each frame via GPU reduction (partial sums + global sum).
//
// "temperature" UI is pedagogical: scales effective interaction g_eff (mean-field T≈0 GPE
// does not contain real T; thermal clouds need stochastic GPE / Boltzmann gas layer).
// ============================================================================

import {
    defineSimulation,
    storage, uniform,
    compute, render,
    ui,
} from "../../ts/builder/index";

const GW = 192;
export default defineSimulation({
    name: "2D Gross–Pitaevskii (mean-field BEC)",
    version: "2.0",

    uis: [
        ui.range({
            name: "temperature",
            label: "Effective temperature (↓ g_eff, more dispersive)",
            min: 0.15,
            max: 3.0,
            step: 0.01,
            live: true,
        }),
        ui.range({ name: "dt", label: "Δt", min: 0.0002, max: 0.004, step: 0.0001, live: true }),
        ui.range({ name: "g", label: "Interaction ĝ (cold scale)", min: 0.0, max: 80.0, step: 0.5, live: true }),
        ui.range({ name: "omega", label: "Trap ω", min: 0.15, max: 1.2, step: 0.01, live: true }),
        ui.range({ name: "particleNumber", label: "Norm ∫|ψ|² (dimensionless)", min: 0.5, max: 8.0, step: 0.05, live: true }),
    ],

    metadata: {
        gridWidth: GW,
        gridHeight: GW,
        temperature: 2.3,
        dt: 0.001,
        g: 14.0,
        omega: 0.48,
        particleNumber: 2.5,
        domainHalf: 7.0,
        time: 0.0,
        partialNormCount: 576,
    },

    resources: {
        BecParams: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            temperature: "f32",
            dt: "f32",
            g: "f32",
            omega: "f32",
            particleNumber: "f32",
            domainHalf: "f32",
            time: "f32",
            partialNormCount: "f32",
        }),
        Psi: storage({
            access: "read_write",
            format: "vec2<f32>",
            count: "$metadata.gridWidth * $metadata.gridHeight",
            bufferCount: 2,
        }),
        PsiRK: storage({
            access: "read_write",
            format: "vec2<f32>",
            count: "$metadata.gridWidth * $metadata.gridHeight",
        }),
        PartialNorm: storage({
            access: "read_write",
            format: "f32",
            count: "$metadata.partialNormCount",
        }),
        NormScalar: storage({
            access: "read_write",
            format: "f32",
            count: 4,
        }),
    },

    nodes: {
        InitPsi: compute({
            shader: "init_psi",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 0, varName: "psiOut", access: "read_write" },
            ],
        }),
        NormPartialA: compute({
            shader: "norm_partial",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 0, varName: "psiR", access: "read" },
                { binding: 2, resource: "PartialNorm", access: "read_write" },
            ],
        }),
        NormTotalA: compute({
            shader: "norm_total",
            workgroups: [1, 1, 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "PartialNorm", access: "read" },
                { binding: 2, resource: "NormScalar", access: "read_write" },
            ],
        }),
        NormApplyA: compute({
            shader: "norm_apply",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 0, varName: "psiN", access: "read_write" },
                { binding: 2, resource: "NormScalar", access: "read" },
            ],
        }),
        Rk2Mid: compute({
            shader: "rk2_mid",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 1, varName: "psiIn" },
                { binding: 2, resource: "PsiRK", access: "read_write" },
            ],
        }),
        Rk2Finish: compute({
            shader: "rk2_finish",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 1, varName: "psi0" },
                { binding: 2, resource: "PsiRK", access: "read" },
                { binding: 3, resource: "Psi", historyLevel: 0, varName: "psiOut", access: "read_write" },
            ],
        }),
        NormPartialB: compute({
            shader: "norm_partial",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 0, varName: "psiR", access: "read" },
                { binding: 2, resource: "PartialNorm", access: "read_write" },
            ],
        }),
        NormTotalB: compute({
            shader: "norm_total",
            workgroups: [1, 1, 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "PartialNorm", access: "read" },
                { binding: 2, resource: "NormScalar", access: "read_write" },
            ],
        }),
        NormApplyB: compute({
            shader: "norm_apply",
            workgroups: ["$metadata.gridWidth / 8", "$metadata.gridHeight / 8", 1],
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi", historyLevel: 0, varName: "psiN", access: "read_write" },
                { binding: 2, resource: "NormScalar", access: "read" },
            ],
        }),
        RenderPsi: render({
            shader: "render_psi",
            topology: "triangle-list",
            vertexCount: 6,
            bindings: [
                { binding: 0, resource: "BecParams", varName: "params" },
                { binding: 1, resource: "Psi" },
            ],
        }),
    },

    script: ({ call, loop, swap, yieldFrame, blank }) => {
        call.InitPsi();
        blank();
        call.NormPartialA();
        call.NormTotalA();
        call.NormApplyA();
        blank();
        call.RenderPsi();
        blank();
        swap("Psi");
        yieldFrame();
        blank();
        loop(() => {
            call.Rk2Mid();
            call.Rk2Finish();
            blank();
            call.NormPartialB();
            call.NormTotalB();
            call.NormApplyB();
            blank();
            call.RenderPsi();
            blank();
            swap("Psi");
            yieldFrame();
        });
    },
});
