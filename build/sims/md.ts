// ============================================================================
// Lennard-Jones molecular dynamics (N² pairwise, periodic minimum image).
// Open with: ?schema=md
// Compile: npx tsx build/cli.ts build/sims/md.ts --check
// ============================================================================

import {
    defineSimulation,
    storage, uniform, texture, sampler,
    compute, render,
} from "../../ts/builder/index";

const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Keep N modest: each thread sums forces over all particles (O(N²) per step). */
const N = 256;

export default defineSimulation({
    name: "Lennard-Jones molecular dynamics",
    version: "1.0",

    metadata: {
        particleCount: N,
        viewProjection: I4,
        view: I4,
        baseSphereFloatCount: 23040,
        baseSphereVertexCount: 3840,
        dt: 0.00035,
        lj_epsilon: 1.0,
        lj_sigma: 1.0,
        box_size: 12.0,
        particle_count: N,
    },

    resources: {
        CameraParams: uniform({ viewProjection: "mat4x4<f32>", view: "mat4x4<f32>" }),
        MdParams: uniform({
            dt: "f32",
            lj_epsilon: "f32",
            lj_sigma: "f32",
            box_size: "f32",
            particle_count: "f32",
        }),
        BaseSphere: storage({ format: "f32", count: "$metadata.baseSphereFloatCount" }),
        ParticlePos: storage({ format: "vec4<f32>", count: "$metadata.particleCount", bufferCount: 2 }),
        ParticleVel: storage({ format: "vec4<f32>", count: "$metadata.particleCount", bufferCount: 2 }),
        MatCapTex: texture(),
        MatCapSampler: sampler(),
    },

    nodes: {
        InitParticles: compute({
            shader: "init_particles",
            workgroups: ["$metadata.particleCount / 64", 1, 1],
            bindings: [
                { group: 0, binding: 0, resource: "MdParams" },
                { group: 0, binding: 1, resource: "ParticlePos", historyLevel: 0, varName: "posOut", access: "read_write" },
                { group: 0, binding: 2, resource: "ParticleVel", historyLevel: 0, varName: "velOut", access: "read_write" },
            ],
        }),
        MdStep: compute({
            shader: "md_step",
            workgroups: ["$metadata.particleCount / 64", 1, 1],
            bindings: [
                { group: 0, binding: 0, resource: "MdParams" },
                { group: 0, binding: 1, resource: "ParticlePos", historyLevel: 1, varName: "posIn" },
                { group: 0, binding: 2, resource: "ParticleVel", historyLevel: 1, varName: "velIn" },
                { group: 0, binding: 3, resource: "ParticlePos", historyLevel: 0, varName: "posOut", access: "read_write" },
                { group: 0, binding: 4, resource: "ParticleVel", historyLevel: 0, varName: "velOut", access: "read_write" },
            ],
        }),
        RenderParticles: render({
            shader: "matcap_spheres",
            topology: "triangle-list",
            depthTest: true,
            vertexCount: "$metadata.baseSphereVertexCount",
            instanceCount: "$metadata.particleCount",
            bindings: [
                { group: 0, binding: 0, resource: "CameraParams", varName: "camera" },
                { group: 0, binding: 1, resource: "BaseSphere" },
                { group: 0, binding: 2, resource: "ParticlePos" },
                { group: 0, binding: 3, resource: "MatCapTex" },
                { group: 0, binding: 4, resource: "MatCapSampler" },
            ],
        }),
    },

    script: ({ call, loop, swap, yieldFrame, blank }) => {
        call.InitParticles();
        call.RenderParticles();
        blank();
        swap("ParticlePos", "ParticleVel");
        yieldFrame();
        blank();
        loop(() => {
            call.MdStep();
            call.RenderParticles();
            blank();
            swap("ParticlePos", "ParticleVel");
            yieldFrame();
        });
    },
});
