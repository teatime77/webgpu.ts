// ============================================================================
// ball simulation — authored via the typed builder API.
// Exercises: textures, samplers, ping-pong storage (bufferCount: 2 +
// historyLevel + swapPingPong), varName, depthTest, matrix metadata, and
// top-level (pre-loop) DSL statements.
//
// Compile to JSON + DSL with:
//   npx tsx webgpu/build/cli.ts webgpu/build/sims/ball.ts --check
// ============================================================================

import {
    defineSimulation,
    storage, uniform, texture, sampler,
    compute, render,
} from "../../ts/builder/index";

const I4 = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1];

export default defineSimulation({
    name: "Sphere GPU Physics",
    version: "2.0",

    metadata: {
        particleCount: 1024,
        viewProjection: I4,
        view:           I4,
        baseSphereFloatCount:  23040,
        baseSphereVertexCount: 3840,
    },

    resources: {
        CameraParams: uniform({ viewProjection: "mat4x4<f32>", view: "mat4x4<f32>" }),
        BaseSphere:   storage({ format: "f32",       count: "$metadata.baseSphereFloatCount" }),
        ParticlePos:  storage({ format: "vec4<f32>", count: "$metadata.particleCount", bufferCount: 2 }),
        ParticleVel:  storage({ format: "vec4<f32>", count: "$metadata.particleCount", bufferCount: 2 }),
        MatCapTex:     texture(),
        MatCapSampler: sampler(),
    },

    nodes: {
        InitParticles: compute({
            shader: "init_particles",
            workgroups: ["$metadata.particleCount / 64", 1, 1],
            bindings: [
                { group: 0, binding: 0, resource: "ParticlePos", historyLevel: 0, varName: "posOut", access: "read_write" },
                { group: 0, binding: 1, resource: "ParticleVel", historyLevel: 0, varName: "velOut", access: "read_write" },
            ],
        }),
        UpdateParticles: compute({
            shader: "update_particles",
            workgroups: ["$metadata.particleCount / 64", 1, 1],
            bindings: [
                { group: 0, binding: 0, resource: "ParticlePos", historyLevel: 1, varName: "posIn" },
                { group: 0, binding: 1, resource: "ParticlePos", historyLevel: 0, varName: "posOut", access: "read_write" },
                { group: 0, binding: 2, resource: "ParticleVel", historyLevel: 1, varName: "velIn" },
                { group: 0, binding: 3, resource: "ParticleVel", historyLevel: 0, varName: "velOut", access: "read_write" },
            ],
        }),
        RenderParticles: render({
            shader: "matcap_spheres",
            topology: "triangle-list",
            depthTest: true,
            vertexCount:   "$metadata.baseSphereVertexCount",
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
            call.UpdateParticles();
            call.RenderParticles();
            blank();
            swap("ParticlePos", "ParticleVel");
            yieldFrame();
        });
    },
});
