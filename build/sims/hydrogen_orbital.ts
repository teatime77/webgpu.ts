import {
    defineSimulation,
    uniform,
    render,
    ui,
} from "../../ts/builder/index";

const I4 = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1];

export default defineSimulation({
    name: "Hydrogen Orbital Volume Rendering",
    version: "1.0",

    uis: [
        ui.select({
            name: "orbitalMode",
            label: "Orbital",
            options: [
                { value: 0, label: "1s (spherical)" },
                { value: 1, label: "2p_z (dumbbell)" },
                { value: 2, label: "3d_z2 (donut + lobes)" },
            ],
            live: true,
        }),
        ui.range({
            name: "densityScale",
            label: "Density scale",
            min: 4.0,
            max: 56.0,
            step: 0.5,
            live: true,
        }),
        ui.range({
            name: "opacityScale",
            label: "Opacity scale",
            min: 0.5,
            max: 8.0,
            step: 0.05,
            live: true,
        }),
    ],

    metadata: {
        orbitalMode: 1,
        densityScale: 24.0,
        opacityScale: 4.0,
        time: 0.0,
        viewProjection: I4,
        view: I4,
    },

    resources: {
        OrbitalParams: uniform({
            orbitalMode: "f32",
            densityScale: "f32",
            opacityScale: "f32",
            time: "f32",
        }),
        Camera: uniform({
            viewProjection: "mat4x4<f32>",
            view: "mat4x4<f32>",
        }),
    },

    nodes: {
        RenderOrbital: render({
            shader: "render_orbital",
            topology: "triangle-list",
            vertexCount: 3,
            bindings: [
                { binding: 0, resource: "OrbitalParams", varName: "params" },
                { binding: 1, resource: "Camera", varName: "camera" },
            ],
        }),
    },

    script: ({ call, loop, yieldFrame }) => {
        loop(() => {
            call.RenderOrbital();
            yieldFrame();
        });
    },
});
