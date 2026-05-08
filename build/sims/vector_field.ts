import {
    defineSimulation,
    storage, uniform,
    compute, render,
} from "../../ts/builder/index";

export default defineSimulation({
    name: "vector_field",
    version: "1.0",

    metadata: {
        gridWidth: 20,
        gridHeight: 20,
        time: 0.0,
        baseArrowFloatCount: 0,
        baseArrowVertexCount: 0,
    },

    resources: {
        Params: uniform({
            gridWidth: "f32",
            gridHeight: "f32",
            time: "f32",
        }),
        Camera: uniform({
            viewProjection: "mat4x4<f32>",
            view: "mat4x4<f32>",
        }),
        VectorFieldData: storage({
            access: "read_write",
            format: "vec4<f32>",
            count: "$metadata.gridWidth * $metadata.gridHeight",
        }),
        BaseArrow: storage({
            access: "read",
            format: "f32",
            count: "$metadata.baseArrowFloatCount",
        }),
    },

    nodes: {
        compute_vector_field: compute({
            shader: "compute_vector_field",
            workgroups: ["$metadata.gridWidth / 8 + 1", "$metadata.gridHeight / 8 + 1", 1],
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "VectorFieldData" },
            ],
        }),
        render_vector_field: render({
            shader: "render_vector_field",
            topology: "triangle-list",
            depthTest: true,
            vertexCount: "$metadata.baseArrowVertexCount",
            instanceCount: "$metadata.gridWidth * $metadata.gridHeight",
            bindings: [
                { binding: 0, resource: "Params" },
                { binding: 1, resource: "Camera" },
                { binding: 2, resource: "VectorFieldData", access: "read" },
                { binding: 3, resource: "BaseArrow", access: "read" },
            ],
        }),
    },

    script: ({ call, loop, yieldFrame }) => {
        loop(() => {
            call.compute_vector_field();
            call.render_vector_field();
            yieldFrame();
        });
    },
});
