struct Input {
    data: array<f32>,
};

struct Output {
    data: array<f32>,
};

@group(0) @binding(0)
var<storage, read> input : Input;

@group(0) @binding(1)
var<storage, read_write> output : Output;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    output.data[global_id.x] = input.data[global_id.x] * 2.0;
}
