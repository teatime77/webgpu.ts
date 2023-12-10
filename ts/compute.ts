
async function asyncBodyOnLoadCom() {
    const shader = `
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
`;
    
    if (!navigator.gpu) {
        throw Error('WebGPU not supported.');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw Error('Couldn\'t request WebGPU adapter.');
    }

    const device = await adapter!.requestDevice();

    // シェーダーコードを準備
    const shaderModule = device.createShaderModule({
        code: shader
    });

    // 入力配列を作成
    const inputArray = new Float32Array([1, 2, 3, 4]);

    // Compute Shader用のバッファーを作成
    // 入力用バッファ
    const inputBuffer = device.createBuffer({
        size: inputArray.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });

    // 入力データをinputBufferに書き込み
    device.queue.writeBuffer(inputBuffer, 0, inputArray);

    // 出力用バッファ
    const output = device.createBuffer({
        size: inputArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Get a GPU buffer for reading in an unmapped state.
    const stagingBuffer = device.createBuffer({
        mappedAtCreation: false,
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout =
        device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }]
        });

    // bindGroupを生成
    /*
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            // { binding: 0, resource: { buffer: inputBuffer } },
            // { binding: 1, resource: { buffer: outputBuffer } }
        ]
    });
    */

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: output,
            }
        }]
    });

    // ComputePipelineを生成
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // 5: Create GPUCommandEncoder to issue commands to the GPU
    const commandEncoder = device.createCommandEncoder();

    // 6: Initiate render pass
    const passEncoder = commandEncoder.beginComputePass();

    // 7: Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupSize = 4;
    passEncoder.dispatchWorkgroups(inputArray.length / workgroupSize);

    // End the render pass
    passEncoder.end();

    // Encode commands for copying buffer to buffer.
    // const copyEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        output, //source buffer,
        0, //source offset,
        stagingBuffer, //destination buffer,
        0, //destination offset,
        16 //size
    );

    // 8: End frame by passing array of command buffers to command queue for execution
    device.queue.submit([commandEncoder.finish()]);

    // const copyCommands = copyEncoder.finish();
    // device.queue.submit([copyCommands]);

    // 結果を取得
    console.log(`staging buffer size:${stagingBuffer.size}`);

    // map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ, 
        0, 
        stagingBuffer.size
    );

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, stagingBuffer.size);
    const data = copyArrayBuffer.slice(0);
    stagingBuffer.unmap();
    console.log(new Float32Array(data));  // [2, 4, 6, 8]

}
