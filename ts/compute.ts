namespace webgputs {


export async function asyncBodyOnLoadCom() {
    const shader = await fetchText('../wgsl/compute.wgsl');

    // シェーダーコードを準備
    const shaderModule = g_device.createShaderModule({
        code: shader
    });

    // 入力配列を作成
    const inputArray = new Float32Array(Array.from(range(16)));

    // Compute Shader用のバッファーを作成
    // 入力用バッファ
    const inputBuffer = g_device.createBuffer({
        size: inputArray.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });

    // 入力データをinputBufferに書き込み
    g_device.queue.writeBuffer(inputBuffer, 0, inputArray);

    // 出力用バッファ
    const outputBuffer = g_device.createBuffer({
        size: inputArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Get a GPU buffer for reading in an unmapped state.
    const stagingBuffer = g_device.createBuffer({
        mappedAtCreation: false,
        size: outputBuffer.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const comp = new Compute();

    await comp.makePipeline("compute");

    // ComputePipelineを生成

    const bindGroup = g_device.createBindGroup({
        layout: comp.pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: inputBuffer,
                }
            }
            ,
            {
                binding: 1,
                resource: {
                    buffer: outputBuffer,
                }
            }
        ]
    });

    // 5: Create GPUCommandEncoder to issue commands to the GPU
    const commandEncoder = g_device.createCommandEncoder();

    // 6: Initiate render pass
    const passEncoder = commandEncoder.beginComputePass();

    // 7: Issue commands
    passEncoder.setPipeline(comp.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupSize = 4;
    passEncoder.dispatchWorkgroups(inputArray.length / workgroupSize);

    // End the render pass
    passEncoder.end();

    // Encode commands for copying buffer to buffer.
    // const copyEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        outputBuffer, //source buffer,
        0, //source offset,
        stagingBuffer, //destination buffer,
        0, //destination offset,
        outputBuffer.size
    );

    // 8: End frame by passing array of command buffers to command queue for execution
    g_device.queue.submit([commandEncoder.finish()]);

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

export class Compute extends Pipeline {
    pipeline! : GPUComputePipeline;
    updateBuffers: GPUBuffer[] = new Array(2);
    bindGroups: GPUBindGroup[] = new Array(2)

    async makePipeline(shader_name: string){
        const shader_module = await fetchModule(shader_name);

        this.pipeline = g_device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shader_module.module,
                entryPoint: 'main'
            }
        });
    }

    makeUpdateBuffers(initial_update_Data : Float32Array){
        for (let i = 0; i < 2; ++i) {
            this.updateBuffers[i] = g_device.createBuffer({
                size: initial_update_Data.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
                mappedAtCreation: true,
            });
            new Float32Array(this.updateBuffers[i].getMappedRange()).set(
                initial_update_Data
            );
            this.updateBuffers[i].unmap();
        }    
    }
}

}