namespace webgpu_ts {

// Define global buffer size
const BUFFER_SIZE = 1000;

// Main function

export async function asyncBodyOnLoadDemo() {

    // Compute shader
    const shader = await fetchText('../wgsl/demo.wgsl');

    // 2: Create a shader module from the shader template literal
    const shaderModule = makeShaderModule(shader);

    // 3: Create an output buffer to read GPU calculations to, and a staging buffer to be mapped for JavaScript access

    const output = g_device.createBuffer({
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const stagingBuffer = g_device.createBuffer({
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // 4: Create a GPUBindGroupLayout to define the bind group structure, create a GPUBindGroup from it,
    // then use it to create a GPUComputePipeline

    const bindGroupLayout =
        g_device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            }]
        });

    const bindGroup = g_device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: output,
            }
        }]
    });

    const computePipeline = g_device.createComputePipeline({
        layout: g_device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // 5: Create GPUCommandEncoder to issue commands to the GPU
    const commandEncoder = g_device.createCommandEncoder();

    // 6: Initiate render pass
    const passEncoder = commandEncoder.beginComputePass();

    // 7: Issue commands
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(BUFFER_SIZE / 64));

    // End the render pass
    passEncoder.end();

    // Copy output buffer to staging buffer
    commandEncoder.copyBufferToBuffer(
        output,
        0, // Source offset
        stagingBuffer,
        0, // Destination offset
        BUFFER_SIZE
    );

    // 8: End frame by passing array of command buffers to command queue for execution
    g_device.queue.submit([commandEncoder.finish()]);

    // map staging buffer to read results back to JS
    await stagingBuffer.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        BUFFER_SIZE // Length
    );

    const copyArrayBuffer = stagingBuffer.getMappedRange(0, BUFFER_SIZE);
    const data = copyArrayBuffer.slice(0);
    stagingBuffer.unmap();
    console.log(new Float32Array(data));
}

}