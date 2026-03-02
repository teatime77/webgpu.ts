import { fetchText, range, assert } from "@i18n";
import { Module } from "./parser.js";
import { ui3D } from "./ui.js";
import { makeShaderModule, g_device, particleDim, fetchModule } from "./util.js";

export async function asyncBodyOnLoadCom() {
    const shader = await fetchText('./wgsl/compute.wgsl');

    // シェーダーコードを準備
    const shaderModule = makeShaderModule(shader);

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

    const comp = new ComputePipeline("compute", [], inputArray.length);

    await comp.makeComputePipeline();

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

export abstract class AbstractPipeline {
    bindGroupLayout!: GPUBindGroupLayout;
    bindGroups    : GPUBindGroup[] = new Array(2);
    uniformBuffer! : GPUBuffer;

    constructor(){
    }

    makeUniformBuffer(uniform_buffer_size : number){
        this.uniformBuffer = g_device.createBuffer({
            size: uniform_buffer_size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });    
    }

    writeUniformBuffer(data : Float32Array, offset : number){
        g_device.queue.writeBuffer(this.uniformBuffer, offset, data as any);

        return offset + data.byteLength;
    }
}

export class ComputePipeline extends AbstractPipeline {
    compName : string;
    varNames : string[];
    instanceArray : Float32Array;
    instanceCount : number;
    workgroupCounts : [number, number, number] | null = null;

    pipeline!     : GPUComputePipeline;
    compModule!   : Module;
    updateBuffers : GPUBuffer[] = new Array(2);

    constructor(comp_name : string, var_names : string[], instance_array_length : number){
        super();
        this.compName = comp_name;
        this.varNames = var_names;
        this.instanceArray = new Float32Array(instance_array_length);

        this.instanceCount = this.instanceArray.length / particleDim;   // Math.floor(this.array.length / 2);
    }

    async initCompute(){
        await this.makeComputePipeline();
        this.makeUniformBuffer(ui3D.env.byteLength);
        this.makeUpdateBuffers();
    }

    async makeComputePipeline(){
        this.compModule = await fetchModule(this.compName);

        if(this.compName == "compute"){

            this.bindGroupLayout = g_device.createBindGroupLayout({
                label: "Bind-Group-Layout (Input + Output)",
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" }, // The Input
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "storage" },           // The Output
                    },
                ],
            });
        }
        else{

            this.bindGroupLayout = g_device.createBindGroupLayout({
                label: "Combined Layout (Uniforms + Ping-Pong)",
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },          // The Shared Uniform
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" }, // The Input
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "storage" },           // The Output
                    },
                ],
            });
        }

        const pipelineLayout = g_device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout] // Index 0 matches group(0) in shader
        });

        this.pipeline = g_device.createComputePipeline({
            // layout: 'auto',
            layout: pipelineLayout,
            compute: {
                module: this.compModule.module,
                entryPoint: 'main'
            }
        });
    }

    makeUpdateBuffers(){
    
        for (let i = 0; i < 2; ++i) {
            this.updateBuffers[i] = g_device.createBuffer({
                size: this.instanceArray.byteLength,
                usage: GPUBufferUsage.STORAGE,
                mappedAtCreation: true,
            });
            new Float32Array(this.updateBuffers[i].getMappedRange()).set(
                this.instanceArray
            );
            this.updateBuffers[i].unmap();
        }

        assert(this.compModule.vars.length == 3);
        for(const [i, v] of this.compModule.vars.entries()){
            assert(v.mod.binding == i);
        }

        for (let i = 0; i < 2; ++i) {
            this.bindGroups[i] = g_device.createBindGroup({
                // layout: this.pipeline.getBindGroupLayout(0),
                layout: this.bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.uniformBuffer,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.updateBuffers[i],
                            offset: 0,
                            size: this.instanceArray.byteLength,
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.updateBuffers[(i + 1) % 2],
                            offset: 0,
                            size: this.instanceArray.byteLength,
                        },
                    },
                ],
            });
        }
    }
}
