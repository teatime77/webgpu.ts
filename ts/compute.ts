import { fetchText, range, assert, MyError, msg } from "@i18n";
import { makeShaderModule, g_device, fetchModule, number123 } from "./util.js";
import { getReadStorageVars, Struct, getUniformVars, getWriteStorageVars, Module } from "./syntax"
import { ui3D } from "./ui.js";

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

    const comp = new ComputePipeline("compute");

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
    bindGroups    : GPUBindGroup[] = [];
    uniformBuffer! : GPUBuffer;

    constructor(){
    }

    abstract writeUniform() : void;

    getUniformBufferSize(){
        return ui3D.env.byteLength;
    }

    makeUniformBuffer(){
        const uniform_buffer_size = this.getUniformBufferSize();

        this.uniformBuffer = g_device.createBuffer({
            // The pipeline requires a buffer binding which is at least 256 bytes
            size: Math.max(256, uniform_buffer_size),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });    
    }

    writeUniformBuffer(data : Float32Array, offset : number){
        g_device.queue.writeBuffer(this.uniformBuffer, offset, data as any);

        return offset + data.byteLength;
    }

    writeUniformBufferFloat32(data : number, offset : number){
        const buffer = new Float32Array([data]);
        g_device.queue.writeBuffer(this.uniformBuffer, offset, buffer);

        return offset + 4;
    }

    swapBindGroups(){
        if(this.bindGroups.length == 2){
            const tmp = this.bindGroups[0];
            this.bindGroups[0] = this.bindGroups[1];
            this.bindGroups[1] = tmp;
        }
    }
}

export class ComputePipeline extends AbstractPipeline {
    compName : string;
    globalGrid : number | [number, number] | [number, number, number] | undefined;
    instanceArray! : Float32Array;
    instanceCount! : number;
    workgroupCounts : number123 | null = null;

    pipeline!     : GPUComputePipeline;
    compModule!   : Module;
    updateBuffers : GPUBuffer[] = [];

    constructor(comp_name : string){
        super();
        this.compName = comp_name;
    }

    async initCompute(){
        this.makeUniformBuffer();
        this.makeUpdateBuffers();
    }

    async makeComputePipeline(){
        this.compModule = await fetchModule(this.compName);

        let binding = 0;
        const entries : GPUBindGroupLayoutEntry[] = [];
        if(getUniformVars(this.compModule).length == 1){
            entries.push({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "uniform" },
            });

            binding++;
        }

        if(getReadStorageVars(this.compModule).length == 1){
            entries.push({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" }, // The Input
            });

            binding++;
        }

        if(getWriteStorageVars(this.compModule).length == 1){
            entries.push({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },           // The Output
            });

            binding++;
        }

        this.bindGroupLayout = g_device.createBindGroupLayout({
            label: "Compute Bind Group Layout",
            entries,
        });

        const pipelineLayout = g_device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout] // Index 0 matches group(0) in shader
        });

        const compShaderModule = makeShaderModule(this.compModule.text);

        this.pipeline = g_device.createComputePipeline({
            // layout: 'auto',
            layout: pipelineLayout,
            compute: {
                module: compShaderModule,
                entryPoint: 'main'
            }
        });
    }

    makeUpdateBuffers(){
        assert(getReadStorageVars(this.compModule).length == 1 && getWriteStorageVars(this.compModule).length == 1);
        this.updateBuffers = [];
        this.bindGroups    = [];

        const storageCount = getReadStorageVars(this.compModule).length + getWriteStorageVars(this.compModule).length;
    
        for (let i = 0; i < storageCount; ++i) {
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

        for(let i = 0; i < storageCount; ++i) {
            const entries : GPUBindGroupEntry[] = [];
            let binding = 0;

            if(getUniformVars(this.compModule).length == 1){
                entries.push({
                    binding,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                });

                binding++;
            }

            if(getReadStorageVars(this.compModule).length == 1){
                entries.push({
                    binding,
                    resource: {
                        buffer: this.updateBuffers[i],
                        offset: 0,
                        size: this.instanceArray.byteLength,
                    },
                });

                binding++;
            }

            if(getWriteStorageVars(this.compModule).length == 1){
                entries.push({
                    binding,
                    resource: {
                        buffer: this.updateBuffers[(i + 1) % 2],
                        offset: 0,
                        size: this.instanceArray.byteLength,
                    },
                });

                binding++;
            }

            const bindGroup = g_device.createBindGroup({
                // layout: this.pipeline.getBindGroupLayout(0),
                layout: this.bindGroupLayout,
                entries
            });

            this.bindGroups.push(bindGroup);
        }
    }

    writeUniform() : void {
        let offset = 0;
        const uniform_var_struct = this.compModule.getUniformVar().type as Struct;
        for(const member of uniform_var_struct.members){
            switch(member.name){
            case "env":
                offset = this.writeUniformBuffer(ui3D.env, offset);
                break;
            default:
                throw new MyError(`unknown uniform:${member.name}`);
            }
        }

    }
}
