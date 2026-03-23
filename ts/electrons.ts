import { $, fetchText } from "@i18n";
import { g_device, initContext } from "./util";
import { initUI3D } from "./ui";

function getComputeShader(orbitalType: string): string {
  // We define the specific math for each orbital here.
  // We use scaled versions to make them look good in our -15 to +15 bounding box.
  let orbitalMath: string = '';

  switch (orbitalType) {
    case '1s':
      // The 1s orbital is a simple, dense sphere peaking at the nucleus.
      orbitalMath = `
        let psi = exp(-r / a0);
        var density = psi * psi;
      `;
      break;
    
    case '2pz':
      // The 2p_z orbital has two lobes along the Z axis and a nodal plane at XY.
      orbitalMath = `
        let psi = (r / a0) * exp(-r / (2.0 * a0)) * cos(theta);
        var density = psi * psi * 0.5; 
      `;
      break;

    case '3dz2':
      // The 3d_z^2 orbital has a central "donut" and two large Z-axis lobes.
      orbitalMath = `
        let cos2 = cos(theta) * cos(theta);
        let psi = (r * r / (a0 * a0)) * exp(-r / (3.0 * a0)) * (3.0 * cos2 - 1.0);
        var density = psi * psi * 0.05; // Scaled down because the math yields higher raw values
      `;
      break;
      
    default:
      orbitalMath = `var density = 0.0;`;
  }

  // Return the complete WGSL shader, injecting the chosen math.
  return `
    @group(0) @binding(0) var volume_tex: texture_storage_3d<rgba16float, write>;

    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let dimensions = vec3<f32>(textureDimensions(volume_tex));
        let grid_pos = vec3<f32>(id);

        if (grid_pos.x >= dimensions.x || grid_pos.y >= dimensions.y || grid_pos.z >= dimensions.z) {
            return;
        }

        let normalized_pos = grid_pos / dimensions;
        let physical_pos = (normalized_pos - 0.5) * 30.0; 

        let r = max(length(physical_pos), 0.00001);
        let theta = acos(physical_pos.z / r); 
        let a0 = 1.0; 
        
        // --- INJECTED ORBITAL MATH ---
        ${orbitalMath}
        // -----------------------------

        textureStore(volume_tex, id, vec4<f32>(density, 0.0, 0.0, 0.0));
    }
  `;
}

function computeVolume(
    device: GPUDevice,
    volumeTexture: GPUTexture,
    computeShaderCode: string
){
    // 1. Create the shader module
    const computeModule: GPUShaderModule = device.createShaderModule({ code: computeShaderCode });

    // 2. Create the compute pipeline
    const computePipeline: GPUComputePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: computeModule,
            entryPoint: 'main',
        },
    });

    // 3. Create a bind group holding your 3D texture view
    const bindGroup: GPUBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: volumeTexture.createView(),
            },
        ],
    });

    // 4. Dispatch the compute shader
    const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();
    const passEncoder: GPUComputePassEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Calculate workgroups based on texture size and @workgroup_size(8, 8, 8)
    const width: number = volumeTexture.width;
    const height: number = volumeTexture.height;
    const depth: number = volumeTexture.depthOrArrayLayers;

    passEncoder.dispatchWorkgroups(
        Math.ceil(width  / 4),
        Math.ceil(height / 4),
        Math.ceil(depth  / 4)
    );

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

function setupRender(
    device: GPUDevice,
    context: GPUCanvasContext,
    volumeTexture: GPUTexture,
    renderShaderCode: string
){
    const sampler: GPUSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
    });

    // 1. Create the Uniform Buffer (64 bytes = 16 floats * 4 bytes each)
    const uniformBuffer: GPUBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const renderModule: GPUShaderModule = device.createShaderModule({ code: renderShaderCode });

    const renderPipeline: GPURenderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: renderModule,
            entryPoint: 'vs_main',
        },
        fragment: {
            module: renderModule,
            entryPoint: 'fs_main',
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
        primitive: { topology: 'triangle-list' },
    });

    // 2. Add the Uniform Buffer to the Bind Group (binding: 2)
    const renderBindGroup: GPUBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: volumeTexture.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: uniformBuffer } }, // <-- NEW
        ],
    });

    // A typed array to hold our 4x4 matrix
    const matrixData: Float32Array = new Float32Array(16);

    function frame(time: number): void {
        // 3. Calculate a Y-axis rotation matrix based on time
        const angle: number = time * 0.001; // Convert ms to seconds
        const cosT: number = Math.cos(angle);
        const sinT: number = Math.sin(angle);

        // Column-major 4x4 matrix for Y-axis rotation
        matrixData.set([
            cosT, 0.0, sinT, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sinT, 0.0, cosT, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]);

        // 4. Send the new matrix to the GPU
        device.queue.writeBuffer(uniformBuffer, 0, matrixData as BufferSource);

        const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();
        const currentTexture: GPUTexture = context.getCurrentTexture();
        const textureView: GPUTextureView = currentTexture.createView();

        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        };

        const passEncoder: GPURenderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(renderPipeline);
        passEncoder.setBindGroup(0, renderBindGroup);
        passEncoder.draw(3);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

export async function showElectrons(){
    $("orbital-panel").style.display = "inline-block";

    const canvas = $('world') as HTMLCanvasElement;
    const context = initContext(canvas, 'opaque');

    initUI3D(canvas);

    // Assuming 'device' is already initialized as a GPUDevice
    const volumeTexture: GPUTexture = g_device.createTexture({
        size: [128, 128, 128], // width, height, depth
        dimension: '3d',
        format: 'rgba16float', // <-- Changed to a filterable 16-bit float
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });

    // const compShader = await fetchText('./wgsl/electrons-comp.wgsl');
    const renderShader = await fetchText('./wgsl/electrons-vert-frag.wgsl');


    const selectElement = document.getElementById('orbital-select') as HTMLSelectElement;

    // 1. A helper function to update the 3D texture
    function updateOrbital(orbitalValue: string){
        const newShaderCode: string = getComputeShader(orbitalValue);

        // We re-run the compute pass to overwrite the 3D texture with new data
        computeVolume(g_device, volumeTexture, newShaderCode);
    }
    // await computeVolume(g_device, volumeTexture, compShader);

    // 2. Listen for UI changes
    selectElement.addEventListener('change', (event: Event) => {
        const target = event.target as HTMLSelectElement;
        updateOrbital(target.value);
    });

    // 3. Initial setup
    // Run the compute shader for whatever is currently selected in the HTML
    updateOrbital(selectElement.value);

    // Start the render loop (it just keeps reading the volumeTexture, 
    // so it will automatically show the new orbital as soon as compute finishes!)
    setupRender(g_device, context, volumeTexture, renderShader);
}


// ===================================================================
