namespace webgputs {

export function initContext(g_device : GPUDevice, canvas: HTMLCanvasElement, alpha_mode : GPUCanvasAlphaMode) : [GPUCanvasContext, GPUTextureFormat] {
    const context = canvas.getContext('webgpu') as GPUCanvasContext;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    const devicePixelRatio = window.devicePixelRatio || 1;

    canvas.width  = canvas.clientWidth  * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;

    context.configure({
        device: g_device,
        format: presentationFormat,
        alphaMode: 'opaque',
    });

    return [context, presentationFormat];
}


export function getTransformationMatrix(eye : any) {
    const projectionMatrix = glMatrix.mat4.create();
    glMatrix.mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, 1, 1, 100.0);

    const viewMatrix = glMatrix.mat4.create();
    glMatrix.mat4.translate(viewMatrix, viewMatrix, eye);

    const worldMatrix = glMatrix.mat4.create();
    const now = Date.now() / 1000;
    glMatrix.mat4.rotate(
        worldMatrix,
        worldMatrix,
        1,
        glMatrix.vec3.fromValues(Math.sin(now), Math.cos(now), 0)
    );

    const pvw = glMatrix.mat4.create();
    glMatrix.mat4.mul(pvw, projectionMatrix, viewMatrix);
    glMatrix.mat4.mul(pvw, pvw, worldMatrix);

    return pvw;
}

export function queueTransformationMatrix(g_device : GPUDevice, uniformBuffer: GPUBuffer, eye : any){
    const pvw = getTransformationMatrix(eye);

    g_device.queue.writeBuffer(
        uniformBuffer,
        4 * 16 * 2,
        pvw.buffer,
        pvw.byteOffset,
        pvw.byteLength
    );
}

}