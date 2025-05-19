namespace webgpu_ts {

const $dic = new Map<string, HTMLElement>();

export function $(id : string) : HTMLElement {
    let ele = $dic.get(id);
    if(ele == undefined){
        ele = document.getElementById(id)!;
        $dic.set(id, ele);
    }

    return ele;
}

export function $inp(id : string) : HTMLInputElement {    
    return $(id) as HTMLInputElement;
}

export function mat4fromMat3(m3 : Float32Array){
    const m4 = glMatrix.mat4.create();

    m4[ 0] = m3[0];
    m4[ 1] = m3[1];
    m4[ 2] = m3[2];
    m4[ 3] = 0;
    
    m4[ 4] = m3[3];
    m4[ 5] = m3[4];
    m4[ 6] = m3[5];
    m4[ 7] = 0;
    
    m4[ 8] = m3[6];
    m4[ 9] = m3[7];
    m4[10] = m3[8];
    m4[11] = 0;
    
    // Set the last row and column to identity values
    m4[15] = 1;

    return m4;
}

class UI3D {
    autoRotate : boolean = false;

    ProjViewMatrix!               : Float32Array;
    viewMatrix!        : Float32Array;

    ambientColor!      : Float32Array;
    directionalColor!  : Float32Array;
    lightingDirection! : Float32Array;

    startTime          : number;
    tick               : number = 0;
    env                : Float32Array = new Float32Array([0,0,0,0]);

    constructor(canvas : HTMLCanvasElement){
        this.startTime = Date.now();

        canvas.addEventListener('pointermove', this.pointermove.bind(this));
        canvas.addEventListener("wheel", this.wheel.bind(this));

        $inp("auto-rotate").addEventListener("click", this.click.bind(this));

        this.autoRotate = $inp("auto-rotate").checked;
    }

    click(ev : MouseEvent){
        this.autoRotate = $inp("auto-rotate").checked;
    }

    pointermove(ev: PointerEvent){
        editor.tool.pointermove(ev);
    }

    wheel(ev: WheelEvent){
        editor.tool.wheel(ev);
    }

    getTransformationMatrix() {

        if(this.autoRotate){
            editor.camera.setAutoAngle(Date.now() - this.startTime);
            this.setViewMatrix();
        }
        this.setViewMatrix();

        const projectionMatrix = glMatrix.mat4.create();
        glMatrix.mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, 1, 1, 100.0);

        this.ProjViewMatrix = glMatrix.mat4.create();
        glMatrix.mat4.mul(this.ProjViewMatrix, projectionMatrix, this.viewMatrix);
    }
    
    setViewMatrix() {

        this.viewMatrix = glMatrix.mat4.create();
        const cameraPosition = editor.camera.cameraPos();
        const lookAtPosition = [0, 0, 0];
        const upDirection    = [0, 1, 0];
        glMatrix.mat4.lookAt(this.viewMatrix, cameraPosition, lookAtPosition, upDirection);
    }

    setEnv(){
        const elapsed_time = Math.round(Date.now() - this.startTime);
        this.env[0] = elapsed_time;
        this.env[1] = this.tick;

        this.tick++;
    }

    setTransformationMatrixAndLighting(){
        // @uniform
        this.getTransformationMatrix();

        this.ambientColor      = getColor("ambient");
        this.directionalColor  = getColor("directional");
        this.lightingDirection = glMatrix.vec4.create();
        glMatrix.vec3.normalize( this.lightingDirection, glMatrix.vec4.fromValues(0.25, 0.25, 1, 0) );
    }

}

export let ui3D : UI3D;

export function initUI3D(canvas : HTMLCanvasElement){
    ui3D = new UI3D(canvas);
}

export function makeLightDir(){
    const v = glMatrix.vec3.create();

    v[0] = 1.0;
    v[1] = 1.0;
    v[2] = 1.0;

    glMatrix.vec3.normalize(v, v);

    return v;
}

}