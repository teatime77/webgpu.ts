namespace webgputs {

class UI3D {
    autoRotate : boolean = false;

    eye : any;

    camPhi : number = 0.0;
    camTheta : number = 0.5 * Math.PI;
    camDistance    : number = -5.0;

    lastMouseX : number | null = null;
    lastMouseY : number | null = null;


    constructor(canvas : HTMLCanvasElement, eye : any){
        this.eye = eye;
        this.camDistance = eye[2];

        canvas.addEventListener('pointermove', (ev: PointerEvent)=> {

            // タッチによる画面スクロールを止める
            // ev.preventDefault(); 

            var newX = ev.clientX;
            var newY = ev.clientY;

            if (ev.buttons != 0 && this.lastMouseX != null && this.lastMouseY != null) {

                this.camTheta += (newY - this.lastMouseY) / 300;
                this.camPhi -= (newX - this.lastMouseX) / 300;
            }
            // console.log(`phi:${this.camPhi} theta:${this.camTheta}`)

            this.lastMouseX = newX
            this.lastMouseY = newY;
        }, { passive: true });
    
    
        canvas.addEventListener("wheel",  (ev: WheelEvent)=> {
    
            this.camDistance += 0.002 * ev.deltaY;
    
            // ホイール操作によるスクロールを無効化する
            // ev.preventDefault();
        }, { passive: true });

        const auto_rotate = document.getElementById("auto-rotate") as HTMLInputElement;
        this.autoRotate = auto_rotate.checked;

        auto_rotate.addEventListener("click", (ev : MouseEvent)=>{
            this.autoRotate = auto_rotate.checked;            
        });
    }    

    getTransformationMatrix() {
        if(this.autoRotate){
            return this.getAutoTransformationMatrix();
        }
        else{
            return this.getManualTransformationMatrix();
        }
    }    

    getAutoTransformationMatrix() {
        const projectionMatrix = glMatrix.mat4.create();
        glMatrix.mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, 1, 1, 100.0);
    
        const viewMatrix = glMatrix.mat4.create();
        glMatrix.mat4.translate(viewMatrix, viewMatrix, this.eye);
    
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
    
    getManualTransformationMatrix() {
        const camY = this.camDistance * Math.cos(this.camTheta);
        const r = this.camDistance * Math.abs(Math.sin(this.camTheta));
        const camZ = r * Math.cos(this.camPhi);
        const camX = r * Math.sin(this.camPhi);

        const projectionMatrix = glMatrix.mat4.create();
        glMatrix.mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, 1, 1.0, 100.0);

        const viewMatrix = glMatrix.mat4.create();
        const cameraPosition = [camX, camY, camZ];
        const lookAtPosition = [0, 0, 0];
        const upDirection    = [0, 1, 0];
        glMatrix.mat4.lookAt(viewMatrix, cameraPosition, lookAtPosition, upDirection);

        const worldMatrix = glMatrix.mat4.create();
        // const now = Date.now() / 1000;
        // glMatrix.mat4.rotate(
        //     worldMatrix,
        //     worldMatrix,
        //     1,
        //     glMatrix.vec3.fromValues(Math.sin(now), Math.cos(now), 0)
        // );

        const pvw = glMatrix.mat4.create();
        glMatrix.mat4.mul(pvw, projectionMatrix, viewMatrix);
        glMatrix.mat4.mul(pvw, pvw, worldMatrix);

        return pvw;
    }

}

export let ui3D : UI3D;

export function initUI3D(canvas : HTMLCanvasElement, eye : any){
    ui3D = new UI3D(canvas, eye);
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