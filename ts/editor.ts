export class Editor {
    camera : Camera = new Camera();
    tool : Tool = this.camera;
}

export let editor : Editor;

export function initEditor(){
    editor = new Editor();
}

abstract class Tool {
    lastMouseX : number | null = null;
    lastMouseY : number | null = null;

    pointerdown(ev: PointerEvent){

    }

    pointermove(ev: PointerEvent){
    }

    pointerup(ev: PointerEvent){
    }

    wheel(ev: WheelEvent){
    }


    click(ev: MouseEvent){
    }

}

class Camera extends Tool {
    camPhi : number = 0.0;
    camTheta : number = 0.5 * Math.PI;
    camDistance    : number = -5.0;

    pointermove(ev: PointerEvent){
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
    }

    pointerup(ev: PointerEvent){
    }

    wheel(ev: WheelEvent){
        this.camDistance += 0.002 * ev.deltaY;

        // ホイール操作によるスクロールを無効化する
        ev.preventDefault();
    }

    setAutoAngle(milliseconds : number){
        const cnt = 10 * 1000;

        // i changes from 0 to 2999 in 3000 milliseconds.
        const i = Math.round(milliseconds) % cnt;

        this.camPhi     = 2.0 * Math.PI * i / cnt;
        this.camTheta   = Math.PI / 3.0;
    }

    cameraPos() : [number, number, number] {
        const camY = this.camDistance * Math.cos(this.camTheta);
        const r = this.camDistance * Math.abs(Math.sin(this.camTheta));
        const camZ = r * Math.cos(this.camPhi);
        const camX = r * Math.sin(this.camPhi);

        return [camX, camY, camZ];
    }

}
