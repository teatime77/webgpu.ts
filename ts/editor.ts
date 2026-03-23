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

    private pointerCache = new Map<number, PointerEvent>();
    private prevPinchDist: number | null = null;

    pointerdown(ev: PointerEvent) {
        this.lastMouseX = ev.clientX;
        this.lastMouseY = ev.clientY;
        this.pointerCache.set(ev.pointerId, ev);
    }

    pointermove(ev: PointerEvent){
        ev.preventDefault();

        // Update the pointer's position in the cache
        if (this.pointerCache.has(ev.pointerId)) {
            this.pointerCache.set(ev.pointerId, ev);
        }

        if (this.pointerCache.size === 2) {
            // Two-finger pinch for zooming
            const pointers = Array.from(this.pointerCache.values());
            const p1 = pointers[0];
            const p2 = pointers[1];

            const dx = p1.clientX - p2.clientX;
            const dy = p1.clientY - p2.clientY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (this.prevPinchDist !== null) {
                const delta = dist - this.prevPinchDist;
                this.camDistance += delta * 0.02; // Sensitivity factor
            }

            this.prevPinchDist = dist;
            
            // Reset one-finger drag state to prevent rotation while pinching
            this.lastMouseX = null;
            this.lastMouseY = null;
        } else if (this.pointerCache.size === 1 && ev.buttons !== 0) {
            // One-finger drag for rotation
            this.prevPinchDist = null; // Ensure pinch state is reset

            if (this.lastMouseX !== null && this.lastMouseY !== null) {
                const newX = ev.clientX;
                const newY = ev.clientY;
                this.camTheta += (newY - this.lastMouseY) / 300;
                this.camPhi -= (newX - this.lastMouseX) / 300;
            }
            
            // For the next move event
            this.lastMouseX = ev.clientX;
            this.lastMouseY = ev.clientY;
        }
    }

    pointerup(ev: PointerEvent){
        this.pointerCache.delete(ev.pointerId);
        if (this.pointerCache.size < 2) {
            this.prevPinchDist = null;
        }
        if (this.pointerCache.size < 1) {
            this.lastMouseX = null;
            this.lastMouseY = null;
        }
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
