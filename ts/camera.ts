import { mat4, vec3 } from 'gl-matrix';

export class OrbitCamera {
    private canvas: HTMLCanvasElement;
    
    // --- カメラの極座標パラメータ ---
    public target = [0.0, 0.0, 0.0]; // 注視点 (パンで移動)
    public distance: number = 40.0;        // カメラとの距離 (ズームで変化)
    public theta: number = 0.0;            // 水平角度 (回転で変化)
    public phi: number = Math.PI / 3;      // 垂直角度 (回転で変化)

    // --- マウス操作の内部状態 ---
    private isDragging = false;
    private dragButton = -1;
    private lastMouseX = 0;
    private lastMouseY = 0;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.attachEvents();
    }

    private attachEvents() {
        // マウスを押した時
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.dragButton = e.button;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        // マウスを離した時 (画面外に出た時も考慮してwindowに付与)
        window.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.dragButton = -1;
        });

        // マウスを動かした時
        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;

            if (this.dragButton === 0) { 
                // 【左クリック：回転 (Orbit)】
                this.theta -= dx * 0.01;
                this.phi -= dy * 0.01;
                // phiの制限 (真上・真下を越えて反転しないように)
                const epsilon = 0.001;
                this.phi = Math.max(epsilon, Math.min(Math.PI - epsilon, this.phi));
            } 
            else if (this.dragButton === 2) { 
                // 【右クリック：パン (Pan)】
                // 🌟 修正: 現在の角度から「前」「右」「上」のベクトルを再計算する
                
                // 1. カメラが向いている方向（前）のベクトル
                const forward = vec3.fromValues(
                    -Math.sin(this.phi) * Math.sin(this.theta),
                    -Math.cos(this.phi),
                    -Math.sin(this.phi) * Math.cos(this.theta)
                );
                
                // 2. 世界の上方向（Y軸）
                const worldUp = vec3.fromValues(0, 1, 0);
                
                // 3. 右ベクトル (前 × 上 の外積)
                const right = vec3.create();
                vec3.cross(right, forward, worldUp);
                vec3.normalize(right, right); // 念のため正規化
                
                // 4. カメラの本当の上ベクトル (右 × 前 の外積)
                const up = vec3.create();
                vec3.cross(up, right, forward);
                vec3.normalize(up, up);

                // 5. ターゲットの座標を移動させる
                const panSpeed = this.distance * 0.002;
                vec3.scaleAndAdd(this.target, this.target, right, -dx * panSpeed);
                vec3.scaleAndAdd(this.target, this.target, up, dy * panSpeed);
            }
        });

        // ホイールを回した時
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            // 【ホイール：ズーム (Zoom)】
            this.distance += e.deltaY * this.distance * 0.001;
            this.distance = Math.max(1.0, this.distance); // 近づきすぎ防止
        }, { passive: false });
        
        // 右クリック時のブラウザメニューを無効化
        this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    }

    // 極座標から実際のカメラの[x, y, z]座標を計算する
    private getEyePosition(): [number,number,number] {
        const x = this.target[0] + this.distance * Math.sin(this.phi) * Math.sin(this.theta);
        const y = this.target[1] + this.distance * Math.cos(this.phi);
        const z = this.target[2] + this.distance * Math.sin(this.phi) * Math.cos(this.theta);
        return [x, y, z];
    }

    // カメラのビュー行列 (どこからどこを見ているか)
    private getViewMatrix(): mat4 {
        const eye = this.getEyePosition();
        const view = mat4.create();
        mat4.lookAt(view, eye, this.target, [0, 1, 0]); // 上方向は常にY軸
        return view;
    }

    // 毎フレーム呼ばれ、GPUに渡すための2つの行列を生成して返す
    public getMatrices(aspectRatio: number) {
        const view = this.getViewMatrix();
        
        const proj = mat4.create();
        // 画角: 45度 (PI/4), Nearクリップ: 0.1, Farクリップ: 1000.0
        mat4.perspective(proj, Math.PI / 4, aspectRatio, 0.1, 1000.0);

        const viewProj = mat4.create();
        mat4.multiply(viewProj, proj, view); // プロジェクション行列 × ビュー行列

        return {
            view: Array.from(view),
            viewProjection: Array.from(viewProj)
        };
    }
}