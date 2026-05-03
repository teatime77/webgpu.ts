// ============================================================================
// 1. 型定義 (Types & Interfaces)
// ============================================================================

import { assert, fetchText, msg, MyError } from "@i18n";
import { Context, Parser } from "./parser";
import { lexicalAnalysis, TokenType } from "./lex";
import { App, BlockStatement, CallStatement, RefVar, setParentSub, Statement, WhileStatement, YieldStatement } from "./syntax";

export type WgslFormat = 'f32' | 'u32' | 'i32' | 'vec2<f32>' | 'vec3<f32>' | 'vec4<f32>' | 'mat4x4<f32>' | 'atomic<u32>' | 'atomic<i32>';
export type GenType    = NodeDef | CallStatement | YieldStatement;

export interface MetaData {
    [key: string]: number | string | boolean | number[]; 
}

export interface ResourceDef {
    type: 'storage' | 'uniform' | 'indirect' | 'texture' | 'sampler';
    access?: 'read' | 'read_write';
    format?: WgslFormat;
    count?: string | number;
    pingPong?: boolean;
    fields?: Record<string, string>;
}

export interface BindingDefinition {
    group?: number;                 // デフォルトは 0
    binding: number;
    resource: string;               // 参照するリソースID
    state?: 'current' | 'previous'; // Ping-Pongバッファの場合の指定
    varName?: string;               // 🌟 追加: WGSL上で使用する明示的な変数名
    access?: 'read' | 'read_write';
}

export interface BaseNodeDef {
    id : string,
    type: 'compute' | 'render' | 'reduction' | 'readback';
    shader: string;
    bindings: BindingDefinition[];
}

export interface ComputeNodeDef extends BaseNodeDef {
    type: 'compute';
    dispatch: {
        type: 'direct' | 'indirect';
        // [X, Y, Z] の各次元でも "$metadata.maxParticles / 64" のような文字列を許容
        workgroups: [number | string, number | string, number | string] | string;
    };
}

export interface RenderNodeDef extends BaseNodeDef {
    type: 'render';
    topology: 'point-list' | 'line-list' | 'triangle-list';
    depthTest?: boolean; // 追加: 深度テストの有無
    draw: {              // 追加: 描画コマンドの定義
        type: 'direct' | 'indirect';
        vertexCount: string | number;
        instanceCount?: string | number;
    };
}

export type NodeDef = ComputeNodeDef | RenderNodeDef;

export interface StateDef {
    execute: string[]; // 実行するノードIDのリスト
    onExit?: { action: string; resources: string[] }[]; // 例: Ping-Pongのスワップ
    transitions: { condition: string; target: string }[];
}

export interface SimulationSchema {
    name: string;
    version: string;
    metadata: MetaData;
    resources: Record<string, ResourceDef>;
    nodes: NodeDef[];
}

// 型定義の追加
interface UniformLayoutInfo {
    totalByteSize: number;
    offsets: Record<string, { byteOffset: number; format: string }>;
}

function getNodeById(nodes: NodeDef[], id : string) : NodeDef {
    const node = nodes.find(x => x.id == id)!;
    assert(node != undefined);

    return node;
}

// ============================================================================
// 2. リソース管理ラッパー (ResourceWrapper)
// ============================================================================

export class ResourceWrapper {
    public id: string;
    public buffers: GPUBuffer[];
    public isPingPong: boolean;
    private currentIndex: number = 0;

    constructor(id: string, buffers: GPUBuffer[], isPingPong: boolean) {
        this.id = id;
        this.buffers = buffers;
        this.isPingPong = isPingPong;
    }

    /** 最新状態(書き込み先または最新の読み取り元)のバッファを取得 */
    public getCurrentBuffer(): GPUBuffer {
        return this.buffers[this.currentIndex];
    }

    /** 1つ前の状態(読み取り専用)のバッファを取得 */
    public getPreviousBuffer(): GPUBuffer {
        if (!this.isPingPong) return this.buffers[0];
        return this.buffers[1 - this.currentIndex];
    }

    /** フレーム/ステップ終了時にA/Bを反転させる */
    public swap(): void {
        if (this.isPingPong) {
            this.currentIndex = 1 - this.currentIndex;
        }
    }
}

// ============================================================================
// 3. WGSL ヘッダー自動生成ジェネレータ (WgslHeaderGenerator)
// ============================================================================

export class WgslHeaderGenerator {
    
    /** 特定のノード用のWGSL宣言部を自動生成する */
    public static generateForNode(schema: SimulationSchema, nodeId: string): string {
        const node = getNodeById(schema.nodes, nodeId);
        if (!node) throw new Error(`Node ${nodeId} not found.`);

        let header = `// ==========================================\n`;
        header += `// AUTO-GENERATED HEADER FOR NODE: ${nodeId}\n`;
        header += `// ==========================================\n\n`;

        header += this.generateStructs(schema, node.bindings);
        header += this.generateBindings(schema, node.bindings);

        header += `\n// --- YOUR COMPUTE LOGIC BELOW ---\n\n`;
        msg(`header:\n${header}`);
        
        return header;
    }

    private static generateStructs(schema: SimulationSchema, bindings: BindingDefinition[]): string {
        let code = '';
        const generatedStructs = new Set<string>();

        for (const bind of bindings) {
            const resource = schema.resources[bind.resource];
            if (resource.type === 'uniform' && resource.fields) {
                if (generatedStructs.has(bind.resource)) continue;
                generatedStructs.add(bind.resource);

                code += `struct ${bind.resource} {\n`;
                for (const [fieldName, fieldType] of Object.entries(resource.fields)) {
                    code += `    ${fieldName}: ${fieldType},\n`;
                }
                code += `};\n\n`;
            }
        }
        return code;
    }

    private static generateBindings(schema: SimulationSchema, bindings: BindingDefinition[]): string {
        let code = '';
        
        for (const bind of bindings) {
            const resource = schema.resources[bind.resource];
            const group = bind.group || 0;
            const bindingNum = bind.binding;
            
            // Ping-Pongの過去バッファを参照する場合は変数名に _prev を付ける
            let varName = bind.varName || bind.resource;

            if (resource.type === 'uniform') {
                code += `@group(${group}) @binding(${bindingNum}) var<uniform> ${varName}: ${bind.resource};\n`;
            } 
            else if (resource.type === 'storage') {
                // stateが'previous'の場合は強制的にread-onlyにする安全対策
                let access = bind.access;

                const isAtomic = resource.format && resource.format.includes('atomic');

                if (isAtomic) {
                    // WGSLの仕様上、atomicを含むバッファは必ず read_write
                    access = 'read_write';
                } else if (!access) {
                    // デフォルトのフォールバック (previousは安全のため強制read)
                    access = (resource.access === 'read_write' && bind.state !== 'previous') ? 'read_write' : 'read';
                }
                code += `@group(${group}) @binding(${bindingNum}) var<storage, ${access}> ${varName}: array<${resource.format}>;\n`;
            }
            else if (resource.type === 'texture') {
                code += `@group(${group}) @binding(${bindingNum}) var ${varName}: texture_2d<f32>;\n`;
            }
            else if (resource.type === 'sampler') {
                code += `@group(${group}) @binding(${bindingNum}) var ${varName}: sampler;\n`;
            }
        }
        return code;
    }
}

// ============================================================================
// 4. グラフマネージャ本体 (GraphManager)
// ============================================================================

export class GraphManager {
    private device: GPUDevice;
    private schemaName : string;
    public schema!: SimulationSchema;
    private script : BlockStatement | undefined;
    private scriptGen : Generator<GenType> | undefined;
    public resources: Map<string, ResourceWrapper> = new Map();

    // Uniformバッファのレイアウト設計図をキャッシュする
    private uniformLayouts: Map<string, UniformLayoutInfo> = new Map();
    
    // --- パフォーマンス最適化のためのキャッシュ ---
    private computePipelines: Map<string, GPUComputePipeline> = new Map();
    // バインドグループはPing-Pongの状態によって変わるため、毎フレーム動的生成するか、
    // 「NodeID + 現在のPingPongインデックス」をキーにしてキャッシュします。
    private bindGroupCache: Map<string, GPUBindGroup> = new Map();

    // --- レンダリング用プロパティ ---
    private canvasContext: GPUCanvasContext | null = null;
    private presentationFormat: GPUTextureFormat = 'bgra8unorm'; // 初期値
    private depthTextureView: GPUTextureView | null = null;
    
    private renderPipelines: Map<string, GPURenderPipeline> = new Map();

    // --- リードバック用の管理機構 ---
    private stagingBuffers: Map<string, GPUBuffer> = new Map();
    private pendingReadbacks: Array<{
        resourceId: string;
        byteSize?: number; // バッファ全体ではなく一部だけ読みたい場合
        resolve: (data: Float32Array) => void; 
        reject: (reason?: any) => void;
    }> = [];

    private externalResources: Map<string, GPUBindingResource> = new Map();

    constructor(device: GPUDevice, schemaName : string) {
        this.device = device;
        this.schemaName = schemaName;
    }

    async parseSchemaScript(){
        const jsText = await fetchText(`./wgsl/${this.schemaName}/${this.schemaName}.js`);
        const tokens = lexicalAnalysis(jsText);
        const parser = new Parser(tokens, 0);

        const ctx = Context.unknown;
        const statements : Statement[] = [];
        while (!parser.isEoT()) {
            const statement = parser.parseStatement(ctx);
            statements.push(statement);
        }

        this.script = new BlockStatement(statements);
        setParentSub(this.script, this.script.statements);
        this.scriptGen = this.exec(this.script);
    }

    *exec(stmt : Statement) : Generator<GenType> {
        if(stmt instanceof CallStatement){
            const app = stmt.app;
            const fncName = app.fncName;
            if(fncName == "swapPingPong"){

                assert(app.args.every(x => x instanceof RefVar));
                const varNames = app.args.map(x => (x as RefVar).name);
                // msg(`ping-pong:${varNames.join(", ")}`);
                yield stmt;
            }
            else{

                const node = getNodeById(this.schema.nodes, fncName);
                if(node == undefined){
                    throw new MyError();
                }

                // msg(`gen:call ${fncName} node:${node}`);
                yield node;
            }
        }
        else if(stmt instanceof YieldStatement){
            // msg("gen:yield");
            yield stmt;
        }
        else if(stmt instanceof BlockStatement){
            // msg("gen:block");
            for(const child of stmt.statements){
                yield* this.exec(child);
            }
        }
        else if(stmt instanceof WhileStatement){
            // msg("gen:while");
            assert(stmt.condition instanceof RefVar && stmt.condition.name == "true");
            while(true){
                yield* this.exec(stmt.block);
            }
        }
        else{
            throw new MyError();
        }
    }

    async initGraphManager(){
        await this.parseSchemaScript();
    }

    public setExternalResource(id: string, resource: GPUBindingResource) {
        this.externalResources.set(id, resource);
    }

    /** JSONスキーマをロードし、GPUリソースを確保する */
    public loadSchema(schema: SimulationSchema) {
        console.log(`Loading Simulation: ${schema.name} (v${schema.version})`);
        this.schema = schema;
        this.buildUniformLayouts(); // アロケーションの前にレイアウトを計算
        this.allocateResources(schema.resources);
    }

    /** リソース定義からGPUBufferを動的に確保する */
    private allocateResources(resourceDefs: Record<string, ResourceDef>) {
        for (const [id, def] of Object.entries(resourceDefs)) {
            const byteSize = this.calculateByteSize(id, def);
            const usage = this.resolveBufferUsage(def);

            const buffers: GPUBuffer[] = [];
            const bufferCount = def.pingPong ? 2 : 1;

            for (let i = 0; i < bufferCount; i++) {
                buffers.push(this.device.createBuffer({
                    label: `${id}_buffer_${i}`,
                    size: byteSize,
                    usage: usage,
                }));
            }

            this.resources.set(id, new ResourceWrapper(id, buffers, !!def.pingPong));
            console.log(`Allocated resource [${id}]: ${byteSize} bytes ${def.pingPong ? '(Ping-Pong)' : ''}`);
        }
    }

    // --- 計算式・変数展開ヘルパー ---

    /** 文字列内の数式(例: "$metadata.maxParticles / 64")を評価して数値を返す */
    public evaluateExpression(expression: string | number): number {
        if (typeof expression === 'number') return expression;

        // 変数展開 ($metadata.xxx を実数値に置換)
        let parsedExpr = expression.replace(/\$metadata\.([a-zA-Z0-9_]+)/g, (match, key) => {
            const val = this.schema.metadata[key];
            if (typeof val !== 'number') throw new Error(`Metadata ${key} is not a number`);
            return val.toString();
        });

        // 簡易的な数式評価 (安全なevalの代替。実際はFunctionコンストラクタか専用パーサを推奨)
        try {
            return new Function(`return ${parsedExpr}`)();
        } catch (e) {
            throw new Error(`Failed to evaluate expression: ${expression}`);
        }
    }

    private calculateByteSize(id: string, def: ResourceDef): number {
        if (def.type === 'uniform') {
            return this.uniformLayouts.get(id)!.totalByteSize;
        }

        const elementSize = this.getFormatByteSize(def.format || 'f32');
        const count = this.evaluateExpression(def.count || 1);
        return elementSize * count;
    }

    private getFormatByteSize(format: WgslFormat): number {
        switch (format) {
            case 'f32': case 'u32': case 'i32': 
            case 'atomic<u32>': case 'atomic<i32>':
                return 4;
            case 'vec2<f32>': return 8;
            case 'vec3<f32>': return 16; 
            case 'vec4<f32>': return 16;
            case 'mat4x4<f32>': return 64;
            default: return 4;
        }
    }

    private resolveBufferUsage(def: ResourceDef): number {
        let usage = GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        if (def.type === 'storage') usage |= GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX;
        if (def.type === 'uniform') usage |= GPUBufferUsage.UNIFORM;
        if (def.type === 'indirect') usage |= GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE;
        return usage;
    }

    /** 指定したリソースのA/Bスワップを実行 (StateのonExitなどで呼ばれる) */
    public swapPingPongResources(resourceIds: string[]) {
        for (const id of resourceIds) {
            const res = this.resources.get(id);
            if (res && res.isPingPong) {
                res.swap();
            }
        }
    }

    // ========================================================================
    // 1. 初期化フェーズ (Pre-computation)
    // ========================================================================

    /** 
     * 実行前にすべてのシェーダをコンパイルし、パイプラインを準備する。
     * 実行ループ中に createPipeline するとカクつくため、必ず非同期で行う。
     */
    public async compilePipelines(wgslCodes: Record<string, string>) {
        console.log("Compiling pipelines...");

        for (const nodeDef of this.schema.nodes) {
            const nodeId = nodeDef.id;
            // --- 共通: シェーダモジュールの作成 ---
            const header = WgslHeaderGenerator.generateForNode(this.schema, nodeId);
            const fullCode = header + (wgslCodes[nodeDef.shader] || '');
            const module = this.device.createShaderModule({ 
                label: `ShaderModule_${nodeId}`,
                code: fullCode 
            });

            // --- Compute パイプラインの作成 ---
            if (nodeDef.type === 'compute') {
                const pipeline = await this.device.createComputePipelineAsync({
                    label: `ComputePipeline_${nodeId}`,
                    layout: 'auto',
                    compute: { module, entryPoint: 'main' }
                });
                this.computePipelines.set(nodeId, pipeline);
            } 
            // --- Render パイプラインの作成 ---
            else if (nodeDef.type === 'render') {
                const depthStencil: GPUDepthStencilState | undefined = nodeDef.depthTest ? {
                    depthWriteEnabled: true,
                    depthCompare: 'less',
                    format: 'depth24plus',
                } : undefined;

                const pipeline = await this.device.createRenderPipelineAsync({
                    label: `RenderPipeline_${nodeId}`,
                    layout: 'auto',
                    vertex: { module, entryPoint: 'vs_main', buffers: [] },
                    fragment: {
                        module, entryPoint: 'fs_main',
                        targets: [{ 
                            format: this.presentationFormat,
                            blend: {
                                color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                            }
                        }]
                    },
                    primitive: { topology: nodeDef.topology as GPUPrimitiveTopology, cullMode: 'none' },
                    depthStencil
                });
                this.renderPipelines.set(nodeId, pipeline);
            }
        }
        console.log("All pipelines compiled successfully.");
    }

    execNode(encoder: GPUCommandEncoder, renderPass: GPURenderPassEncoder | null, nodeId : string, nodeDef : NodeDef) : GPURenderPassEncoder | null{            
        if (nodeDef.type === 'compute') {
            if (renderPass) {
                renderPass.end();
                renderPass = null;
            }
            this.encodeComputeNode(encoder, nodeId, nodeDef as ComputeNodeDef);
        } 
        else if (nodeDef.type === 'render') {
            const renderNode = nodeDef as RenderNodeDef; // 型を確定させる

            // レンダーパスの遅延初期化
            if (!renderPass && this.canvasContext) {
                const textureView = this.canvasContext.getCurrentTexture().createView();
                
                // 🚨 修正: 現在のノードが depthTest を要求しているかチェック
                const useDepth = renderNode.depthTest === true && this.depthTextureView !== null;

                renderPass = encoder.beginRenderPass({
                    label: `RenderPass_${nodeId}`,
                    colorAttachments: [{
                        view: textureView,
                        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                    // 🚨 修正: 要求されている場合のみデプスバッファをアタッチする
                    depthStencilAttachment: useDepth ? {
                        view: this.depthTextureView!,
                        depthClearValue: 1.0,
                        depthLoadOp: 'clear',
                        depthStoreOp: 'store',
                    } : undefined
                });
            }
            
            if (renderPass) {
                this.encodeRenderNode(renderPass, nodeId, renderNode);
            }
        }

        return renderPass;
    }

    // ========================================================================
    // 2. 実行フェーズ (Execution Loop)
    // ========================================================================
    public step() {
        const encoder = this.device.createCommandEncoder();
        let renderPass: GPURenderPassEncoder | null = null;

        let result: IteratorResult<GenType, any> | undefined;
        if(this.scriptGen != undefined){
            while(true){
                result = this.scriptGen.next();
                if(result.done){
                    this.scriptGen = undefined;
                    msg("generator completed.");
                    break;
                }
                if(result.value instanceof CallStatement || result.value instanceof YieldStatement){
                    break;
                }
                else{

                    const nodeDef = result.value;
                    renderPass = this.execNode(encoder, renderPass, nodeDef.id, nodeDef);
                }
            }
        }

        if (renderPass) {
            renderPass.end();
        }

        // 2. リードバックのコピー・コマンドを積む
        const activeReadbacks = [...this.pendingReadbacks]; // 現在のキューを退避
        this.pendingReadbacks = []; // キューをクリア

        for (const rb of activeReadbacks) {
            this.encodeReadbackCopy(encoder, rb.resourceId, rb.byteSize);
        }

        // 3. GPUへコマンドを送信
        this.device.queue.submit([encoder.finish()]);

        // 4. 送信後に非同期でメモリをマッピングしてPromiseを解決する
        for (const rb of activeReadbacks) {
            this.processAsyncMapping(rb);
        }

        if(result != undefined){
            if(result.value instanceof CallStatement){
                const app = result.value.app;
                assert(app.args.every(x => x instanceof RefVar));
                const varNames = app.args.map(x => (x as RefVar).name);
                this.swapPingPongResources(varNames);
            }
            else{
                throw new MyError();
            }

            result = this.scriptGen!.next();
            assert(result.value instanceof YieldStatement);
        }
    }

    // ========================================================================
    // 3. コマンドエンコーディングとバインディング
    // ========================================================================

    private encodeComputeNode(encoder: GPUCommandEncoder, nodeId: string, nodeDef: ComputeNodeDef) {
        const pipeline = this.computePipelines.get(nodeId);
        if (!pipeline) throw new Error(`Pipeline for ${nodeId} not found.`);

        const pass = encoder.beginComputePass({ label: `Pass_${nodeId}` });
        pass.setPipeline(pipeline);

        // --- BindGroup の構築 ---
        const bindGroup = this.buildBindGroup(nodeId, nodeDef, pipeline);
        pass.setBindGroup(0, bindGroup);

        // --- ディスパッチサイズの解決 ---
        if (nodeDef.dispatch.type === 'direct') {
            const workgroups = nodeDef.dispatch.workgroups as (string | number)[];
            const x = this.evaluateExpression(workgroups[0]);
            const y = this.evaluateExpression(workgroups[1] || 1);
            const z = this.evaluateExpression(workgroups[2] || 1);
            pass.dispatchWorkgroups(x, y, z);
        } else if (nodeDef.dispatch.type === 'indirect') {
            // (オプション) 間接ディスパッチの実装
        }

        pass.end();
    }

    /** 
     * ノードの定義から GPUBindGroup を組み立てる。
     * コンピュートパイプラインとレンダーパイプラインの両方に対応。
     */
    private buildBindGroup(
        nodeId: string, 
        nodeDef: BaseNodeDef, // ComputeNodeDefとRenderNodeDefの共通親クラスに変更
        pipeline: GPUComputePipeline | GPURenderPipeline // 両方のパイプラインを許容
    ): GPUBindGroup {
        let cacheKey = `${nodeId}_`;
        
        const entries: GPUBindGroupEntry[] = nodeDef.bindings.map(bind => {
            let gpuResource: GPUBindingResource;
            if (this.externalResources.has(bind.resource)) {
                gpuResource = this.externalResources.get(bind.resource)!;
            } else {
                const wrapper = this.resources.get(bind.resource)!;
                
                // 🌟 追加: Ping-Pongの「今どっちの面か(0 or 1)」をキャッシュキーに足す！
                // (anyキャストを使ってprivateプロパティを強引に読みます)
                cacheKey += `${wrapper.isPingPong ? (wrapper as any).currentIndex : 0}_`;
                
                gpuResource = { buffer: bind.state === 'previous' ? wrapper.getPreviousBuffer() : wrapper.getCurrentBuffer() };
            }

            return { binding: bind.binding, resource: gpuResource };
        });

        if (this.bindGroupCache.has(cacheKey)) {
            return this.bindGroupCache.get(cacheKey)!;
        }

        const bindGroup = this.device.createBindGroup({
            label: `BindGroup_${nodeId}`,
            // getBindGroupLayoutは両方のパイプラインに共通して存在するメソッド
            layout: pipeline.getBindGroupLayout(0), 
            entries: entries
        });

        this.bindGroupCache.set(cacheKey, bindGroup);
        return bindGroup;
    }

    // ========================================================================
    // 1. ロード時のレイアウト計算
    // ========================================================================

    /** 
     * Uniformバッファのフィールド定義から、WebGPUのアライメント規則(std140)に
     * 従ったバイトオフセットを計算する
     */
    private buildUniformLayouts() {
        for (const [id, def] of Object.entries(this.schema.resources)) {
            if (def.type === 'uniform' && def.fields) {
                let currentOffset = 0;
                const offsets: Record<string, { byteOffset: number; format: string }> = {};

                for (const [fieldName, format] of Object.entries(def.fields)) {
                    // アライメントの計算 (簡略化版。実際はvec3などに注意が必要)
                    let alignment = 4;
                    let size = 4;
                    if (format === 'vec2<f32>') { alignment = 8; size = 8; }
                    if (format === 'vec3<f32>' || format === 'vec4<f32>') { alignment = 16; size = 16; }
                    else if (format === 'mat4x4<f32>') { alignment = 16; size = 64; }

                    // 現在のオフセットをアライメントの倍数に切り上げる (パディングの挿入)
                    currentOffset = Math.ceil(currentOffset / alignment) * alignment;

                    offsets[fieldName] = { byteOffset: currentOffset, format };
                    currentOffset += size;
                }

                // 構造体全体のサイズは最大の16バイトアライメントに切り上げる
                const totalByteSize = Math.ceil(currentOffset / 16) * 16;
                this.uniformLayouts.set(id, { totalByteSize, offsets });
            }
        }
    }

    // ========================================================================
    // 2. 毎フレームの動的更新処理
    // ========================================================================

    /**
     * UIなどから受け取った最新の変数をJSONの metadata にマージし、
     * GPUの Uniform バッファへバイナリデータとして転送する。
     */
    public updateVariables(newVariables: Record<string, number | number[]>) {
        // 1. メタデータを更新
        this.schema.metadata = { ...this.schema.metadata, ...newVariables };

        // 2. 各Uniformバッファに書き込む
        for (const [id, layout] of this.uniformLayouts.entries()) {
            const resource = this.resources.get(id);
            if (!resource) continue;

            // 送信用のバイナリバッファ(ArrayBuffer)を用意
            const arrayBuffer = new ArrayBuffer(layout.totalByteSize);
            const dataView = new DataView(arrayBuffer);

            // レイアウト設計図に従って、メタデータの値をバイナリに書き込む
            for (const [fieldName, info] of Object.entries(layout.offsets)) {
                const value = this.schema.metadata[fieldName];
                
                // 値が存在しなければスキップ (初期値のまま)
                if (value === undefined) continue;

                if (info.format === 'f32' && typeof value === 'number') {
                    dataView.setFloat32(info.byteOffset, value, true); // true = Little Endian
                } 
                else if (info.format === 'i32' && typeof value === 'number') {
                    dataView.setInt32(info.byteOffset, value, true);
                }
                else if ((info.format.startsWith('vec') || info.format.startsWith('mat')) && Array.isArray(value)) {
                    // vec2, vec3, vec4, mat4x4 などの配列書き込み
                    for (let i = 0; i < value.length; i++) {
                        dataView.setFloat32(info.byteOffset + i * 4, value[i], true);
                    }
                }
            }

            // WebGPUのキューに書き込み命令を積む (エンコーダーを介さず直接送信できるので超高速)
            this.device.queue.writeBuffer(
                resource.getCurrentBuffer(), 
                0, 
                arrayBuffer
            );
        }
    }


    // ========================================================================
    // 1. Canvasの初期化
    // ========================================================================

    /** WebGPUのCanvasContextをエンジンに紐付ける */
    public setContext(context: GPUCanvasContext, format: GPUTextureFormat, width: number, height: number) {
        this.canvasContext = context;
        this.presentationFormat = format;
        this.resizeDepthBuffer(width, height);
    }

    private resizeDepthBuffer(width: number, height: number) {
        const depthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTextureView = depthTexture.createView();
    }

    // ========================================================================
    // 2. パイプラインのコンパイル (拡張)
    // ========================================================================


    // ========================================================================
    // 3. 実行ループ内の描画エンコーディング
    // ========================================================================


    private encodeRenderNode(pass: GPURenderPassEncoder, nodeId: string, nodeDef: RenderNodeDef) {
        const pipeline = this.renderPipelines.get(nodeId);
        if (!pipeline) return;

        pass.setPipeline(pipeline);

        // BindGroupの構築 (コンピュートと全く同じロジックを再利用！)
        const bindGroup = this.buildBindGroup(nodeId, nodeDef, pipeline);
        pass.setBindGroup(0, bindGroup);

        // 描画コマンドの発行
        if (nodeDef.draw.type === 'direct') {
            const vertexCount = this.evaluateExpression(nodeDef.draw.vertexCount);
            const instanceCount = this.evaluateExpression(nodeDef.draw.instanceCount || 1);
            pass.draw(vertexCount, instanceCount, 0, 0);
        } else if (nodeDef.draw.type === 'indirect') {
             // 間接描画 (GPU-Driven Rendering) の実装
             // pass.drawIndirect(...)
        }
    }

    // ========================================================================
    // 1. TS側からのリードバック要求 (Promiseを返す)
    // ========================================================================

    /**
     * GPUバッファの内容を非同期で読み出す要求をキューに積む。
     * 実際の読み出しは次の step() 実行時に一括で処理される。
     */
    public requestReadback(resourceId: string, byteSize?: number): Promise<Float32Array> {
        return new Promise((resolve, reject) => {
            this.pendingReadbacks.push({ resourceId, byteSize, resolve, reject });
        });
    }

    // ========================================================================
    // 3. 裏側のバッファ管理ロジック
    // ========================================================================

    private encodeReadbackCopy(encoder: GPUCommandEncoder, resourceId: string, overrideByteSize?: number) {
        const wrapper = this.resources.get(resourceId);
        if (!wrapper) throw new Error(`Resource ${resourceId} not found for readback.`);

        const srcBuffer = wrapper.getCurrentBuffer();
        const copySize = overrideByteSize || srcBuffer.size;

        // Staging Buffer が無ければ作成 (Lazy Creation)
        let stagingBuffer = this.stagingBuffers.get(resourceId);
        if (!stagingBuffer || stagingBuffer.size < copySize) {
            stagingBuffer = this.device.createBuffer({
                label: `StagingBuffer_${resourceId}`,
                size: copySize,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            this.stagingBuffers.set(resourceId, stagingBuffer);
        }

        // Storage から Staging へコピー
        encoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, copySize);
    }

    private async processAsyncMapping(rb: { resourceId: string, byteSize?: number, resolve: Function, reject: Function }) {
        const stagingBuffer = this.stagingBuffers.get(rb.resourceId);
        if (!stagingBuffer) return rb.reject("Staging buffer missing");

        try {
            // GPUの処理完了を待ってメモリをマップする
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            
            // マップされたメモリを参照する (slice(0) でデータをクローンして安全に返す)
            const mappedRange = stagingBuffer.getMappedRange(0, rb.byteSize);
            const dataCopy = new Float32Array(mappedRange.slice(0)); 
            
            // unmap して次回のコピーに備える
            stagingBuffer.unmap();
            
            // TS側の await を解決！
            rb.resolve(dataCopy);

        } catch (err) {
            rb.reject(err);
        }
    }
}