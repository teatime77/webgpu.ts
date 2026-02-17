declare var glMatrix: any;

import { Vec3 } from "@i18n";
import { AbstractPipeline, ComputePipeline } from "./compute.js";
import { ShapeInfo } from "./package.js";
import { Module, Struct } from "./parser.js";
import { mat4fromMat3, ui3D } from "./ui.js";
import { g_device, fetchModule, g_presentationFormat, MyError, range } from "./util.js";

function vecLen(p: Vec3) {
    return Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

function vecDiff(p: Vec3, q: Vec3) {
    var dx = p.x - q.x;
    var dy = p.y - q.y;
    var dz = p.z - q.z;

    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
    
function vecSub(a: Vec3, b: Vec3) {
    return new Vec3( a.x - b.x, a.y - b.y, a.z - b.z );
}
        
function vecDot(a: Vec3, b: Vec3) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

function vecCross(a: Vec3, b: Vec3) {
    return new Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

export class Vertex extends Vec3 {
    nx: number = 0;
    ny: number = 0;
    nz: number = 0;
    texX: number = 0;
    texY: number = 0;

    adjacentVertexes: Vertex[];

    constructor(x: number, y: number, z: number) {
        super(x, y, z);
        this.adjacentVertexes = [];
    }
}


class Edge {
    Endpoints: Vertex[];

    constructor(p1: Vertex, p2: Vertex) {
        this.Endpoints = [p1, p2];
    }
}


export class Color {
    r: number;
    g: number;
    b: number;
    a: number;

    constructor(r:number, g: number, b: number, a: number){
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = a;
    }

    static get red(): Color {
        return new Color(1, 0, 0, 1);
    }

    static get green(): Color {
        return new Color(0, 1, 0, 1);
    }

    static get blue(): Color {
        return new Color(0, 0, 1, 1);
    }
}
    
class Triangle {
    Vertexes: Vertex[];

    constructor(p: Vertex, q: Vertex, r: Vertex, orderd: boolean = false) {
        if (orderd == true) {

            this.Vertexes = [p, q, r];
        }
        else {

            var a = vecSub(q, p);
            var b = vecSub(r, q);

            var c = vecCross(a, b);
            var dir = vecDot(p, c);
            if (0 < dir) {
                this.Vertexes = [p, q, r];
            }
            else {
                this.Vertexes = [q, p, r];
            }
        }
    }
}


export class RenderPipeline extends AbstractPipeline {
    vertName : string;
    fragName : string

    vertexCount!: number;
    vertexArray!: Float32Array;

    topology!: GPUPrimitiveTopology;

    materialColor = new Float32Array([1.0, 1.0, 1.0, 1.0]);
    shapeInfo     = new Float32Array([0,0,0,0]);

    compute  : ComputePipeline | null = null;

    pipeline!         : GPURenderPipeline;
    vertModule!       : Module;
    fragModule!       : Module;
    uniformBindGroup! : GPUBindGroup;
    vertexBuffer!     : GPUBuffer;

    constructor(shape : ShapeInfo){
        super();

        if(shape.vertName != undefined){

            this.vertName = shape.vertName;
        }
        else{
            this.vertName = "shape-vert";
        }
        this.fragName = "depth-frag";
    }

    red() : RenderPipeline {
        this.materialColor = new Float32Array([1.0, 0.0, 0.0, 1.0]);
        return this;
    }

    green() : RenderPipeline {
        this.materialColor = new Float32Array([0.0, 1.0, 0.0, 1.0]);
        return this;
    }

    blue() : RenderPipeline {
        this.materialColor = new Float32Array([0.0, 0.0, 1.0, 1.0]);
        return this;
    }

    getWorldMatrix() : Float32Array {
        return glMatrix.mat4.create();
    }

    makeUniformBufferAndBindGroup(){
        // @uniform
        const uniform_size  = this.vertModule.uniformSize();
        const uniform_buffer_size = Math.ceil(uniform_size / 32) * 32;

        this.makeUniformBuffer(uniform_buffer_size);

        this.uniformBindGroup = g_device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
            ],
        });
    }

    makeVertexBuffer(){

        // Create a vertex buffer from the quad data.
        this.vertexBuffer = g_device.createBuffer({
            size: this.vertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.vertexArray);
        this.vertexBuffer.unmap();
    }

    async makeRenderPipeline(){
        this.vertModule = await fetchModule(this.vertName);
        this.fragModule = await fetchModule(this.fragName);
    
        const vertex_buffer_layouts = this.vertModule.makeVertexBufferLayouts(this.compute);
    
        const pipeline_descriptor : GPURenderPipelineDescriptor = {
            layout: 'auto',
            vertex: {
                module: this.vertModule.module,
                entryPoint: 'main',
                buffers: vertex_buffer_layouts,
            },
            fragment: {
                module: this.fragModule.module,
                entryPoint: 'main',
                targets: [
                    // 0
                    { // @location(0) in fragment shader
                        format: g_presentationFormat,
                    },
                ],
            },
            primitive: {
                topology: this.topology,
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        };
    
        this.pipeline = g_device.createRenderPipeline(pipeline_descriptor);
    }

    writeUniform(){
        let worldMatrix  = this.getWorldMatrix();
        let normalMatrix = glMatrix.mat3.create();

        glMatrix.mat3.normalFromMat4(normalMatrix, worldMatrix);
        normalMatrix = mat4fromMat3(normalMatrix);

        let pvw = glMatrix.mat4.create();
        glMatrix.mat4.mul(pvw, ui3D.ProjViewMatrix, worldMatrix);

        let offset = 0;

        // vec4 align is 16
        // https://www.w3.org/TR/WGSL/#alignment-and-size
        // offset += 12;

        const uniform_var_struct = this.vertModule.getUniformVar().type as Struct;
        for(const member of uniform_var_struct.members){
            switch(member.name){
            case "viewMatrix":
                offset = this.writeUniformBuffer(pvw, offset);
                break;
            case "normMatrix":
                offset = this.writeUniformBuffer(normalMatrix, offset);
                break;
            case "materialColor":
                offset = this.writeUniformBuffer(this.materialColor, offset);
                break;
            case "ambientColor":
                offset = this.writeUniformBuffer(ui3D.ambientColor     , offset);
                break;
            case "directionalColor":
                offset = this.writeUniformBuffer(ui3D.directionalColor , offset);
                break;
            case "lightingDirection":
                offset = this.writeUniformBuffer(ui3D.lightingDirection, offset);
                break;
            case "env":
                offset = this.writeUniformBuffer(ui3D.env              , offset);
                break;
            case "shapeInfo":
                offset = this.writeUniformBuffer(this.shapeInfo        , offset);
                break;
            default:
                throw new MyError(`unknown uniform:${member.name}`);
            }
        }
    }
    
    render(tick : number, passEncoder : GPURenderPassEncoder){
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(this.vertModule.vertexSlot, this.vertexBuffer);
        if(this.compute != null){

            passEncoder.setVertexBuffer(this.vertModule.instanceSlot, this.compute.updateBuffers[(tick + 1) % 2]);
            passEncoder.draw(this.vertexCount, this.compute.instanceCount);
        }
        else{

            passEncoder.draw(this.vertexCount);
        }
    }
}


export class Point extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);
        this.vertName    = "point-vert";
        this.vertexCount = 1;
        this.vertexArray = new Float32Array(1 * (3 + 3));

        this.topology = 'point-list';
    }
}


export class Line extends RenderPipeline {
    constructor(shape : ShapeInfo, vertex_array : Float32Array | null = null){
        super(shape);
        if(vertex_array == null){

            this.vertName    = "line-vert";
            this.vertexCount = 2;
            this.vertexArray = new Float32Array(2 * (3 + 3));
        }
        else{

            this.vertName = "line-vert-fix";
        
            // 位置の配列
            this.vertexArray = vertex_array;
            this.vertexCount = vertex_array.length / (3 + 3);
            console.assert(vertex_array.length % (3 + 3) == 0)
        }
    
        this.topology = 'line-list';
    }
}

export class Tube extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);
        const num_division = 16;
        
        this.vertexCount = (num_division + 1) * 2;
    
        // 位置の配列
        this.vertexArray = new Float32Array(this.vertexCount * (3 + 3));

        let base = 0;
        for(let idx of range(num_division + 1)){
            let theta = 2 * Math.PI * idx / num_division;
            let x = Math.cos(theta);
            let y = Math.sin(theta);

            for(const z of [0, 1]){

                setPosNorm(this.vertexArray, base, x, y, z, x, y, 0);
                base++;
            }
        }
    
        this.topology = 'triangle-strip';
    }
}

export class Cube extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);

        // position: vec3<f32>, norm: vec3<f32>
        // prettier-ignore
        this.vertexArray = new Float32Array([
             1, -1,  1,   0, -1,  0,
            -1, -1,  1,   0, -1,  0,
            -1, -1, -1,   0, -1,  0,
             1, -1, -1,   0, -1,  0,
             1, -1,  1,   0, -1,  0,
            -1, -1, -1,   0, -1,  0,

             1,  1,  1,   1,  0,  0,
             1, -1,  1,   1,  0,  0,
             1, -1, -1,   1,  0,  0,
             1,  1, -1,   1,  0,  0,
             1,  1,  1,   1,  0,  0,
             1, -1, -1,   1,  0,  0,

            -1,  1,  1,   0,  1,  0,
             1,  1,  1,   0,  1,  0,
             1,  1, -1,   0,  1,  0,
            -1,  1, -1,   0,  1,  0,
            -1,  1,  1,   0,  1,  0,
             1,  1, -1,   0,  1,  0,

            -1, -1,  1,  -1,  0,  0,
            -1,  1,  1,  -1,  0,  0,
            -1,  1, -1,  -1,  0,  0,
            -1, -1, -1,  -1,  0,  0,
            -1, -1,  1,  -1,  0,  0,
            -1,  1, -1,  -1,  0,  0,

             1,  1,  1,   0,  0,  1,
            -1,  1,  1,   0,  0,  1,
            -1, -1,  1,   0,  0,  1,
            -1, -1,  1,   0,  0,  1,
             1, -1,  1,   0,  0,  1,
             1,  1,  1,   0,  0,  1,

             1, -1, -1,   0,  0, -1,
            -1, -1, -1,   0,  0, -1,
            -1,  1, -1,   0,  0, -1,
             1,  1, -1,   0,  0, -1,
             1, -1, -1,   0,  0, -1,
            -1,  1, -1,   0,  0, -1,
        ]);

        this.vertexCount = this.vertexArray.length / 6;
        this.topology = 'triangle-list';
    }
}



export class Rect extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);

        const num_triangles = 2;

        this.topology = 'triangle-list';
        this.vertexCount = num_triangles * 3;

        const v = new Float32Array(this.vertexCount * (3 + 3));

        const xys = [
            [ -1, -1 ],
            [  1,  1 ],
            [ -1,  1 ],

            [  1,  1 ],
            [ -1, -1 ],
            [  1, -1 ],
        ]

        let idx = 0;
        for(let vert_i = 0; vert_i < this.vertexCount; vert_i++){
            const [x, y] = xys[vert_i];

            v[idx + 0] = x;
            v[idx + 1] = y;
            v[idx + 2] = 0;

            v[idx + 3] = 0;
            v[idx + 4] = 0;
            v[idx + 5] = 1;

            idx += 6;
        }

        this.vertexArray = v;
    }
}



export class Disc extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);

        const num_division = 16;

        this.topology = 'triangle-list';
        this.vertexCount = num_division * 3;

        const v = new Float32Array(this.vertexCount * (3 + 3));

        let idx = 0;
        for(let tri_i = 0; tri_i < num_division; tri_i++){

            for(let vert_i = 0; vert_i < 3; vert_i++){
                let x : number, y : number;

                if(vert_i == 0){

                    const phi = 2.0 * Math.PI * tri_i / num_division;
                    x = Math.cos(phi);
                    y = Math.sin(phi);
                }
                else if(vert_i == 1){

                    const phi = 2.0 * Math.PI * (tri_i + 1) / num_division;
                    x = Math.cos(phi);
                    y = Math.sin(phi);                        
                }
                else{
                    x = 0;
                    y = 0;
                }

                v[idx + 0] = x;
                v[idx + 1] = y;
                v[idx + 2] = 0;

                v[idx + 3] = 0;
                v[idx + 4] = 0;
                v[idx + 5] = 1;

                idx += 6;
            }
        }        

        this.vertexArray = v;
    }
}

export class Cone extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);

        const num_division = 16;

        this.topology = 'triangle-list';
        this.vertexCount = num_division * 3;
        this.vertexArray = new Float32Array(this.vertexCount * (3 + 3));

        const xs = new Float32Array(3);
        const ys = new Float32Array(3);
        const zs = new Float32Array(3);

        const nxs = new Float32Array(3);
        const nys = new Float32Array(3);
        const nzs = new Float32Array(3);

        const root2 = Math.sqrt(2);

        for(let tri_i = 0; tri_i < num_division; tri_i++){

            // coneを横から見たら正三角形になるようにする。
            // 正三角形なので半径が1なら斜辺の長さは2になる。
            // この正三角形の高さは√3
            const tri_h = Math.sqrt(3);

            // 正三角形の斜辺の傾きは60°なので、斜辺の法線の傾きは30°
            // 斜辺の法線の高さはsin(30°)
            const norm_h = Math.sin(Math.PI / 6);

            // 1st vertex
            const phi1 = 2.0 * Math.PI * tri_i / num_division;
            xs[0] = Math.cos(phi1);
            ys[0] = Math.sin(phi1);
            zs[0] = -tri_h;

            nxs[0] = xs[0];
            nys[0] = ys[0];
            nzs[0] = norm_h;

            // 2nd vertex
            const phi2 = 2.0 * Math.PI * (tri_i + 1) / num_division;
            xs[1] = Math.cos(phi2);
            ys[1] = Math.sin(phi2);
            zs[1] = -tri_h;

            nxs[1] = xs[1];
            nys[1] = ys[1];
            nzs[1] = norm_h;

            // 3rd vertex
            const phi3 = 2.0 * Math.PI * (tri_i + 0.5) / num_division;
            xs[2] = 0;
            ys[2] = 0;
            zs[2] = 0;

            nxs[2] = Math.cos(phi3);
            nys[2] = Math.sin(phi3);
            nzs[2] = norm_h;
    
            for(let i = 0; i < 3; i++){
                const base = (tri_i * 3 + i) * (3 + 3);

                this.vertexArray[base    ] = xs[i];
                this.vertexArray[base + 1] = ys[i];
                this.vertexArray[base + 2] = zs[i];

                this.vertexArray[base + 3] = nxs[i] / root2;
                this.vertexArray[base + 4] = nys[i] / root2;
                this.vertexArray[base + 5] = nzs[i] / root2;
            }
        }
    }
}


export class CompositeRenderPipeline extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);
    }
}

export class Axis extends CompositeRenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);
    }
}

export function makeArrow(shape : ShapeInfo) : RenderPipeline[] {

    const disc1 = new Disc(shape);
    disc1.shapeInfo  = new Float32Array([ 1, 1, 0, 0]);

    const disc2 = new Disc(shape);
    disc2.shapeInfo  = new Float32Array([ 1, 2, 0, 0]);
    
    const tube  = new Tube(shape);
    tube.shapeInfo   = new Float32Array([ 1, 3, 0, 0]);
    
    const cone  = new Cone(shape);
    cone.shapeInfo   = new Float32Array([ 1, 4, 0, 0]);

    return [ disc1, tube, disc2, cone ];
}

export function makeLines(shape : ShapeInfo) : RenderPipeline[] {
    const th = Math.PI / 3.0;
    const s = 100 * Math.sin(th);
    const c = 100 * Math.cos(th);

    const line = new Line(shape, new Float32Array([ 
        -100,0,0, 1,0,0, 
         100,0,0, 1,0,0,

        0,-100,0, 0,1,0,
        0, 100,0, 0,1,0,
        
        0,0,-100, 0,0,1,
        0,0, 100, 0,0,1,

        -c,-s,0, 1,1,0,
         c, s,0, 1,1,0,

         0,-c,-s, 0,1,1,
         0, c, s, 0,1,1,

         -s,0,-c, 1,0,1,
          s,0, c, 1,0,1
    ]));

    return [ line ];
}



export class GeodesicPolyhedron extends RenderPipeline {
    constructor(shape : ShapeInfo){
        super(shape);

        this.topology = 'triangle-list';

        const divide_cnt = 3;

        const [ points1, triangles1, sphere_r ] = makeRegularIcosahedron();
        const [ points2, triangles2, edges ] = divideTriangle(points1, triangles1, sphere_r, divide_cnt);
    
        this.vertexCount = triangles2.length * 3;
        this.vertexArray = new Float32Array(this.vertexCount * (3 + 3) );
    
        let idx = 0;
        for(let i = 0; i < triangles2.length; i++){
            const tri = triangles2[i];
            for(let j = 0; j < 3; j++){
                const vert = tri.Vertexes[j];
    
                this.vertexArray[idx    ] = vert.x;
                this.vertexArray[idx + 1] = vert.y;
                this.vertexArray[idx + 2] = vert.z;
    
                const len = vert.len();
    
                this.vertexArray[idx + 3] = vert.x / len;
                this.vertexArray[idx + 4] = vert.y / len;
                this.vertexArray[idx + 5] = vert.z / len;
    
                idx += 3 + 3;
            }
        }

        if(shape.scale != undefined){
            const [sx, sy, sz] = shape.scale;
            for(let i = 0; i < this.vertexArray.length; i +=6 ){
                this.vertexArray[i    ] *= sx;
                this.vertexArray[i + 1] *= sy;
                this.vertexArray[i + 2] *= sz;
            }
        }

        if(shape.position != undefined){
            const [dx, dy, dz] = shape.position;
            for(let i = 0; i < this.vertexArray.length; i +=6 ){
                this.vertexArray[i    ] += dx;
                this.vertexArray[i + 1] += dy;
                this.vertexArray[i + 2] += dz;
            }
        }
    }
}

function makeRegularIcosahedron() : [ Vertex[], Triangle[], number ] {
    var G = (1 + Math.sqrt(5)) / 2;

    // 頂点のリスト
    var points = [
        new Vertex( 1,  G,  0), // 0
        new Vertex( 1, -G,  0), // 1
        new Vertex(-1,  G,  0), // 2
        new Vertex(-1, -G,  0), // 3

        new Vertex( 0,  1,  G), // 4
        new Vertex( 0,  1, -G), // 5
        new Vertex( 0, -1,  G), // 6
        new Vertex( 0, -1, -G), // 7

        new Vertex( G,  0,  1), // 8
        new Vertex(-G,  0,  1), // 9
        new Vertex( G,  0, -1), // 10
        new Vertex(-G,  0, -1), // 11
    ];

    /*
0 2 4
2 0 5
0 4 8
5 0 10
0 8 10
3 1 6
1 3 7
6 1 8
1 7 10
8 1 10
4 2 9
2 5 11
9 2 11
3 6 9
7 3 11
3 9 11
4 6 8
6 4 9
7 5 10
5 7 11        
    */

    var sphere_r = vecLen(points[0]);

    points.forEach(function (x) {
        console.assert(Math.abs(sphere_r - vecLen(x)) < 0.001);
    });


    // 三角形のリスト
    var triangles : Triangle[] = []

    for (var i1 = 0; i1 < points.length; i1++) {
        for (var i2 = i1 + 1; i2 < points.length; i2++) {
            //            println("%.2f : %d %d %.2f", sphere_r, i1, i2, vecDiff(points[i1], points[i2]));

            if (Math.abs(vecDiff(points[i1], points[i2]) - 2) < 0.01) {
                for (var i3 = i2 + 1; i3 < points.length; i3++) {
                    if (Math.abs(vecDiff(points[i2], points[i3]) - 2) < 0.01 && Math.abs(vecDiff(points[i1], points[i3]) - 2) < 0.01) {

                        var pnts = [ points[i1], points[i2], points[i3] ]

                        var tri = new Triangle(pnts[0], pnts[1], pnts[2]);
                        for (var i = 0; i < 3; i++) {
                            pnts[i].adjacentVertexes.push(pnts[(i + 1) % 3], pnts[(i + 2) % 3])
                        }
                            
//                            println("正20面体 %d %d %d", points.indexOf(tri.Vertexes[0]), points.indexOf(tri.Vertexes[1]), points.indexOf(tri.Vertexes[2]))

                        triangles.push(tri);
                    }
                }
            }
        }
    }
    console.assert(triangles.length == 20);

    points.forEach(function (p) {
        // 隣接する頂点の重複を取り除く。
        p.adjacentVertexes = Array.from(new Set(p.adjacentVertexes));

        console.assert(p.adjacentVertexes.length == 5);
    });

    return [ points, triangles, sphere_r ];
}

function divideTriangle(points1: Vertex[], triangles: Triangle[], sphere_r: number, divideCnt: number) : [ Vertex[], Triangle[], Edge[] ] {
    let points2 = Array.from(points1) as Vertex[];
    let edges : Edge[] = [];
    let mids = new Map<Edge, Vertex>();

    for (var divide_idx = 0; divide_idx < divideCnt; divide_idx++) {

        // 三角形を分割する。
        var new_triangles: Triangle[] = [];

        triangles.forEach(function (x) {
            // 三角形の頂点のリスト。
            var pnts = [ x.Vertexes[0], x.Vertexes[1], x.Vertexes[2] ];

            // 中点のリスト
            var midpoints : Vertex[] = [];

            for (var i1 = 0; i1 < 3; i1++) {

                // 三角形の2点
                var p1 = pnts[i1];
                var p2 = pnts[(i1 + 1) % 3];

                // 2点をつなぐ辺を探す。
                var edge = edges.find(x => x.Endpoints[0] == p1 && x.Endpoints[1] == p2 || x.Endpoints[1] == p1 && x.Endpoints[0] == p2);
                if (edge == undefined) {
                    // 2点をつなぐ辺がない場合

                    // 2点をつなぐ辺を作る。
                    edge = new Edge(p1, p2);

                    // 辺の中点を作る。
                    mids.set(edge, new Vertex((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2));

                    for (var i = 0; i < i; k++) {

                        var k = edge.Endpoints[i].adjacentVertexes.indexOf(edge.Endpoints[(i + 1) % 2]);
                        console.assert(k != -1);
                        edge.Endpoints[i].adjacentVertexes[k] = mids.get(edge)!;
                    }

                    edges.push(edge);
                }

                var mid = mids.get(edge)!;

                midpoints.push(mid);

                var d = vecLen(mid);
                mid.x *= sphere_r / d;
                mid.y *= sphere_r / d;
                mid.z *= sphere_r / d;

                points2.push(mid);

                console.assert(Math.abs(sphere_r - vecLen(mid)) < 0.001);
            }

            for (var i = 0; i < 3; i++) {
                var pnt = pnts[i];
                var mid = midpoints[i];

                if (mid.adjacentVertexes.length == 0) {

                    mid.adjacentVertexes.push(pnts[(i + 1) % 3], midpoints[(i + 1) % 3], midpoints[(i + 2) % 3], pnts[i]);
                }
                else {

                    mid.adjacentVertexes.push(pnts[(i + 1) % 3], midpoints[(i + 2) % 3]);
                }
            }

            new_triangles.push(new Triangle(midpoints[0], midpoints[1], midpoints[2], true));
            new_triangles.push(new Triangle(pnts[0], midpoints[0], midpoints[2], true));
            new_triangles.push(new Triangle(pnts[1], midpoints[1], midpoints[0], true));
            new_triangles.push(new Triangle(pnts[2], midpoints[2], midpoints[1], true));
        });

        points2.forEach(function (p) {
            console.assert(p.adjacentVertexes.length == 5 || p.adjacentVertexes.length == 6);
        });

        triangles = new_triangles;
    }

    /*

    var new_triangles = [];
    triangles.forEach(function (x) {
        if (x.Vertexes.every(p => p.adjacentVertexes.length == 6)) {
            new_triangles.push(x);
        }
    });
    triangles = new_triangles;
    */


   console.log(`半径:${sphere_r} 三角形 ${triangles.length}`);

    return [ points2, triangles, edges ];
}

function setPosNorm(v : Float32Array, idx : number, x : number, y : number, z : number, nx : number, ny : number, nz : number){
    const base = idx * (3 + 3);

    v[base    ] = x;
    v[base + 1] = y;
    v[base + 2] = z;

    v[base + 3] = nx;
    v[base + 4] = ny;
    v[base + 5] = nz;
}
