import { ComputePipeline } from "./compute.js";
import { CalcRenderPipeline, ComputeRenderPipeline, Cone, Cube, Disc, GeodesicPolyhedron, Line, makeArrow, Point, Rect, RenderPipeline, Surface, Tube } from "./primitive.js";

class ComputeInfo {
    compName!    : string;
    shapes!      : ShapeInfo[];
    globalGrid!  : number[];
}

export class ShapeInfo {
    type!     : string;
    scale?    : [ number, number, number ] | undefined;
    position? : [ number, number, number ] | undefined;
    gridSize? : number[] | undefined;
    vertName  : string | undefined = undefined;
    fragName  : string | undefined = undefined;
    divideCount?: number;
    numDivision?: number;
    renderBufferPair? : boolean | undefined = undefined;
}

export class Package {
    name     : string | undefined;
    computes : ComputeInfo[] | undefined;
    shapes   : ShapeInfo[] | undefined;
}

export function makeComputeRenderPipeline(compute  : ComputePipeline, shape : ShapeInfo) : ComputeRenderPipeline[] {
    switch(shape.type){
        case "Rect"  : return [new Rect(compute, shape)];
        case "Cone"  : return [new Cone(compute, shape)];
        case "Tube"  : return [new Tube(compute, shape)];
        case "Cube"  : return [new Cube(compute, shape)];
        case "Disc"  : return [new Disc(compute, shape)];
        case "arrow" : return makeArrow(compute, shape);
        case "GeodesicPolyhedron" : return [new GeodesicPolyhedron(compute, shape)];
    }
    
    throw Error("shape info");
}

export function makeCalcRenderPipeline(shape : ShapeInfo) : CalcRenderPipeline[] {
    switch(shape.type){
        case "Point" : return [new Point(shape)];
        case "Line"  : return [new Line(shape)];
        case "Surface": return [new Surface(shape)];
    }
    throw Error("shape info");
}
