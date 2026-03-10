import { Cone, Cube, Disc, GeodesicPolyhedron, Line, makeArrow, Point, Rect, RenderPipeline, Surface, Tube } from "./primitive.js";

class ComputeInfo {
    compName!    : string;
    shapes!      : ShapeInfo[];
    globalGrid!  : number[];
}

export class ShapeInfo {
    type!    : string;
    scale    : [ number, number, number ] | undefined;
    position : [ number, number, number ] | undefined;
    gridSize : number[] | undefined;
    vertName : string | undefined = undefined;
    fragName : string | undefined = undefined;
}

export class Package {
    computes : ComputeInfo[] | undefined;
    shapes   : ShapeInfo[] | undefined;
}

export function makeMesh(shape : ShapeInfo) : RenderPipeline[] {
    switch(shape.type){
        case "Rect"  : return [new Rect(shape)];
        case "Cone"  : return [new Cone(shape)];
        case "Tube"  : return [new Tube(shape)];
        case "Cube"  : return [new Cube(shape)];
        case "Disc"  : return [new Disc(shape)];
        case "Point" : return [new Point(shape)];
        case "Line"  : return [new Line(shape)];
        case "arrow" : return makeArrow(shape);
        case "GeodesicPolyhedron" : return [new GeodesicPolyhedron(shape)];
        case "Surface": return [new Surface(shape)];
    }
    
    throw Error("shape info");
}
