namespace webgputs {

class ComputeInfo {
    compName!    : string;
    varNames!    : string[];
    shapes!      : ShapeInfo[];
    params!      : string;
}

export class ShapeInfo {
    type!    : string;
    scale    : [ number, number, number ] | undefined;
    position : [ number, number, number ] | undefined;
    vertName : string | undefined = undefined;
    // fragName : string | undefined = undefined;
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
        case "point" : return [new Point(shape)];
        case "line"  : return [new Line(shape)];
        case "arrow" : return makeArrow(shape);
        case "lines" : return makeLines(shape);
        case "GeodesicPolyhedron" : return [new GeodesicPolyhedron(shape)];
    }
    
    throw Error("shape info");
}


}