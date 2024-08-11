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
    // vertName : string | undefined = undefined;
    // fragName : string | undefined = undefined;
}

export class Package {
    computes : ComputeInfo[] | undefined;
    shapes   : ShapeInfo[] | undefined;
}

export function makeMesh(shape : ShapeInfo, is_instance : boolean) : RenderPipeline[] {
    switch(shape.type){
        case "Cone"  : return [new Cone()];
        case "Tube"  : return [new Tube()];
        case "Cube"  : return [new Cube()];
        case "Disc"  : return [new Disc()];
        case "point" : return [new Point()];
        case "line"  : return [new Line()];
        case "arrow" : return makeArrow();
        case "lines" : return makeLines();
        case "GeodesicPolyhedron" : return [new GeodesicPolyhedron(shape, is_instance)];
    }
    
    throw Error("shape info");
}


}