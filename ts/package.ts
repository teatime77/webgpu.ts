namespace webgputs {

class ComputeInfo {
    compName!    : string;
    varNames!    : string[];
    shapes!      : ShapeInfo[];
    params!      : string;
}

class ShapeInfo {
    type!    : string;
    // vertName : string | undefined = undefined;
    // fragName : string | undefined = undefined;
}

export class Package {
    computes : ComputeInfo[] | undefined;
    shapes   : ShapeInfo[] | undefined;
}

export function makeMesh(shape : ShapeInfo) : RenderPipeline[] {
    switch(shape.type){
        case "Cone"  : return [new Cone()];
        case "Tube"  : return [new Tube()];
        case "Cube"  : return [new Cube()];
        case "Disc"  : return [new Disc()];
        case "arrow" : return makeArrow();
        case "lines" : return makeLines();
        case "GeodesicPolyhedron" : return [new GeodesicPolyhedron()];
    }
    
    throw Error("shape info");
}


}