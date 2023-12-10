namespace webgputs {

export function makeSphere() : [number, Float32Array]{
    const [top_number, top_array] = makeConeSub(true);
    const [bottom_number, bottom_array] = makeConeSub(false);

    const concat_array = new Float32Array(top_array.length + bottom_array.length);
    concat_array.set(top_array);
    concat_array.set(bottom_array, top_array.length);

    return [top_number + bottom_number, concat_array]
}

function makeSphere3() : Float32Array{
    const top_array    = makeConeSub3(true);
    const bottom_array = makeConeSub3(false);

    const concat_array = new Float32Array(top_array.length + bottom_array.length);
    concat_array.set(top_array);
    concat_array.set(bottom_array, top_array.length);

    return concat_array;
}

export function makeCone() : [number, Float32Array]{
    return makeConeSub(true);
}

function makeConeSub(is_top : boolean) : [number, Float32Array]{
    const phiCnt = 16;
    const cubeVertexCount = 3 * phiCnt;

    const cubeVertexArray = new Float32Array(cubeVertexCount * (4 + 4));
    let theta : number;
    if(is_top){
        theta = 0.2 * Math.PI;
    }
    else{
        theta = 0.8 * Math.PI;
    }

    for(let idx = 0; idx < cubeVertexCount; idx++){        
        const phi1 = 2.0 * Math.PI * idx / phiCnt;
        const phi2 = 2.0 * Math.PI * (idx + 1) / phiCnt;

        let x : number;
        let y : number;
        let z : number;

        let rd = 0.0;
        let gr = 0.0;
        let bl = 0.0;

        const j = idx % 3;
        const k = (idx / 3) % 2;
        if(j == 0){
            x = 0.0;
            y = 0.0;
            if(is_top){

                z = 1.0;
            }
            else{

                z = - 1.0;
            }

            rd = 1.0;
        }
        else{
            z = Math.cos(theta);
            const r = Math.abs(Math.sin(theta));

            if(j == 1 && k == 0 || j == 2 && k == 1){

                x = r * Math.cos(phi1);
                y = r * Math.sin(phi1);
    
                gr = 1.0;
            }
            else{
                x = r * Math.cos(phi2);
                y = r * Math.sin(phi2);
                z = Math.cos(theta);
    
                bl = 1.0;
            }
        }

        const base = idx * (4 + 4);

        cubeVertexArray[base] = x;
        cubeVertexArray[base + 1] = y;
        cubeVertexArray[base + 2] = z;
        cubeVertexArray[base + 3] = 1.0;

        cubeVertexArray[base + 4] = rd;
        cubeVertexArray[base + 5] = gr;
        cubeVertexArray[base + 6] = bl;
        cubeVertexArray[base + 7] = 1.0;
    }

    console.log("make cone");
    return [cubeVertexCount, cubeVertexArray];
}


export function makeConeSub3(is_top : boolean) : Float32Array{
    const phiCnt = 16;
    const cubeVertexCount = 3 * phiCnt;

    const cubeVertexArray = new Float32Array(cubeVertexCount * (3 + 3));
    let theta : number;
    if(is_top){
        theta = 0.2 * Math.PI;
    }
    else{
        theta = 0.8 * Math.PI;
    }

    let base = 0;
    for(let idx = 0; idx < phiCnt; idx++){        
        const phi1 = 2.0 * Math.PI * idx / phiCnt;
        const phi2 = 2.0 * Math.PI * (idx + 1) / phiCnt;

        for(let j = 0; j < 3; j++){
            let x : number;
            let y : number;
            let z : number;

            if(j == 0){
                x = 0.0;
                y = 0.0;

                if(is_top){

                    z = 1.0;
                }
                else{

                    z = - 1.0;
                }
            }
            else{
                z = Math.cos(theta);
                const r = Math.abs(Math.sin(theta));

                if(j == 1){   // j == 1 && k == 0 || j == 2 && k == 1

                    x = r * Math.cos(phi2);
                    y = r * Math.sin(phi2);
                }
                else{
                    x = r * Math.cos(phi1);
                    y = r * Math.sin(phi1);
                }
            }

            cubeVertexArray[base    ] = x;
            cubeVertexArray[base + 1] = y;
            cubeVertexArray[base + 2] = z;

            cubeVertexArray[base + 3] = x;
            cubeVertexArray[base + 4] = y;
            cubeVertexArray[base + 5] = z;

            base += 3 + 3;
        }
    }

    console.log("make cone");
    return cubeVertexArray;
}




export function makeCube() : [number, Float32Array]{
    const cubeVertexArray = new Float32Array([
        // float4 position, float4 color
        1, -1, 1, 1, 1, 0, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,
        1, -1, -1, 1, 1, 0, 0, 1,
        1, -1, 1, 1, 1, 0, 1, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,

        1, 1, 1, 1, 1, 1, 1, 1,
        1, -1, 1, 1, 1, 0, 1, 1,
        1, -1, -1, 1, 1, 0, 0, 1,
        1, 1, -1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, -1, -1, 1, 1, 0, 0, 1,

        -1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, -1, 1, 1, 1, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
        -1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, -1, 1, 1, 1, 0, 1,

        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, 1, 1, 1, 0, 1, 1, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,

        1, 1, 1, 1, 1, 1, 1, 1,
        -1, 1, 1, 1, 0, 1, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        -1, -1, 1, 1, 0, 0, 1, 1,
        1, -1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,

        1, -1, -1, 1, 1, 0, 0, 1,
        -1, -1, -1, 1, 0, 0, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
        1, 1, -1, 1, 1, 1, 0, 1,
        1, -1, -1, 1, 1, 0, 0, 1,
        -1, 1, -1, 1, 0, 1, 0, 1,
    ]);

    const cubeVertexCount = cubeVertexArray.length / (4 + 4);

    return [cubeVertexCount , cubeVertexArray];
}


function makeCube3() : Float32Array{
    const cubeVertexArray = new Float32Array([
        // float4 position, float4 color
        1, -1, 1,
        -1, -1, 1,
        -1, -1, -1,
        1, -1, -1,
        1, -1, 1,
        -1, -1, -1,

        1, 1, 1,
        1, -1, 1,
        1, -1, -1,
        1, 1, -1,
        1, 1, 1,
        1, -1, -1,

        -1, 1, 1,
        1, 1, 1,
        1, 1, -1,
        -1, 1, -1,
        -1, 1, 1,
        1, 1, -1,

        -1, -1, 1,
        -1, 1, 1,
        -1, 1, -1,
        -1, -1, -1,
        -1, -1, 1,
        -1, 1, -1,

        1, 1, 1,
        -1, 1, 1,
        -1, -1, 1,
        -1, -1, 1,
        1, -1, 1,
        1, 1, 1,

        1, -1, -1,
        -1, -1, -1,
        -1, 1, -1,
        1, 1, -1,
        1, -1, -1,
        -1, 1, -1,
    ]);

    return cubeVertexArray;
}

}
