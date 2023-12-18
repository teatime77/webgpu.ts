namespace webgputs {

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

}
