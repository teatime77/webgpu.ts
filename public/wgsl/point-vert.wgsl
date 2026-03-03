struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat4x4<f32>,

    materialColor     : vec4<f32>,
    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
    lightingDirection : vec4<f32>,
    env               : vec4<f32>,
    shapeInfo         : vec4<f32>,
    gridSize          : vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec4<f32>,
}

const PI: f32 = 3.14159265359;

fn rgba(theta_arg : f32) -> vec4<f32>{
    var theta = theta_arg;
    if(theta < 0){
        theta += 2 * PI;
    }

    var r : f32 = 0;
    var g : f32 = 0;
    var b : f32 = 0;

    var t = theta * 3.0 / (2.0 * PI);
    if(t <= 1.0){
        r = (1.0 - t);
        g = t;
    }
    else{
        t -= 1.0;
        if(t <= 1.0){
            g = (1.0 - t);
            b = t;
        }
        else{
            t -= 1.0;

            b = (1.0 - t);
            r = t;
        }
    }

    return vec4<f32>(r, g, b, 1.0);
}

@vertex
fn main(
    @builtin(vertex_index) vIndex : u32
) -> VertexOutput {
    var xs : u32 = u32(uniforms.gridSize.x);

    var ix = f32(vIndex % xs);
    var iy = f32(vIndex / xs);

    var theta = 2.0 * PI * ix / uniforms.gridSize.x;
    var phi   = PI * iy / uniforms.gridSize.y;

    var r = 3.0;
    var z = r * cos(phi);
    var r2 = r * sin(phi);
    var x = r2 * cos(theta);
    var y = r2 * sin(theta);

    var pos = vec4<f32>(x, y, z, 1.0);
    let color = rgba(theta);

    var output : VertexOutput;

    output.Position = uniforms.viewMatrix * vec4<f32>(pos.xyz, 1.0);

    output.fragColor = color;
    
    return output;
}
