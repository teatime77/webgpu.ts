struct Uniforms {
    // @uniform
    viewMatrix        : mat4x4<f32>,
    normMatrix        : mat4x4<f32>,

    materialColor     : vec4<f32>,
    ambientColor      : vec4<f32>,
    directionalColor  : vec4<f32>,
    lightingDirection : vec4<f32>,
    env               : vec4<f32>,
    shapeInfo         : vec4<f32>
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec4<f32>,
}


struct Complex {
    real: f32,
    imag: f32,
};

fn comp(a: f32, b: f32) -> Complex {
    return Complex(a, b);
}

fn add(a: Complex, b: Complex) -> Complex {
    return Complex(a.real + b.real, a.imag + b.imag);
}

fn sub(a: Complex, b: Complex) -> Complex {
    return Complex(a.real - b.real, a.imag - b.imag);
}

fn mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

fn mulR(a: f32, b: Complex) -> Complex {
    return Complex(a * b.real, a * b.imag);
}

fn absC(c: Complex) -> f32 {
    return sqrt(c.real * c.real + c.imag * c.imag);
}

fn arg(c: Complex) -> f32 {
    return atan2(c.imag, c.real);
}

fn pow2(a: Complex) -> Complex {
    return mul(a, a);
}

fn pow3(a: Complex) -> Complex {
    return mul(pow2(a), a);
}

fn conjugate(a: Complex) -> Complex {
    return Complex(a.real, -a.imag);
}

fn rgba(theta_arg : f32) -> vec4<f32>{
    const pi      = 3.14159265359;

    var theta = theta_arg;
    if(theta < 0){
        theta += 2 * pi;
    }

    var r : f32 = 0;
    var g : f32 = 0;
    var b : f32 = 0;

    var t = theta * 3.0 / (2.0 * pi);
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


fn colorC(c: Complex) -> vec4<f32> {
    var theta = arg(c);
    return rgba(theta);
}


@vertex
fn main(
    @location(0) vertPos: vec3<f32>,
    @location(1) vertNorm: vec3<f32>,
    @location(2) meshPos : vec4<f32>,
    @location(3) meshVec : vec4<f32>
) -> VertexOutput {
    const pi      = 3.14159265359;
    const sz      = 100;
    const max_xy  = 2.0;
    const step    = (2.0 * max_xy) / f32(sz);
    const scale   = step / 2.0;

    var x = meshPos.x + scale * vertPos.x;
    var y = meshPos.y + scale * vertPos.y;

    // var c  = comp(x, y);

    // var c2 = pow2(c);
    // var c3 = pow3(c);
    // var c2 = add(pow3(c), mulR(-6.0, pow2(c)));
    // var c2 = comp()
    // var z = sin(x + y);
    // var z  = sqrt(x*x + y*y);

    // var t = atan2(y, x);
    // var color4 = rgba(t);

    var z = exp(x);
    var color4 = rgba(y);

    // var z  = absC(c2);
    // var color4 = colorC(c2);

    var pos = vec4<f32>(x, y, z, 1.0);

    var output : VertexOutput;

    output.Position = uniforms.viewMatrix * pos;

    var transformedNormal = uniforms.normMatrix * vec4<f32>(vertNorm, 1.0);
    var directionalLightWeighting = max(dot(transformedNormal.xyz, uniforms.lightingDirection.xyz), 0.0);
    output.fragColor = color4 * (uniforms.ambientColor + directionalLightWeighting * uniforms.directionalColor);
    
    return output;
}
