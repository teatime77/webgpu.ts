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


fn rotz(x : f32) -> mat3x3<f32> {
    return mat3x3<f32>(
         cos(x),  sin(x), 0.0,
        -sin(x),  cos(x), 0.0,
            0.0,     0.0, 1.0
    );
}

fn roty(x : f32) -> mat3x3<f32> {
    return mat3x3<f32>(
        cos(x), 0.0, -sin(x),
        0.0   , 1.0,  0.0   ,
         sin(x), 0.0,  cos(x)
    );
}

fn rotVec(pos : vec3<f32>, dir : vec3<f32>) -> vec3<f32> {
    var zaxis = vec3<f32>(0.0, 0.0, 1.0);
    var v1    = normalize(dir);
    var c     = dot(v1, zaxis);
    var theta = acos(c);
    var m1    = roty(theta);
    var pos2  = m1 * pos;

    var z2    = m1 * zaxis;
    var v2    = normalize(vec2<f32>(v1.x, v1.y));
    var v3    = normalize(vec2<f32>(z2.x, z2.y));
    if(length(v2) < 0.9 || length(v3) < 0.9){
        return pos2;
    }
    // var phi   = atan2(v2.y - v3.y, v2.x - v3.x);
    var phi   = atan2(v1.y, v1.x);
    var m2    = rotz(phi);

    return m2 * pos2;
}

fn conjugate_q(q : vec4<f32>) -> vec4<f32> {
  return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn mulq(q1 : vec4<f32>, q2 : vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    q2.w * q1.x - q2.z * q1.y + q2.y * q1.z + q2.x * q1.w,
    q2.z * q1.x + q2.w * q1.y - q2.x * q1.z + q2.y * q1.w,
    -q2.y * q1.x + q2.x * q1.y + q2.w * q1.z + q2.z * q1.w,
    -q2.x * q1.x - q2.y * q1.y - q2.z * q1.z + q2.w * q1.w
  );
}

fn axisAngle(axis : vec3<f32>, radian : f32) -> vec4<f32> {
  var naxis = normalize(axis);
  var h = 0.5 * radian;
  var s = sin(h);
  return vec4<f32>(naxis.x * s, naxis.y * s, naxis.z * s, cos(h));
}

fn rotate(v : vec3<f32>, q : vec4<f32>) -> vec3<f32> {
  // norm of q must be 1.
  var vq = vec4<f32>(v.x, v.y, v.z, 0.0);
  var cq = conjugate_q(q);
  var mq = mulq(mulq(cq, vq), q);
  return vec3<f32>(mq.x, mq.y, mq.z);
}

fn rotVec2(pos : vec3<f32>, dir : vec3<f32>) -> vec3<f32> {
    var z = vec3<f32>(0.0,0.0,0.1);
    var axis = cross(z, dir);
    if(length(axis) < 0.001){
        return pos;
    }

    var radian = acos(dot(z, normalize(dir)));
    var q = axisAngle(axis, radian);

    return rotate(pos, q);
}

fn scale(v : vec3<f32>, x : f32, y : f32, z : f32) -> vec3<f32> {
    return vec3<f32>(x * v.x, y * v.y, z * v.z);
}


@vertex
fn main(
    @location(0) vertPos: vec3<f32>,
    @location(1) vertNorm: vec3<f32>,
    @location(2) meshPos : vec4<f32>,
    @location(3) meshVec : vec4<f32>
) -> VertexOutput {

    var pos = vertPos;

    const disk1 = 1.0;
    const disk2 = 2.0;
    const tube  = 3.0;
    const cone  = 4.0;
    const cone_h = 0.1;

    if(uniforms.shapeInfo.x == 1.0){
        if(uniforms.shapeInfo.y == cone){

            pos.x *= 0.2;
            pos.y *= 0.2;
        }
        else{

            pos.x *= 0.1;
            pos.y *= 0.1;
        }

        var len = length(meshVec.xyz);
        if(uniforms.shapeInfo.y == tube){
            // pos = scale(pos, 1.0, 1.0, length(meshVec.xyz));
            pos.z *= max(0.0, len - cone_h);
        }
        if(uniforms.shapeInfo.y == cone){
            if(cone_h <= len){

                pos.z *= cone_h;
            }
            else{

                pos.z *= len;
            }
        }

        pos = rotVec(pos, meshVec.xyz);
        if(cone_h <= len){

            if(uniforms.shapeInfo.y == disk2){

                pos += (len - cone_h) * meshVec.xyz;
            }
            if(uniforms.shapeInfo.y == cone){

                pos += meshVec.xyz;
            }
        }
    }

    var output : VertexOutput;

    output.Position = uniforms.viewMatrix * (meshPos + vec4<f32>(pos, 1));


    var transformedNormal = uniforms.normMatrix * vec4<f32>(vertNorm, 1.0);
    var directionalLightWeighting = max(dot(transformedNormal.xyz, uniforms.lightingDirection.xyz), 0.0);
    output.fragColor = uniforms.materialColor * (uniforms.ambientColor + directionalLightWeighting * uniforms.directionalColor);
    
    return output;
}
