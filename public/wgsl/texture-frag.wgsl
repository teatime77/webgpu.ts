@group(0) @binding(1) var myTexture: texture_2d<f32>;
@group(0) @binding(2) var mySampler: sampler;

@fragment
fn main(
    @location(0) fragUV: vec2<f32>,
) -> @location(0) vec4<f32> {
    return textureSample(myTexture, mySampler, fragUV);
}
