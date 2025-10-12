import lygia::color::space::rgb2oklab::RGB2OKLAB_B;
import lygia::color::space::oklab2rgb::OKLAB2RGB_B;
import lygia::color::space::{srgb2rgb::srgb2rgb, rgb2srgb::rgb2srgb};

/*
contributors:
    - Bjorn Ottosson
    - Inigo Quiles
description: |
    Mix function by Inigo Quiles (https://www.shadertoy.com/view/ttcyRS)
    utilizing Bjorn Ottosso's OkLab color space, which provides smooth transitions
    Learn more about it [his article](https://bottosson.github.io/posts/oklab/)
use: <vec3f|vec4f> mixOklab(<vec3f|vec4f> colorA, <vec3f|vec4f> colorB, <f32> pct)
options:
    - MIXOKLAB_SRGB: color arguments are in sRGB and returns sRGB
examples:
    - /shaders/color_mix.frag
license:
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
    - MIT License (MIT) Copyright (c) 2020 Inigo Quilez
*/

@if(MIXOKLAB_SRGB)
fn mixOklab(colA: vec3f, colB: vec3f, h: f32) -> vec3f {
    let colorA = srgb2rgb(colA);
    let colorB = srgb2rgb(colB);

    // rgb to cone (arg of pow can't be negative)
    let lmsA = pow(RGB2OKLAB_B * colorA, vec3f(0.33333));
    let lmsB = pow(RGB2OKLAB_B * colorB, vec3f(0.33333));

    let lms = mix(lmsA, lmsB, h);

    // cone to rgb
    let rgb = OKLAB2RGB_B * (lms * lms * lms);

    return rgb2srgb(rgb);
}

@if(!MIXOKLAB_SRGB)
fn mixOklab(colA: vec3f, colB: vec3f, h: f32) -> vec3f {
    let colorA = colA;
    let colorB = colB;

    // rgb to cone (arg of pow can't be negative)
    let lmsA = pow(RGB2OKLAB_B * colorA, vec3f(0.33333));
    let lmsB = pow(RGB2OKLAB_B * colorB, vec3f(0.33333));

    let lms = mix(lmsA, lmsB, h);

    // cone to rgb
    let rgb = OKLAB2RGB_B * (lms * lms * lms);

    return rgb;
}

fn mixOklab4(colA: vec4f, colB: vec4f, h: f32) -> vec4f {
    return vec4f(mixOklab(colA.rgb, colB.rgb, h), mix(colA.a, colB.a, h));
}
