import lygia::math::decimate::decimate3;

/*
contributors: Patricio Gonzalez Vivo
description: 'Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf'
use:
    - <vec4f|vec3f|f32> ditherVlachos(<vec4f|vec3f|f32> value, <vec2f> st, <i32> precision)
    - <vec4f|vec3f|f32> ditherVlachos(<vec4f|vec3f|f32> value, <vec2f> st)
    - <vec4f|vec3f|f32> ditherVlachos(<vec4f|vec3f|f32> value)
    - <vec3f> ditherVlachos(<vec2f> st)
options:
    - DITHER_VLACHOS_COORD: coordinate to use (default: vec2f(0.0))
    - DITHER_VLACHOS_TIME: time value for animated dithering
    - DITHER_VLACHOS_PRECISION: precision for decimation (default: 256)
    - DITHER_PRECISION: global dither precision (default: 256)
    - DITHER_TIME: global dither time
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(!DITHER_VLACHOS_COORD)
const DITHER_VLACHOS_COORD = vec2f(0.0);

@if(DITHER_TIME)
const DITHER_VLACHOS_TIME = DITHER_TIME;

@if(!DITHER_VLACHOS_PRECISION && DITHER_PRECISION)
const DITHER_VLACHOS_PRECISION: i32 = DITHER_PRECISION;

@if(!DITHER_VLACHOS_PRECISION && !DITHER_PRECISION)
const DITHER_VLACHOS_PRECISION: i32 = 256;


// Core Vlachos noise function for scalar
@if(DITHER_VLACHOS_TIME)
fn ditherVlachos(b: f32, ist: vec2f) -> f32 {
    var st = ist;
    st += 1337.0 * fract(DITHER_VLACHOS_TIME);

    let noise = dot(vec2f(171.0, 231.0), st);
    let noise_fract = fract(noise / 71.0);
    let noise_normalized = (noise_fract * 2.0) - 1.0;
    return b + (noise_normalized / 255.0);
}

@if(!DITHER_VLACHOS_TIME)
fn ditherVlachos(b: f32, ist: vec2f) -> f32 {
    let st = ist;

    let noise = dot(vec2f(171.0, 231.0), st);
    let noise_fract = fract(noise / 71.0);
    let noise_normalized = (noise_fract * 2.0) - 1.0;
    return b + (noise_normalized / 255.0);
}

// Core Vlachos noise function returning vec3
@if(DITHER_VLACHOS_TIME)
fn ditherVlachos3Noise(ist: vec2f) -> vec3f {
    var st = ist;
    st += 1337.0 * fract(DITHER_VLACHOS_TIME);

    var noise = vec3f(dot(vec2f(171.0, 231.0), st));
    noise = fract(noise / vec3f(103.0, 71.0, 97.0));
    return noise;
}

@if(!DITHER_VLACHOS_TIME)
fn ditherVlachos3Noise(ist: vec2f) -> vec3f {
    let st = ist;

    var noise = vec3f(dot(vec2f(171.0, 231.0), st));
    noise = fract(noise / vec3f(103.0, 71.0, 97.0));
    return noise;
}

// Main dithering function for vec3 with precision control
fn ditherVlachos3Precision(color: vec3f, st: vec2f, pres: i32) -> vec3f {
    let d = f32(pres);
    let ditherPattern = ditherVlachos3Noise(st);
    return decimate3(color + ditherPattern / d, vec3f(d));
}

// Overloads with xy coordinate using default precision
fn ditherVlachos3(color: vec3f, xy: vec2f) -> vec3f {
    return ditherVlachos3Precision(color, xy, DITHER_VLACHOS_PRECISION);
}

fn ditherVlachos4(color: vec4f, xy: vec2f) -> vec4f {
    return vec4f(ditherVlachos3Precision(color.rgb, xy, DITHER_VLACHOS_PRECISION), color.a);
}

// Overloads using default coordinate
fn ditherVlachos1(val: f32) -> f32 {
    return ditherVlachos(val, DITHER_VLACHOS_COORD);
}

fn ditherVlachos3Simple(color: vec3f) -> vec3f {
    return ditherVlachos3Precision(color, DITHER_VLACHOS_COORD, DITHER_VLACHOS_PRECISION);
}

fn ditherVlachos4Simple(color: vec4f) -> vec4f {
    return vec4f(ditherVlachos3Precision(color.rgb, DITHER_VLACHOS_COORD, DITHER_VLACHOS_PRECISION), color.a);
}
