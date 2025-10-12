import lygia::math::decimate::decimate;
import lygia::math::decimate::decimate3;
import lygia::math::saturate::saturate3;

/*
contributors: Patricio Gonzalez Vivo
description: blue noise dithering
use:
    - <vec4f|vec3f|f32> ditherBlueNoise(<vec4f|vec3f|f32> value, <vec2f> st, <i32> precision)
    - <vec4f|vec3f|f32> ditherBlueNoise(<vec4f|vec3f|f32> value, <vec2f> st)
    - <vec4f|vec3f|f32> ditherBlueNoise(<vec4f|vec3f|f32> value)
options:
    - DITHER_BLUENOISE_COORD: coordinate to use (default: vec2f(0.0))
    - DITHER_BLUENOISE_TIME: time value for animated dithering
    - DITHER_BLUENOISE_CHROMATIC: enable chromatic dithering
    - DITHER_BLUENOISE_PRECISION: precision for decimation (default: 256)
    - DITHER_PRECISION: global dither precision (default: 256)
    - DITHER_TIME: global dither time
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(!DITHER_BLUENOISE_COORD)
const DITHER_BLUENOISE_COORD = vec2f(0.0);

@if(DITHER_TIME)
const DITHER_BLUENOISE_TIME = DITHER_TIME;

@if(!DITHER_BLUENOISE_PRECISION && DITHER_PRECISION)
const DITHER_BLUENOISE_PRECISION: i32 = DITHER_PRECISION;

@if(!DITHER_BLUENOISE_PRECISION && !DITHER_PRECISION)
const DITHER_BLUENOISE_PRECISION: i32 = 256;


// Core blue noise generation function
@if(DITHER_BLUENOISE_TIME)
fn ditherBlueNoise(inp: vec2f) -> f32 {
    let SEED1 = 1.705;
    let size = 5.5;
    var p = floor(inp);
    var p1 = p;

    p += 1337.0 * fract(DITHER_BLUENOISE_TIME * 0.1);

    p = floor(p / size) * size;
    p = fract(p * 0.1) + 1.0 + p * vec2f(0.0002, 0.0003);
    var a = fract(1.0 / (0.000001 * p.x * p.y + 0.00001));
    a = fract(1.0 / (0.000001234 * a + 0.00001));
    var b = fract(1.0 / (0.000002 * (p.x * p.y + p.x) + 0.00001));
    b = fract(1.0 / (0.0000235 * b + 0.00001));
    let r = vec2f(a, b) - 0.5;
    p1 += r * 8.12235325;

    return fract(p1.x * SEED1 + p1.y / (SEED1 + 0.15555));
}

@if(!DITHER_BLUENOISE_TIME)
fn ditherBlueNoise(inp: vec2f) -> f32 {
    let SEED1 = 1.705;
    let size = 5.5;
    var p = floor(inp);
    var p1 = p;

    p += 10.0;

    p = floor(p / size) * size;
    p = fract(p * 0.1) + 1.0 + p * vec2f(0.0002, 0.0003);
    var a = fract(1.0 / (0.000001 * p.x * p.y + 0.00001));
    a = fract(1.0 / (0.000001234 * a + 0.00001));
    var b = fract(1.0 / (0.000002 * (p.x * p.y + p.x) + 0.00001));
    b = fract(1.0 / (0.0000235 * b + 0.00001));
    let r = vec2f(a, b) - 0.5;
    p1 += r * 8.12235325;

    return fract(p1.x * SEED1 + p1.y / (SEED1 + 0.15555));
}

// Main dithering function for vec3 with precision control
fn ditherBlueNoise3Precision(color: vec3f, xy: vec2f, pres: i32) -> vec3f {
    let d = f32(pres);
    let dVec = vec3f(d);
    let decimated = decimate3(color, dVec);
    let diff = (color - decimated) * d;
    return saturate3(decimate3(color + step(vec3f(ditherBlueNoise(xy)), diff) / d, dVec));
}

// Overloads with xy coordinate and precision
fn ditherBlueNoisePrecision(val: f32, xy: vec2f, pres: i32) -> f32 {
    return ditherBlueNoise3Precision(vec3f(val), xy, pres).r;
}

fn ditherBlueNoise4Precision(color: vec4f, xy: vec2f, pres: i32) -> vec4f {
    return vec4f(ditherBlueNoise3Precision(color.rgb, xy, pres), color.a);
}

// Overloads with xy coordinate using default precision
fn ditherBlueNoise1(val: f32, xy: vec2f) -> f32 {
    return ditherBlueNoise3Precision(vec3f(val), xy, DITHER_BLUENOISE_PRECISION).r;
}

fn ditherBlueNoise3(color: vec3f, xy: vec2f) -> vec3f {
    return ditherBlueNoise3Precision(color, xy, DITHER_BLUENOISE_PRECISION);
}

fn ditherBlueNoise4(color: vec4f, xy: vec2f) -> vec4f {
    return vec4f(ditherBlueNoise3Precision(color.rgb, xy, DITHER_BLUENOISE_PRECISION), color.a);
}

// Overloads using default coordinate and default precision
fn ditherBlueNoise1Simple(val: f32) -> f32 {
    return ditherBlueNoise3Precision(vec3f(val), DITHER_BLUENOISE_COORD, DITHER_BLUENOISE_PRECISION).r;
}

fn ditherBlueNoise3Simple(color: vec3f) -> vec3f {
    return ditherBlueNoise3Precision(color, DITHER_BLUENOISE_COORD, DITHER_BLUENOISE_PRECISION);
}

fn ditherBlueNoise4Simple(color: vec4f) -> vec4f {
    return vec4f(ditherBlueNoise3Precision(color.rgb, DITHER_BLUENOISE_COORD, DITHER_BLUENOISE_PRECISION), color.a);
}
