import lygia::color::space::{hsl2rgb::hsl2rgb, rgb2hsl::rgb2hsl};
import lygia::math::consts::TAU;

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue
use: hueShift(<vec3f|vec4f> color, <f32> angle)
options:
    - HUESHIFT_AMOUNT: if defined, uses a normalized value instead of an angle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(!HUESHIFT_AMOUNT)
fn hueShift(color: vec3f, a: f32) -> vec3f {
    var hsl = rgb2hsl(color);
    hsl.r = hsl.r * TAU + a;
    hsl.r = fract(hsl.r / TAU);
    return hsl2rgb(hsl);
}

@if(HUESHIFT_AMOUNT)
fn hueShift(color: vec3f, a: f32) -> vec3f {
    var hsl = rgb2hsl(color);
    hsl.r += a;
    return hsl2rgb(hsl);
}

fn hueShift4(color: vec4f, a: f32) -> vec4f {
    return vec4f(hueShift(color.rgb, a), color.a);
}
