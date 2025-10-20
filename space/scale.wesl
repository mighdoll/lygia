/*
contributors: Patricio Gonzalez Vivo
description: scale a 2D space variable
use: scale(<vec2> st, <vec2|float> scale_factor [, <vec2> center])
options:
    - CENTER_2D: boolean condition to enable custom 2D center point (requires constants::CENTER_2D vec2f value from linker)
    - CENTER_3D: boolean condition to enable custom 3D center point (requires constants::CENTER_3D vec3f value from linker)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CENTER_2D)
import constants::CENTER_2D;

@if(CENTER_3D)
import constants::CENTER_3D;

// Single float variants
fn scale_f_c(st: f32, s: f32, center: vec2f) -> vec2f {
    return (vec2f(st) - center) * s + center;
}

fn scale_f(st: f32, s: f32) -> vec2f {
    @if(CENTER_2D)
    let center = constants::CENTER_2D;
    @else
    let center = vec2f(0.5);

    return (vec2f(st) - center) * s + center;
}

// vec2 variants
fn scale2_c(st: vec2f, s: vec2f, center: vec2f) -> vec2f {
    return (st - center) * s + center;
}

fn scale2_f_c(st: vec2f, s: f32, center: vec2f) -> vec2f {
    return scale2_c(st, vec2f(s), center);
}

fn scale2(st: vec2f, s: vec2f) -> vec2f {
    @if(CENTER_2D)
    let center = constants::CENTER_2D;
    @else
    let center = vec2f(0.5);

    return (st - center) * s + center;
}

fn scale2_f(st: vec2f, s: f32) -> vec2f {
    @if(CENTER_2D)
    let center = constants::CENTER_2D;
    @else
    let center = vec2f(0.5);

    return (st - center) * s + center;
}

// vec3 variants
fn scale3_c(st: vec3f, s: vec3f, center: vec3f) -> vec3f {
    return (st - center) * s + center;
}

fn scale3_f_c(st: vec3f, s: f32, center: vec3f) -> vec3f {
    return (st - center) * s + center;
}

fn scale3(st: vec3f, s: vec3f) -> vec3f {
    @if(CENTER_3D)
    let center = constants::CENTER_3D;
    @else
    let center = vec3f(0.5);

    return (st - center) * s + center;
}

fn scale3_f(st: vec3f, s: f32) -> vec3f {
    @if(CENTER_3D)
    let center = constants::CENTER_3D;
    @else
    let center = vec3f(0.5);

    return (st - center) * s + center;
}

// For tiles
fn scale4_f(st: vec4f, s: f32) -> vec4f {
    return vec4f(scale2_f(st.xy, s), st.zw);
}

fn scale4(st: vec4f, s: vec2f) -> vec4f {
    return vec4f(scale2(st.xy, s), st.zw);
}
