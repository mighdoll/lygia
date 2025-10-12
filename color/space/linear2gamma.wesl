/*
contributors: Patricio Gonzalez Vivo
description: Convert from linear to gamma color space.
use: linear2gamma(<f32|vec3f|vec4f> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
options:
    - GAMMA: gamma value (default 2.2 on desktop, 2.0 on mobile/webgl)
*/

@if(GAMMA)
fn linear2gamma(v: f32) -> f32 {
    return pow(v, 1.0/GAMMA);
}

@if(!GAMMA)
fn linear2gamma(v: f32) -> f32 {
    // assume gamma 2.0
    return sqrt(v);
}

@if(GAMMA)
fn linear2gamma3(v: vec3f) -> vec3f {
    return pow(v, vec3f(1.0/GAMMA));
}

@if(!GAMMA)
fn linear2gamma3(v: vec3f) -> vec3f {
    // assume gamma 2.0
    return sqrt(v);
}

fn linear2gamma4(v: vec4f) -> vec4f {
    return vec4f(linear2gamma3(v.rgb), v.a);
}
