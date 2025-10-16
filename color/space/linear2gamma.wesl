/*
contributors: Patricio Gonzalez Vivo
description: Convert from linear to gamma color space (assumes gamma 2.0).
use: linear2gamma(<f32|vec3f|vec4f> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER: make gamma a @param const when WESL supports that
fn linear2gamma(v: f32) -> f32 {
    // assume gamma 2.0
    return sqrt(v);
}

fn linear2gamma3(v: vec3f) -> vec3f {
    // assume gamma 2.0
    return sqrt(v);
}

fn linear2gamma4(v: vec4f) -> vec4f {
    return vec4f(linear2gamma3(v.rgb), v.a);
}
