/*
contributors: Patricio Gonzalez Vivo
description: clamp a value between 0 and the medium precision max (65504.0) for floating points
use: <f32|vec2f|vec3f|vec4f> saturateMediump(<f32|vec2f|vec3f|vec4f> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const MEDIUMP_FLT_MAX: f32 = 65504.0;

fn saturateMediump(v: f32) -> f32 {
    @if(TARGET_MOBILE)
    return min(v, MEDIUMP_FLT_MAX);
    @if(!TARGET_MOBILE)
    return v;
}

fn saturateMediump2(v: vec2f) -> vec2f {
    @if(TARGET_MOBILE)
    return min(v, vec2(MEDIUMP_FLT_MAX));
    @if(!TARGET_MOBILE)
    return v;
}

fn saturateMediump3(v: vec3f) -> vec3f {
    @if(TARGET_MOBILE)
    return min(v, vec3(MEDIUMP_FLT_MAX));
    @if(!TARGET_MOBILE)
    return v;
}

fn saturateMediump4(v: vec4f) -> vec4f {
    @if(TARGET_MOBILE)
    return min(v, vec4(MEDIUMP_FLT_MAX));
    @if(!TARGET_MOBILE)
    return v;
}
