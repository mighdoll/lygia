/*
contributors: Patricio Gonzalez Vivo
description: clamp a value between 0 and the medium precision max (65504.0) for floating points
use: <f32|vec2f|vec3f|vec4f> saturateMediump(<f32|vec2f|vec3f|vec4f> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const MEDIUMP_FLT_MAX: f32 = 65504.0;

@if(TARGET_MOBILE)
fn saturateMediump(v: f32) -> f32 { return min(v, MEDIUMP_FLT_MAX); }

@if(!TARGET_MOBILE)
fn saturateMediump(v: f32) -> f32 { return v; }

@if(TARGET_MOBILE)
fn saturateMediump2(v: vec2f) -> vec2f { return min(v, vec2(MEDIUMP_FLT_MAX)); }

@if(!TARGET_MOBILE)
fn saturateMediump2(v: vec2f) -> vec2f { return v; }

@if(TARGET_MOBILE)
fn saturateMediump3(v: vec3f) -> vec3f { return min(v, vec3(MEDIUMP_FLT_MAX)); }

@if(!TARGET_MOBILE)
fn saturateMediump3(v: vec3f) -> vec3f { return v; }

@if(TARGET_MOBILE)
fn saturateMediump4(v: vec4f) -> vec4f { return min(v, vec4(MEDIUMP_FLT_MAX)); }

@if(!TARGET_MOBILE)
fn saturateMediump4(v: vec4f) -> vec4f { return v; }
