import lygia::math::rotate4dZ::rotate4dZ;

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateZ(<vec3|vec4> pos, float radian [, vec3 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotateZ4_c(v: vec4f, r: f32, c: vec4f) -> vec4f {
    return rotate4dZ(r) * (v - c) + c;
}

@if(CENTER_4D)
fn rotateZ4(v: vec4f, r: f32) -> vec4f {
    return rotate4dZ(r) * (v - CENTER_4D) + CENTER_4D;
}
@if(!CENTER_4D)
fn rotateZ4(v: vec4f, r: f32) -> vec4f {
    return rotate4dZ(r) * v;
}

fn rotateZ3_c(v: vec3f, r: f32, c: vec3f) -> vec3f {
    return (rotate4dZ(r) * vec4f(v - c, 0.0)).xyz + c;
}

@if(CENTER_3D)
fn rotateZ3(v: vec3f, r: f32) -> vec3f {
    return (rotate4dZ(r) * vec4f(v - CENTER_3D, 0.0)).xyz + CENTER_3D;
}
@if(!CENTER_3D)
fn rotateZ3(v: vec3f, r: f32) -> vec3f {
    return (rotate4dZ(r) * vec4f(v, 0.0)).xyz;
}
