@if(CENTER_3D)
import constants::CENTER_3D;

@if(CENTER_4D)
import constants::CENTER_4D;

import lygia::math::rotate4dY::rotate4dY;

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateY(<vec3> pos, float radian [, vec4 center])
options:
    - CENTER_3D: boolean condition to enable custom 3D center point (requires constants::CENTER_3D vec3f value from linker)
    - CENTER_4D: boolean condition to enable custom 4D center point (requires constants::CENTER_4D vec4f value from linker)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotateY4_c(v: vec4f, r: f32, c: vec4f) -> vec4f {
    return rotate4dY(r) * (v - c) + c;
}

@if(CENTER_4D)
fn rotateY4(v: vec4f, r: f32) -> vec4f {
    return rotate4dY(r) * (v - constants::CENTER_4D) + constants::CENTER_4D;
}
@if(!CENTER_4D)
fn rotateY4(v: vec4f, r: f32) -> vec4f {
    return rotate4dY(r) * v;
}

fn rotateY3_c(v: vec3f, r: f32, c: vec3f) -> vec3f {
    return (rotate4dY(r) * vec4f(v - c, 1.0)).xyz + c;
}

@if(CENTER_3D)
fn rotateY3(v: vec3f, r: f32) -> vec3f {
    return (rotate4dY(r) * vec4f(v - constants::CENTER_3D, 1.0)).xyz + constants::CENTER_3D;
}
@if(!CENTER_3D)
fn rotateY3(v: vec3f, r: f32) -> vec3f {
    return (rotate4dY(r) * vec4f(v, 1.0)).xyz;
}
