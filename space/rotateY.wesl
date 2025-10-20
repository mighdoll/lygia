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

fn rotateY4(v: vec4f, r: f32) -> vec4f {
    @if(CENTER_4D)
    let centered = v - constants::CENTER_4D;
    @else
    let centered = v;

    let result = rotate4dY(r) * centered;

    @if(CENTER_4D)
    return result + constants::CENTER_4D;
    @else
    return result;
}

fn rotateY3_c(v: vec3f, r: f32, c: vec3f) -> vec3f {
    return (rotate4dY(r) * vec4f(v - c, 1.0)).xyz + c;
}

fn rotateY3(v: vec3f, r: f32) -> vec3f {
    @if(CENTER_3D)
    let centered = v - constants::CENTER_3D;
    @else
    let centered = v;

    let result = (rotate4dY(r) * vec4f(centered, 1.0)).xyz;

    @if(CENTER_3D)
    return result + constants::CENTER_3D;
    @else
    return result;
}
