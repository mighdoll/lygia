import lygia::math::rotate2d::rotate2d;
import lygia::math::rotate4d::rotate4d;

@if(CENTER_2D)
import constants::CENTER_2D;

@if(CENTER_3D)
import constants::CENTER_3D;

@if(CENTER_4D)
import constants::CENTER_4D;

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian r
use: rotate(<vec3|vec2> v, float r [, vec2 c])
options:
    - CENTER_2D: boolean condition to enable custom 2D center point (requires constants::CENTER_2D vec2f value from linker)
    - CENTER_3D: boolean condition to enable custom 3D center point (requires constants::CENTER_3D vec3f value from linker)
    - CENTER_4D: boolean condition to enable custom 4D center point (requires constants::CENTER_4D vec4f value from linker)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotate_c(v: vec2f, r: f32, c: vec2f) -> vec2f {
    return rotate2d(r) * (v - c) + c;
}

fn rotate(v: vec2f, r: f32) -> vec2f {
    @if(CENTER_2D)
    return rotate_c(v, r, constants::CENTER_2D);
    @else
    return rotate_c(v, r, vec2f(0.5));
}

fn rotate_axis(v: vec2f, x_axis: vec2f) -> vec2f {
    @if(CENTER_2D)
    let centered = v - constants::CENTER_2D;
    @else
    let centered = v;

    let rta = vec2f(dot(centered, vec2f(-x_axis.y, x_axis.x)), dot(centered, x_axis));

    @if(CENTER_2D)
    return rta + constants::CENTER_2D;
    @else
    return rta;
}

fn rotate3_c(v: vec3f, r: f32, axis: vec3f, c: vec3f) -> vec3f {
    return (rotate4d(axis, r) * vec4f(v - c, 1.0)).xyz + c;
}

fn rotate3(v: vec3f, r: f32, axis: vec3f) -> vec3f {
    @if(CENTER_3D)
    return rotate3_c(v, r, axis, constants::CENTER_3D);
    @else
    return rotate3_c(v, r, axis, vec3f(0.0));
}

fn rotate4_c(v: vec4f, r: f32, axis: vec3f, c: vec4f) -> vec4f {
    return rotate4d(axis, r) * (v - c) + c;
}

fn rotate4(v: vec4f, r: f32, axis: vec3f) -> vec4f {
    @if(CENTER_4D)
    return rotate4_c(v, r, axis, constants::CENTER_4D);
    @else
    return rotate4_c(v, r, axis, vec4f(0.0));
}

// Note: Quaternion rotation variants are not yet supported in WESL
// They require FNC_QUATMULT to be defined, which is deferred until
// the quaternion module is fully converted to WESL.
