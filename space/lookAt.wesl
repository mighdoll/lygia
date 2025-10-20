/*
contributors: Patricio Gonzalez Vivo
description: create a look at matrix. Right handed by default.
use:
    - <mat3f> lookAt(<vec3f> forward, <vec3f> up)
    - <mat3f> lookAt3(<vec3f> eye, <vec3f> center, <vec3f> up)
    - <mat3f> lookAt3Roll(<vec3f> eye, <vec3f> center, <f32> roll)
    - <mat3f> lookAt1(<vec3f> forward)
options:
    - LOOK_AT_LEFT_HANDED: assume a left-handed coordinate system
    - LOOK_AT_RIGHT_HANDED: assume a right-handed coordinate system
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lookAt(forward: vec3f, up: vec3f) -> mat3x3f {
    let zaxis = normalize(forward);

    @if(LOOK_AT_RIGHT_HANDED)
    let xaxis = normalize(cross(zaxis, up));
    @if(!LOOK_AT_RIGHT_HANDED)
    let xaxis = normalize(cross(up, zaxis));

    @if(LOOK_AT_RIGHT_HANDED)
    let yaxis = cross(xaxis, zaxis);
    @if(!LOOK_AT_RIGHT_HANDED)
    let yaxis = cross(zaxis, xaxis);

    return mat3x3f(xaxis, yaxis, zaxis);
}

fn lookAt3(eye: vec3f, center: vec3f, up: vec3f) -> mat3x3f {
    let forward = normalize(center - eye);
    return lookAt(forward, up);
}

fn lookAt3Roll(eye: vec3f, center: vec3f, roll: f32) -> mat3x3f {
    let up = vec3f(sin(roll), cos(roll), 0.0);
    return lookAt3(eye, center, up);
}

fn lookAt1(forward: vec3f) -> mat3x3f {
    return lookAt(forward, vec3f(0.0, 1.0, 0.0));
}
