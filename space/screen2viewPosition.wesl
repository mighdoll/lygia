import lygia::math::inverse::inverse4;

/*
contributors: Patricio Gonzalez Vivo
description: derive view surface position from screen coordinates and depth
use: <vec4f> screen2viewPosition(<vec2f> screenPosition, <f32> depth, <f32> viewZ)
options:
    - CAMERA_PROJECTION_MATRIX: mat4x4f matrix with camera projection
    - INVERSE_CAMERA_PROJECTION_MATRIX: mat4x4f matrix with the inverse camera projection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CAMERA_PROJECTION_MATRIX && INVERSE_CAMERA_PROJECTION_MATRIX)
fn screen2viewPosition(screenPosition: vec2f, depth: f32, viewZ: f32) -> vec4f {
    let clipW = CAMERA_PROJECTION_MATRIX[2][3] * viewZ + CAMERA_PROJECTION_MATRIX[3][3];
    let clipPosition = vec4f((vec3f(screenPosition, depth) - 0.5) * 2.0, 1.0) * clipW;
    return INVERSE_CAMERA_PROJECTION_MATRIX * clipPosition;
}

@if(CAMERA_PROJECTION_MATRIX && !INVERSE_CAMERA_PROJECTION_MATRIX)
fn screen2viewPosition(screenPosition: vec2f, depth: f32, viewZ: f32) -> vec4f {
    let INVERSE_CAMERA_PROJECTION_MATRIX = inverse4(CAMERA_PROJECTION_MATRIX);
    let clipW = CAMERA_PROJECTION_MATRIX[2][3] * viewZ + CAMERA_PROJECTION_MATRIX[3][3];
    let clipPosition = vec4f((vec3f(screenPosition, depth) - 0.5) * 2.0, 1.0) * clipW;
    return INVERSE_CAMERA_PROJECTION_MATRIX * clipPosition;
}

// Note: If CAMERA_PROJECTION_MATRIX is not defined, this function cannot be used.
// Please define CAMERA_PROJECTION_MATRIX before using this function.
