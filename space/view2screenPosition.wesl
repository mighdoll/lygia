/*
contributors: Patricio Gonzalez Vivo
description: get screen coordinates from view position
use: <vec2f> view2screenPosition(<vec3f> viewPosition )
options:
    - CAMERA_PROJECTION_MATRIX: mat4x4f matrix with camera projection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CAMERA_PROJECTION_MATRIX)
fn view2screenPosition(viewPosition: vec3f) -> vec2f {
    let clip = CAMERA_PROJECTION_MATRIX * vec4f(viewPosition, 1.0);
    return (clip.xy / clip.w + 1.0) * 0.5;
}

// Note: If CAMERA_PROJECTION_MATRIX is not defined, this function cannot be used.
// Please define CAMERA_PROJECTION_MATRIX before using this function.
