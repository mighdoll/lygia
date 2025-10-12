/*
contributors: Patricio Gonzalez Vivo
description: Convert camera depth to view depth. based on https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: <float> depth2viewZ( <float> depth [, <float> near, <float> far] )
options:
    - CAMERA_NEAR_CLIP
    - CAMERA_FAR_CLIP
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CAMERA_ORTHOGRAPHIC_PROJECTION)
fn depth2viewZ(depth: f32, near: f32, far: f32) -> f32 {
    // ORTHOGRAPHIC
    return depth * (near - far) - near;
}
@if(!CAMERA_ORTHOGRAPHIC_PROJECTION)
fn depth2viewZ(depth: f32, near: f32, far: f32) -> f32 {
    // PERSPECTIVE
    return (near * far) / ((far - near) * depth - far);
}

@if(CAMERA_NEAR_CLIP && CAMERA_FAR_CLIP)
fn depth2viewZ_default(depth: f32) -> f32 {
    return depth2viewZ(depth, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP);
}
