@if(CAMERA_NEAR_CLIP)
import constants::CAMERA_NEAR_CLIP;

@if(CAMERA_FAR_CLIP)
import constants::CAMERA_FAR_CLIP;

/*
contributors: Patricio Gonzalez Vivo
description: Convert camera depth to view depth. based on https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: <float> depth2viewZ( <float> depth [, <float> near, <float> far] )
options:
    - CAMERA_NEAR_CLIP: boolean condition for using near clip value (requires constants::CAMERA_NEAR_CLIP f32 from linker)
    - CAMERA_FAR_CLIP: boolean condition for using far clip value (requires constants::CAMERA_FAR_CLIP f32 from linker)
    - CAMERA_ORTHOGRAPHIC_PROJECTION: boolean condition for orthographic vs perspective projection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn depth2viewZ(depth: f32, near: f32, far: f32) -> f32 {
    @if(CAMERA_ORTHOGRAPHIC_PROJECTION)
    // ORTHOGRAPHIC
    return depth * (near - far) - near;
    @if(!CAMERA_ORTHOGRAPHIC_PROJECTION)
    // PERSPECTIVE
    return (near * far) / ((far - near) * depth - far);
}

@if(CAMERA_NEAR_CLIP && CAMERA_FAR_CLIP)
fn depth2viewZ_default(depth: f32) -> f32 {
    return depth2viewZ(depth, constants::CAMERA_NEAR_CLIP, constants::CAMERA_FAR_CLIP);
}
