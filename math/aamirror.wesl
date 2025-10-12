import lygia::math::nyquist::nyquist;

/*
contributors: Shadi El Hajj
description: An anti-aliased triangle wave function.
use: <f32> aamirror(<f32> x)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn aamirror(x: f32) -> f32 {
    let afwidth = length(vec2(dpdx(x), dpdy(x)));
    let v = abs(x - floor(x + 0.5)) * 2.0;
    return nyquist(v, afwidth);
}
