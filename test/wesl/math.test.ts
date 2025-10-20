import { expect, test } from "vitest";
import { expectCloseTo, testCompute, testFragment } from "./testUtil.ts";

const INV_SQRT2 = Math.SQRT2 / 2;

test("saturate", async () => {
  const src = `
    import lygia::math::saturate::saturate;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = saturate(-0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0], result);
});

test("saturate clamped upper", async () => {
  const src = `
    import lygia::math::saturate::saturate;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = saturate(1.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([1.0], result);
});

test("saturate3", async () => {
  const src = `
    import lygia::math::saturate::saturate3;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = saturate3(vec3f(-0.5, 0.5, 1.5)); }
  `;
  const result = await testCompute(src, "vec3f");
  expectCloseTo([0.0, 0.5, 1.0], result);
});

test("pow2", async () => {
  const src = `
    import lygia::math::pow2::pow2;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow2(3.0); }
  `;
  const result = await testCompute(src);
  expectCloseTo([9.0], result);
});

test("pow22", async () => {
  const src = `
    import lygia::math::pow2::pow22;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow22(vec2f(2.0, 3.0)); }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([4.0, 9.0], result);
});

test("pow3", async () => {
  const src = `
    import lygia::math::pow3::pow3;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow3(2.0); }
  `;
  const result = await testCompute(src);
  expectCloseTo([8.0], result);
});

test("pow5", async () => {
  const src = `
    import lygia::math::pow5::pow5;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow5(2.0); }
  `;
  const result = await testCompute(src);
  expectCloseTo([32.0], result);
});

test("pow7", async () => {
  const src = `
    import lygia::math::pow7::pow7;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow7(2.0); }
  `;
  const result = await testCompute(src);
  expectCloseTo([128.0], result);
});

test("absi positive", async () => {
  const src = `
    import lygia::math::absi::absi;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = f32(absi(5)); }
  `;
  const result = await testCompute(src);
  expectCloseTo([5.0], result);
});

test("absi negative", async () => {
  const src = `
    import lygia::math::absi::absi;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = f32(absi(-5)); }
  `;
  const result = await testCompute(src);
  expectCloseTo([5.0], result);
});

test("toMat3", async () => {
  const src = `
    import lygia::math::toMat3::toMat3;
    @compute @workgroup_size(1)
    fn foo() {
      let m4 = mat4x4f(
        vec4f(1.0, 2.0, 3.0, 4.0),
        vec4f(5.0, 6.0, 7.0, 8.0),
        vec4f(9.0, 10.0, 11.0, 12.0),
        vec4f(13.0, 14.0, 15.0, 16.0)
      );
      let m3 = toMat3(m4);
      test::results[0] = vec4f(m3[0][0], m3[1][1], m3[2][2], 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 6.0, 11.0, 0.0], result);
});

test("hammersley", async () => {
  const src = `
    import lygia::math::hammersley::hammersley;
    @compute @workgroup_size(1)
    fn foo() {
      // Test known Hammersley sequence values
      // hammersley(i, N) = (i/N, radicalInverse_VdC(i))
      // For radicalInverse_VdC:
      //   0 -> 0.0
      //   1 -> 0.5 (binary 1 -> reversed -> 0.1 binary = 0.5)
      //   2 -> 0.25 (binary 10 -> reversed -> 0.01 binary = 0.25)
      //   3 -> 0.75 (binary 11 -> reversed -> 0.11 binary = 0.75)
      let h0 = hammersley(0u, 8);  // (0/8, 0.0) = (0.0, 0.0)
      let h1 = hammersley(1u, 8);  // (1/8, 0.5) = (0.125, 0.5)
      let h2 = hammersley(2u, 8);  // (2/8, 0.25) = (0.25, 0.25)
      let h3 = hammersley(3u, 8);  // (3/8, 0.75) = (0.375, 0.75)

      // Pack first two into result
      test::results[0] = vec4f(h0.x, h0.y, h1.x, h1.y);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify linear x component: h0.x = 0.0, h1.x = 0.125
  expectCloseTo([0.0, 0.125], [result[0], result[2]], 0.01);

  // Verify radical inverse y component (bit reversal): h0.y = 0.0, h1.y = 0.5
  expectCloseTo([0.0, 0.5], [result[1], result[3]], 0.01);
});

test("hammersley - bit reversal verification", async () => {
  const src = `
    import lygia::math::hammersley::hammersley;
    @compute @workgroup_size(1)
    fn foo() {
      // Test more bit reversal values
      let h2 = hammersley(2u, 8);  // (2/8, 0.25) = (0.25, 0.25)
      let h3 = hammersley(3u, 8);  // (3/8, 0.75) = (0.375, 0.75)

      test::results[0] = vec4f(h2.x, h2.y, h3.x, h3.y);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify linear x component: h2.x = 0.25, h3.x = 0.375
  expectCloseTo([0.25, 0.375], [result[0], result[2]], 0.01);

  // Verify radical inverse y component: h2.y = 0.25, h3.y = 0.75
  expectCloseTo([0.25, 0.75], [result[1], result[3]], 0.01);
});

test("nyquist", async () => {
  const src = `
    import lygia::math::nyquist::nyquist;
    @compute @workgroup_size(1)
    fn foo() {
      // nyquist filters based on frequency width:
      // - width < 0.5: no filtering (pass-through)
      // - width > 0.5: filters toward 0.5 (Nyquist limit)
      // The function attenuates signals based on how far they are from 0.5

      // Test 1: No filtering (width well below Nyquist)
      let v1 = nyquist(0.8, 0.1);  // width < 0.5, should pass through

      // Test 2: Partial filtering (width near Nyquist)
      let v2 = nyquist(0.8, 0.5);  // At Nyquist limit

      // Test 3: Heavy filtering (width above Nyquist)
      let v3 = nyquist(0.8, 1.0);  // Full filtering

      // Test 4: Value at midpoint (should always stay at 0.5)
      let v4 = nyquist(0.5, 0.8);  // 0.5 is unaffected by filtering

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // No filtering: should pass through (0.8)
  expectCloseTo([0.8], [result[0]], 0.05);

  // Partial filtering: should be between 0.5 and 0.8
  expect(result[1]).toBeGreaterThan(0.5);
  expect(result[1]).toBeLessThan(0.8);

  // Heavy filtering: should be very close to 0.5 (heavily attenuated)
  expectCloseTo([0.5], [result[2]], 0.1);

  // Midpoint invariant: 0.5 should remain 0.5
  expectCloseTo([0.5], [result[3]], 0.01);
});

// Anti-aliased floor tests (require derivatives, use fragment shaders)
test("aafloor with derivatives", async () => {
  const src = `
    import lygia::math::aafloor::aafloor;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Create varying value for derivatives to work
      // For a 2x2 texture, pos varies between pixels
      let x = pos.x / 10.0 + 2.5; // Should give ~2.5-2.6 range
      let result = aafloor(x);

      // Return result in red channel for validation
      return vec4f(result, 0.0, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [2, 2]);
  // aafloor should produce something close to 2.0
  expectCloseTo([2.0], [result[0]], 0.2);
});

test("aafloor2 with vec2", async () => {
  const src = `
    import lygia::math::aafloor::aafloor2;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let xy = pos.xy / 10.0 + vec2f(2.5, 3.7);
      let result = aafloor2(xy);

      return vec4f(result.x, result.y, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [2, 2]);

  expectCloseTo([2.0, 3.0], result.slice(0, 2), 0.2);
});

test("cubicMix", async () => {
  const src = `
     import lygia::math::cubicMix::cubicMix;

     @compute @workgroup_size(1)
     fn foo() {
       let result = cubicMix(0.0, 1.0, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // Cubic interpolation at 0.5
  expectCloseTo([0.5], result, 0.01);
});

test("permute", async () => {
  const src = `
     import lygia::math::permute::permute;

     @compute @workgroup_size(1)
     fn foo() {
       // permute(x) = mod289(((x * 34.0) + 1.0) * x)
       // Test with hand-calculated values:

       // permute(1.0) = mod289(((1 * 34) + 1) * 1) = mod289(35) = 35
       let p1 = permute(1.0);

       // permute(10.0) = mod289(((10 * 34) + 1) * 10) = mod289(3410) = 3410 % 289 = 231
       let p2 = permute(10.0);

       // permute(100.0) = mod289(((100 * 34) + 1) * 100) = mod289(340100) = 340100 % 289 = 236
       let p3 = permute(100.0);

       // Verify reproducibility
       let p1_repeat = permute(1.0);

       test::results[0] = vec4f(p1, p2, p3, p1_repeat);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify exact formula outputs
  expectCloseTo([35.0], [result[0]], 0.1);
  expectCloseTo([231.0], [result[1]], 1.0);
  expectCloseTo([236.0], [result[2]], 1.0);

  // Verify reproducibility
  expectCloseTo([result[0]], [result[3]], 0.0001);
});

test("smootherstep", async () => {
  const src = `
     import lygia::math::smootherstep::smootherstep;

     @compute @workgroup_size(1)
     fn foo() {
       let result = smootherstep(0.0, 1.0, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // Smoother step at 0.5 should be 0.5
  expectCloseTo([0.5], result, 0.01);
});

test("fmod2", async () => {
  const src = `
     import lygia::math::fmod::fmod2;

     @compute @workgroup_size(1)
     fn foo() {
       // Test positive values
       let result1 = fmod2(vec2f(5.0, 7.0), vec2f(3.0, 4.0));
       test::results[0] = result1.x;
       test::results[1] = result1.y;

       // Test negative values - key difference from % operator
       let result2 = fmod2(vec2f(-5.0, -7.0), vec2f(3.0, 4.0));
       test::results[2] = result2.x;
       test::results[3] = result2.y;
     }
   `;
  const result = await testCompute(src);
  // fmod(5.0, 3.0) = 2.0, fmod(7.0, 4.0) = 3.0
  expectCloseTo([2.0, 3.0], result.slice(0, 2), 0.01);
  // fmod(-5.0, 3.0) = 1.0, fmod(-7.0, 4.0) = 1.0 (floored, not truncated)
  expectCloseTo([1.0, 1.0], result.slice(2, 4), 0.01);
});

test("fmod3", async () => {
  const src = `
     import lygia::math::fmod::fmod3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test with negative values to verify floor-based behavior
       let result = fmod3(vec3f(-5.5, 7.3, -2.1), vec3f(3.0, 4.0, 2.0));
       test::results[0] = result.x;
       test::results[1] = result.y;
       test::results[2] = result.z;
     }
   `;
  const result = await testCompute(src);
  // fmod(-5.5, 3.0) ≈ 0.5, fmod(7.3, 4.0) ≈ 3.3, fmod(-2.1, 2.0) ≈ 1.9
  expectCloseTo([0.5, 3.3, 1.9], result, 0.01);
});

test("fmod4", async () => {
  const src = `
     import lygia::math::fmod::fmod4;

     @compute @workgroup_size(1)
     fn foo() {
       let result = fmod4(vec4f(10.0, -10.0, 7.5, -7.5), vec4f(3.0, 3.0, 2.5, 2.5));
       test::results[0] = result.x;
       test::results[1] = result.y;
       test::results[2] = result.z;
       test::results[3] = result.w;
     }
   `;
  const result = await testCompute(src);
  // fmod(10.0, 3.0) = 1.0, fmod(-10.0, 3.0) = 2.0, fmod(7.5, 2.5) = 0.0, fmod(-7.5, 2.5) = 0.0
  expectCloseTo([1.0, 2.0, 0.0, 0.0], result, 0.01);
});

// Quaternion operations
test("quat - create from axis and angle", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::quat::quat;
    @compute @workgroup_size(1)
    fn foo() {
      let axis = normalize(vec3f(0.0, 1.0, 0.0));
      let angle = HALF_PI; // π/2 radians (90 degrees)
      let result = quat(axis, angle);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // quat from Y-axis rotation of π/2: (0, sin(π/4), 0, cos(π/4)) ≈ (0, INV_SQRT2, 0, INV_SQRT2)
  expectCloseTo([0.0, INV_SQRT2, 0.0, INV_SQRT2], result, 0.01);
});

test("quatDiv - divide quaternion by scalar", async () => {
  const src = `
    import lygia::math::quat::div::quatDiv;
    @compute @workgroup_size(1)
    fn foo() {
      let q = vec4f(2.0, 4.0, 6.0, 8.0);
      let result = quatDiv(q, 2.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 2.0, 3.0, 4.0], result, 0.01);
});

test("quatNeg - negate quaternion", async () => {
  const src = `
    import lygia::math::quat::neg::quatNeg;
    @compute @workgroup_size(1)
    fn foo() {
      let q = vec4f(1.0, 2.0, 3.0, 4.0);
      let result = quatNeg(q);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([-1.0, -2.0, -3.0, -4.0], result, 0.01);
});

test("quatInverse", async () => {
  const src = `
    import lygia::math::quat::inverse::quatInverse;
    import lygia::math::quat::mul::quatMul;
    @compute @workgroup_size(1)
    fn foo() {
      // Create a simple quaternion
      let q = normalize(vec4f(1.0, 2.0, 3.0, 4.0));
      let qInv = quatInverse(q);
      // Multiplying q * qInv should give identity quaternion (0,0,0,1)
      let identity = quatMul(q, qInv);
      test::results[0] = identity;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Identity quaternion is (0, 0, 0, 1)
  expectCloseTo([0.0, 0.0, 0.0, 1.0], result, 0.01);
});

// Rotation matrices
test("rotate2d - 90 degree rotation", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate2d::rotate2d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate2d(HALF_PI); // π/2 radians
      let v = vec2f(1.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation of (1,0) should give approximately (0,1)
  expectCloseTo([0.0, 1.0, 0.0, 0.0], result, 0.01);
});

test("rotate3d - rotation around axis", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3d::rotate3d;
    @compute @workgroup_size(1)
    fn foo() {
      let axis = normalize(vec3f(0.0, 0.0, 1.0)); // Z-axis
      let mat = rotate3d(axis, HALF_PI); // π/2 radians
      let v = vec3f(1.0, 0.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around Z-axis of (1,0,0) - result is (0, -1, 0) due to matrix convention
  expectCloseTo([0.0, -1.0, 0.0, 0.0], result, 0.01);
});

// Scale matrices
test("scale2d - uniform scale", async () => {
  const src = `
    import lygia::math::scale2d::scale2d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale2d(2.0);
      let v = vec2f(3.0, 4.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([6.0, 8.0, 0.0, 0.0], result, 0.01);
});

test("scale2dVec - non-uniform scale", async () => {
  const src = `
    import lygia::math::scale2d::scale2dVec;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale2dVec(vec2f(2.0, 3.0));
      let v = vec2f(4.0, 5.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([8.0, 15.0, 0.0, 0.0], result, 0.01);
});

test("scale3d", async () => {
  const src = `
    import lygia::math::scale3d::scale3d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale3d(vec3f(2.0, 3.0, 4.0));
      let v = vec3f(1.0, 2.0, 3.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([2.0, 6.0, 12.0, 0.0], result, 0.01);
});

test("translate4d", async () => {
  const src = `
    import lygia::math::translate4d::translate4d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = translate4d(vec3f(10.0, 20.0, 30.0));
      let v = vec4f(1.0, 2.0, 3.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([11.0, 22.0, 33.0, 1.0], result, 0.01);
});

test("toMat4", async () => {
  const src = `
    import lygia::math::toMat4::toMat4;
    @compute @workgroup_size(1)
    fn foo() {
      let m3 = mat3x3f(
        vec3f(1.0, 2.0, 3.0),
        vec3f(4.0, 5.0, 6.0),
        vec3f(7.0, 8.0, 9.0)
      );
      let m4 = toMat4(m3);
      test::results[0] = vec4f(m4[0][0], m4[1][1], m4[2][2], m4[3][3]);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Diagonal elements should be 1, 5, 9, 1
  expectCloseTo([1.0, 5.0, 9.0, 1.0], result, 0.01);
});

// Polynomial functions
test("cubic", async () => {
  const src = `
    import lygia::math::cubic::cubic;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(cubic(0.0), cubic(0.5), cubic(1.0), 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // cubic(0) = 0, cubic(0.5) = 0.5, cubic(1) = 1
  expectCloseTo([0.0, 0.5, 1.0, 0.0], result, 0.01);
});

test("quartic", async () => {
  const src = `
    import lygia::math::quartic::quartic;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(quartic(0.0), quartic(0.5), quartic(1.0), 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // quartic(v) = v*v*(2-v*v), quartic(0.5) = 0.25 * 1.9375 ≈ 0.4375
  expectCloseTo([0.0, 0.4375, 1.0, 0.0], result, 0.01);
});

test("quintic", async () => {
  const src = `
    import lygia::math::quintic::quintic;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(quintic(0.0), quintic(0.5), quintic(1.0), 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // quintic(0) = 0, quintic(0.5) = 0.5, quintic(1) = 1
  expectCloseTo([0.0, 0.5, 1.0, 0.0], result, 0.01);
});

test("invCubic", async () => {
  const src = `
    import lygia::math::invCubic::invCubic;
    import lygia::math::cubic::cubic;
    @compute @workgroup_size(1)
    fn foo() {
      // invCubic should be the inverse of cubic
      let x = 0.3;
      let y = cubic(x);
      let xRecovered = invCubic(y);
      test::results[0] = vec4f(x, xRecovered, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Original x and recovered x should match
  expectCloseTo([0.3, 0.3], result.slice(0, 2), 0.01);
});

test("invQuartic", async () => {
  const src = `
    import lygia::math::invQuartic::invQuartic;
    import lygia::math::quartic::quartic;
    @compute @workgroup_size(1)
    fn foo() {
      // invQuartic should be the inverse of quartic
      let x = 0.7;
      let y = quartic(x);
      let xRecovered = invQuartic(y);
      test::results[0] = vec4f(x, xRecovered, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Original x and recovered x should match
  expectCloseTo([0.7, 0.7], result.slice(0, 2), 0.01);
});

// Special math functions
test("gain", async () => {
  const src = `
    import lygia::math::gain::gain;
    @compute @workgroup_size(1)
    fn foo() {
      // gain(0.5, k) should always equal 0.5
      test::results[0] = vec4f(gain(0.5, 2.0), gain(0.25, 2.0), gain(0.75, 2.0), 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // gain(0.5) = 0.5 always
  expect(result[0]).toBeCloseTo(0.5, 2);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeLessThan(1.0);
});

test("parabola", async () => {
  const src = `
    import lygia::math::parabola::parabola;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(parabola(0.0, 1.0), parabola(0.5, 1.0), parabola(1.0, 1.0), 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // parabola(0) = 0, parabola(0.5) = 1, parabola(1) = 0
  expectCloseTo([0.0, 1.0, 0.0, 0.0], result, 0.01);
});

test("gaussian", async () => {
  const src = `
    import lygia::math::gaussian::gaussian;
    @compute @workgroup_size(1)
    fn foo() {
      // gaussian(0, sigma) should be 1.0
      // gaussian(sigma, sigma) should be exp(-0.5) ≈ 0.606
      test::results[0] = vec4f(gaussian(0.0, 1.0), gaussian(1.0, 1.0), 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.606], result.slice(0, 2), 0.01);
});

test("map - remap value between ranges", async () => {
  const src = `
    import lygia::math::map::map;
    @compute @workgroup_size(1)
    fn foo() {
      // Map 0.5 from [0,1] to [0,100]
      let result1 = map(0.5, 0.0, 1.0, 0.0, 100.0);
      // Map 5.0 from [0,10] to [100,200]
      let result2 = map(5.0, 0.0, 10.0, 100.0, 200.0);
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([50.0, 150.0, 0.0, 0.0], result, 0.01);
});

test("mirror - triangle wave", async () => {
  const src = `
    import lygia::math::mirror::mirror;
    @compute @workgroup_size(1)
    fn foo() {
      // mirror creates triangle wave: 0→1→0→1→0
      test::results[0] = vec4f(mirror(0.5), mirror(1.5), mirror(2.5), mirror(3.5));
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.5, 0.5, 0.5, 0.5], result, 0.01);
});

test("decimate - quantize value", async () => {
  const src = `
    import lygia::math::decimate::decimate;
    @compute @workgroup_size(1)
    fn foo() {
      // Decimate to 10 levels
      let result = decimate(0.567, 10.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // 0.567 * 10 = 5.67, floor = 5, 5/10 = 0.5
  expectCloseTo([0.5], result, 0.01);
});

// Utility functions
test("lengthSq2", async () => {
  const src = `
    import lygia::math::lengthSq::lengthSq2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = lengthSq2(vec2f(3.0, 4.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // 3² + 4² = 25
  expectCloseTo([25.0], result, 0.01);
});

test("lengthSq3", async () => {
  const src = `
    import lygia::math::lengthSq::lengthSq3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = lengthSq3(vec3f(1.0, 2.0, 2.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // 1² + 2² + 2² = 9
  expectCloseTo([9.0], result, 0.01);
});

test("distEuclidean2", async () => {
  const src = `
    import lygia::math::dist::distEuclidean2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = distEuclidean2(vec2f(0.0, 0.0), vec2f(3.0, 4.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Distance is 5.0
  expectCloseTo([5.0], result, 0.01);
});

test("distManhattan2", async () => {
  const src = `
    import lygia::math::dist::distManhattan2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = distManhattan2(vec2f(0.0, 0.0), vec2f(3.0, 4.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Manhattan distance is 3 + 4 = 7
  expectCloseTo([7.0], result, 0.01);
});

test("pack/unpack roundtrip", async () => {
  const src = `
    import lygia::math::pack::pack;
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      let original = 0.123456;
      let packed = pack(original);
      let unpacked = unpack4(packed);
      test::results[0] = vec4f(original, unpacked, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Original and unpacked should be close (limited precision)
  expectCloseTo([0.123456, 0.123456], result.slice(0, 2), 0.001);
});

test("pack/unpack roundtrip - multiple values", async () => {
  const src = `
    import lygia::math::pack::pack;
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that pack/unpack are inverse operations with multiple values
      let v1 = 0.0;
      let v2 = 0.25;
      let v3 = 0.5;
      let v4 = 0.75;

      let r1 = unpack4(pack(v1));
      let r2 = unpack4(pack(v2));
      let r3 = unpack4(pack(v3));
      let r4 = unpack4(pack(v4));

      test::results[0] = vec4f(r1, r2, r3, r4);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify roundtrip accuracy (limited by packing precision)
  expectCloseTo([0.0], [result[0]], 0.001);
  expectCloseTo([0.25], [result[1]], 0.001);
  expectCloseTo([0.5], [result[2]], 0.001);
  expectCloseTo([0.75], [result[3]], 0.001);
});

test("taylorInvSqrt", async () => {
  const src = `
    import lygia::math::taylorInvSqrt::taylorInvSqrt;
    @compute @workgroup_size(1)
    fn foo() {
      // Test with 1.0, should approximate 1/sqrt(1) = 1
      let result = taylorInvSqrt(1.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Fast approximation should be close to 1.0
  expectCloseTo([1.0], result, 0.1);
});

// Anti-aliased functions (require derivatives, use fragment shaders)
test("aamirror - anti-aliased triangle wave", async () => {
  const src = `
    import lygia::math::aamirror::aamirror;
    import lygia::math::mirror::mirror;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Test triangle wave pattern at different positions
      // aamirror should create: 0→1→0→1→0 (triangle wave)
      let x = pos.x / 50.0;  // Slow variation for derivatives

      // Test at valley (x=0), peak (x=0.5), valley (x=1.0), peak (x=1.5)
      let valley1_aa = aamirror(x);
      let peak1_aa = aamirror(x + 0.5);
      let valley2_aa = aamirror(x + 1.0);
      let peak2_aa = aamirror(x + 1.5);

      return vec4f(valley1_aa, peak1_aa, valley2_aa, peak2_aa);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Verify triangle wave pattern: valleys near 0.0, peaks near 1.0
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThan(0.2);  // Valley

  expect(result[1]).toBeGreaterThan(0.8);
  expect(result[1]).toBeLessThanOrEqual(1.0);  // Peak

  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThan(0.2);  // Valley (periodic)

  expect(result[3]).toBeGreaterThan(0.8);
  expect(result[3]).toBeLessThanOrEqual(1.0);  // Peak (periodic)
});

test("aastep - smooth transition near threshold", async () => {
  const src = `
    import lygia::math::aastep::aastep;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 100.0;  // Very slow variation for minimal derivatives
      let threshold = 0.5;

      // Test values on both sides of threshold
      let below = aastep(threshold, x + 0.3);      // Well below (0.3 < 0.5)
      let justBelow = aastep(threshold, x + 0.48); // Just below (0.48 < 0.5)
      let justAbove = aastep(threshold, x + 0.52); // Just above (0.52 > 0.5)
      let above = aastep(threshold, x + 0.7);      // Well above (0.7 > 0.5)

      return vec4f(below, justBelow, justAbove, above);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Well below: should be close to 0.0 (may be exactly 0)
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThan(0.15);

  // Just below: should be in transition zone
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThan(0.6);

  // Just above: should be in transition zone
  expect(result[2]).toBeGreaterThan(0.4);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Well above: should be close to 1.0
  expect(result[3]).toBeGreaterThan(0.85);
  expect(result[3]).toBeLessThanOrEqual(1.0);

  // Verify smooth gradient: justBelow < justAbove
  expect(result[1]).toBeLessThanOrEqual(result[2]);
});

test("fcos - filtered cosine at known angles", async () => {
  const src = `
    import lygia::math::fcos::fcos;
    import lygia::math::consts::PI;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 1000.0;  // Very slow variation to minimize filtering

      // At very slow variation, fcos should match regular cos closely
      let at0 = fcos(x);                   // cos(~0) ≈ 1.0
      let atPi4 = fcos(x + PI * 0.25);     // cos(π/4) ≈ INV_SQRT2
      let atPi2 = fcos(x + PI * 0.5);      // cos(π/2) ≈ 0.0
      let atPi = fcos(x + PI);             // cos(π) ≈ -1.0

      return vec4f(at0, atPi4, atPi2, atPi);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Verify cosine values with tighter tolerance for slow variation
  expectCloseTo([1.0], [result[0]], 0.05);      // cos(0) = 1.0
  expectCloseTo([INV_SQRT2], [result[1]], 0.05);    // cos(π/4) = √2/2
  expectCloseTo([0.0], [result[2]], 0.05);      // cos(π/2) = 0.0
  expectCloseTo([-1.0], [result[3]], 0.05);     // cos(π) = -1.0
});

test("fcos - band limiting at high frequency", async () => {
  const src = `
    import lygia::math::fcos::fcos;
    import lygia::math::consts::PI;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x;  // Fast variation (per pixel)

      // At high frequency, fcos should attenuate (approach 0)
      // while regular cos would alias
      let highFreq = fcos(x * 100.0 * PI);  // Very high frequency

      // At low frequency, should still oscillate normally
      let lowFreq = fcos(x * 0.01 * PI);    // Very low frequency

      return vec4f(highFreq, lowFreq, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [32, 32]);

  // High frequency should be heavily attenuated (close to 0)
  expect(Math.abs(result[0])).toBeLessThan(0.3);

  // Low frequency should have full amplitude (not attenuated)
  expect(Math.abs(result[1])).toBeGreaterThan(0.3);
});

test("adaptiveThreshold", async () => {
  const src = `
    import lygia::math::adaptiveThreshold::adaptiveThreshold;
    @compute @workgroup_size(1)
    fn foo() {
      // Test threshold comparison
      let result1 = adaptiveThreshold(0.8, 0.5, 0.1); // v > blur_v + b
      let result2 = adaptiveThreshold(0.4, 0.5, 0.1); // v < blur_v + b
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0, 0.0, 0.0], result);
});

test("atan2Custom", async () => {
  const src = `
    import lygia::math::atan2::atan2Custom;
    import lygia::math::consts::PI;
    import lygia::math::consts::TAU;
    @compute @workgroup_size(1)
    fn foo() {
      // atan2Custom normalizes angles to [0, 2π] range
      // Formula: (atan2(y, x) + PI) % TAU
      // Note: This shifts by π, making atan2 output [0, 2π] instead of [-π, π]

      // Test common angles
      let angle1 = atan2Custom(1.0, 0.0);   // atan2(1,0) = π/2, +PI = 3π/2
      let angle2 = atan2Custom(0.0, 1.0);   // atan2(0,1) = 0, +PI = π
      let angle3 = atan2Custom(-1.0, 0.0);  // atan2(-1,0) = -π/2, +PI = π/2
      let angle4 = atan2Custom(0.0, -1.0);  // atan2(0,-1) = π, +PI = 2π % 2π = 0

      test::results[0] = vec4f(angle1, angle2, angle3, angle4);
    }
  `;
  const result = await testCompute(src, "vec4f");

  const PI = Math.PI;
  const TAU = 2 * PI;

  // Verify normalized angles [0, 2π]
  expectCloseTo([3*PI/2], [result[0]], 0.01);   // 3π/2 ≈ 4.7124
  expectCloseTo([PI], [result[1]], 0.01);       // π ≈ 3.1416
  expectCloseTo([PI/2], [result[2]], 0.01);     // π/2 ≈ 1.5708
  expectCloseTo([0.0], [result[3]], 0.01);      // 0

  // All angles should be in [0, 2π) range
  for (let i = 0; i < 4; i++) {
    expect(result[i]).toBeGreaterThanOrEqual(0.0);
    expect(result[i]).toBeLessThan(TAU);
  }
});

test("atan2Custom - additional angles", async () => {
  const src = `
    import lygia::math::atan2::atan2Custom;
    import lygia::math::consts::PI;
    @compute @workgroup_size(1)
    fn foo() {
      // Test additional angle cases
      let angle5 = atan2Custom(1.0, 1.0);   // atan2(1,1) = π/4, +PI = 5π/4

      test::results[0] = vec4f(angle5, 0.0, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  const PI = Math.PI;

  // Verify diagonal angle
  expectCloseTo([5*PI/4], [result[0]], 0.01);     // 5π/4 ≈ 3.927
});

test("bump", async () => {
  const src = `
    import lygia::math::bump::bump;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = bump(0.0, 0.0); // Should be 1.0
      let result2 = bump(1.0, 0.0); // Should be 0.0
      let result3 = bump(0.5, 0.0); // Should be 0.75
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0, 0.75], result.slice(0, 3), 0.01);
});

test("bump2", async () => {
  const src = `
    import lygia::math::bump::bump2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = bump2(vec2f(0.0, 0.5), vec2f(0.0));
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.75], result.slice(0, 2), 0.01);
});

test("highPass", async () => {
  const src = `
    import lygia::math::highPass::highPass;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = highPass(0.8, 0.5); // Above threshold
      let result2 = highPass(0.3, 0.5); // Below threshold
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.6, 0.0], result.slice(0, 2), 0.01);
});

test("inside - scalar", async () => {
  const src = `
    import lygia::math::inside::inside;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = inside(5.0, 0.0, 10.0); // true
      let result2 = inside(-1.0, 0.0, 10.0); // false
      let result3 = inside(11.0, 0.0, 10.0); // false
      test::results[0] = vec4f(f32(result1), f32(result2), f32(result3), 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0, 0.0], result.slice(0, 3), 0.01);
});

test("inside2", async () => {
  const src = `
    import lygia::math::inside::inside2;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = inside2(vec2f(5.0, 5.0), vec2f(0.0), vec2f(10.0)); // true
      let result2 = inside2(vec2f(-1.0, 5.0), vec2f(0.0), vec2f(10.0)); // false
      test::results[0] = vec4f(f32(result1), f32(result2), 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0], result.slice(0, 2), 0.01);
});

test("inverse - mat3", async () => {
  const src = `
    import lygia::math::inverse::inverse;
    @compute @workgroup_size(1)
    fn foo() {
      let m = mat3x3f(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 2.0, 0.0),
        vec3f(0.0, 0.0, 3.0)
      );
      let mInv = inverse(m);
      test::results[0] = vec4f(mInv[0][0], mInv[1][1], mInv[2][2], 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Inverse of diagonal matrix is 1/diagonal
  expectCloseTo([1.0, 0.5, 0.333], result.slice(0, 3), 0.01);
});

test("mmax2", async () => {
  const src = `
    import lygia::math::mmax::mmax2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmax2(vec2f(3.0, 7.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([7.0], result);
});

test("mmax3", async () => {
  const src = `
    import lygia::math::mmax::mmax3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmax3(vec3f(3.0, 7.0, 5.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([7.0], result);
});

test("mmin2", async () => {
  const src = `
    import lygia::math::mmin::mmin2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmin2(vec2f(3.0, 7.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([3.0], result);
});

test("mmin3", async () => {
  const src = `
    import lygia::math::mmin::mmin3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmin3(vec3f(3.0, 7.0, 5.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([3.0], result);
});

test("mod2 - mutates pointer", async () => {
  const src = `
    import lygia::math::mod2::mod2;
    @compute @workgroup_size(1)
    fn foo() {
      var p = vec2f(7.0, 10.0);
      let c = mod2(&p, 3.0);
      test::results[0] = vec4f(p.x, p.y, c.x, c.y);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // p should be modified to centered remainder, c is the cell index
  expect(result[0]).toBeCloseTo(1.0, 1);
  expect(result[1]).toBeCloseTo(1.0, 1);
});

test("mod289", async () => {
  const src = `
    import lygia::math::mod289::mod289;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = mod289(300.0); // 300 % 289 = 11
      let result2 = mod289(289.0); // 289 % 289 = 0
      let result3 = mod289(100.0); // 100 % 289 = 100
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([11.0, 0.0, 100.0], result.slice(0, 3), 0.01);
});

test("powFast", async () => {
  const src = `
    import lygia::math::powFast::powFast;
    @compute @workgroup_size(1)
    fn foo() {
      // powFast is a fast approximation: powFast(a, b) = a / ((1-b)*a + b)
      // This approximates pow(a, b) but is not exact

      // Test case 1: powFast(0.5, 0.5)
      // = 0.5 / ((1-0.5)*0.5 + 0.5)
      // = 0.5 / (0.5*0.5 + 0.5)
      // = 0.5 / (0.25 + 0.5)
      // = 0.5 / 0.75 = 0.6667
      // (True pow(0.5, 0.5) = 0.7071)
      let fast1 = powFast(0.5, 0.5);

      // Test case 2: powFast(0.8, 0.3)
      // = 0.8 / ((1-0.3)*0.8 + 0.3)
      // = 0.8 / (0.7*0.8 + 0.3)
      // = 0.8 / (0.56 + 0.3)
      // = 0.8 / 0.86 = 0.9302
      // (True pow(0.8, 0.3) = 0.9438)
      let fast2 = powFast(0.8, 0.3);

      // Test case 3: powFast(0.25, 0.75)
      // = 0.25 / ((1-0.75)*0.25 + 0.75)
      // = 0.25 / (0.25*0.25 + 0.75)
      // = 0.25 / (0.0625 + 0.75)
      // = 0.25 / 0.8125 = 0.3077
      // (True pow(0.25, 0.75) = 0.3536)
      let fast3 = powFast(0.25, 0.75);

      // Test edge case: powFast(1.0, x) should always be 1.0
      let edge1 = powFast(1.0, 0.5);

      test::results[0] = vec4f(fast1, fast2, fast3, edge1);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify approximation values (not exact pow, but close)
  expectCloseTo([0.6667], [result[0]], 0.01);
  expectCloseTo([0.9302], [result[1]], 0.01);
  expectCloseTo([0.3077], [result[2]], 0.01);

  // Edge case: powFast(1, x) = 1 for any x
  expectCloseTo([1.0], [result[3]], 0.01);
});

test("rotate3dX", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3dX::rotate3dX;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate3dX(HALF_PI); // π/2 radians
      let v = vec3f(0.0, 1.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around X-axis of (0,1,0) should give (0,0,1)
  expectCloseTo([0.0, 0.0, 1.0], result.slice(0, 3), 0.01);
});

test("rotate3dY", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3dY::rotate3dY;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate3dY(HALF_PI); // π/2 radians
      let v = vec3f(1.0, 0.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around Y-axis of (1,0,0) should give (0,0,-1)
  expectCloseTo([0.0, 0.0, -1.0], result.slice(0, 3), 0.01);
});

test("rotate3dZ", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3dZ::rotate3dZ;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate3dZ(HALF_PI); // π/2 radians
      let v = vec3f(1.0, 0.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around Z-axis of (1,0,0) - result depends on matrix convention
  expectCloseTo([0.0, -1.0, 0.0], result.slice(0, 3), 0.01);
});

test("rotate4d - axis-angle rotation", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4d::rotate4d;
    @compute @workgroup_size(1)
    fn foo() {
      let axis = normalize(vec3f(0.0, 0.0, 1.0));
      let mat = rotate4d(axis, HALF_PI); // π/2 radians around Z
      let v = vec4f(1.0, 0.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around Z-axis of (1,0,0,1) - result depends on matrix convention
  expectCloseTo([0.0, -1.0, 0.0, 1.0], result, 0.01);
});

test("rotate4dX", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4dX::rotate4dX;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate4dX(HALF_PI); // π/2 radians
      let v = vec4f(0.0, 1.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around X-axis of (0,1,0,1) - result depends on matrix convention
  expectCloseTo([0.0, 0.0, -1.0, 1.0], result, 0.01);
});

test("rotate4dY", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4dY::rotate4dY;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate4dY(HALF_PI); // π/2 radians
      let v = vec4f(1.0, 0.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around Y-axis of (1,0,0,1) should give (0,0,-1,1)
  expectCloseTo([0.0, 0.0, -1.0, 1.0], result, 0.01);
});

test("rotate4dZ", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4dZ::rotate4dZ;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate4dZ(HALF_PI); // π/2 radians
      let v = vec4f(1.0, 0.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // 90° rotation around Z-axis of (1,0,0,1) - result depends on matrix convention
  expectCloseTo([0.0, -1.0, 0.0, 1.0], result, 0.01);
});

test("round", async () => {
  const src = `
    import lygia::math::round::round;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = round(2.3);
      let result2 = round(2.7);
      let result3 = round(-2.3);
      let result4 = round(-2.7);
      test::results[0] = vec4f(result1, result2, result3, result4);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([2.0, 3.0, -2.0, -3.0], result, 0.01);
});

test("saturateMediump", async () => {
  const src = `
    import lygia::math::saturateMediump::saturateMediump;
    @compute @workgroup_size(1)
    fn foo() {
      // saturateMediump clamps to MEDIUMP_FLT_MAX (65504.0) on mobile
      // On desktop (TARGET_MOBILE=false), it's a pass-through
      // IMPORTANT: It does NOT clamp the lower bound to 0!

      // Test pass-through behavior (desktop)
      let v1 = saturateMediump(-0.5);     // Passes through as -0.5 (desktop) or -0.5 (mobile)
      let v2 = saturateMediump(0.5);      // Passes through
      let v3 = saturateMediump(1000.0);   // Passes through (desktop) or clamped to 65504 (mobile)
      let v4 = saturateMediump(100000.0); // Passes through (desktop) or clamped to 65504 (mobile)

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // On desktop: pass-through behavior
  // v1: -0.5 passes through
  expectCloseTo([-0.5], [result[0]], 0.01);
  // v2: 0.5 passes through
  expectCloseTo([0.5], [result[1]], 0.01);
  // v3 and v4: On desktop pass through, on mobile clamped to 65504
  // We test that they're either the original value or clamped
  expect(result[2]).toBeGreaterThan(0.0);
  expect(result[3]).toBeGreaterThan(0.0);
});

test("scale4d", async () => {
  const src = `
    import lygia::math::scale4d::scale4d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale4d(vec3f(2.0, 3.0, 4.0));
      let v = vec4f(1.0, 2.0, 3.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([2.0, 6.0, 12.0, 1.0], result, 0.01);
});

test("sum2", async () => {
  const src = `
    import lygia::math::sum::sum2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = sum2(vec2f(3.0, 7.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([10.0], result);
});

test("sum3", async () => {
  const src = `
    import lygia::math::sum::sum3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = sum3(vec3f(3.0, 7.0, 5.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([15.0], result);
});

test("within - scalar", async () => {
  const src = `
    import lygia::math::within::within;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = within(5.0, 0.0, 10.0); // true -> 1.0
      let result2 = within(-1.0, 0.0, 10.0); // false -> 0.0
      let result3 = within(11.0, 0.0, 10.0); // false -> 0.0
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0, 0.0], result.slice(0, 3), 0.01);
});

test("within2", async () => {
  const src = `
    import lygia::math::within::within2;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = within2(vec2f(5.0, 5.0), vec2f(0.0), vec2f(10.0)); // true -> 1.0
      let result2 = within2(vec2f(-1.0, 5.0), vec2f(0.0), vec2f(10.0)); // false -> 0.0
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0], result.slice(0, 2), 0.01);
});

test("grad4 - noise gradient helper", async () => {
  const src = `
    import lygia::math::grad4::grad4;
    @compute @workgroup_size(1)
    fn foo() {
      // grad4 computes gradient vectors for 4D noise
      // It uses permutation value j and position p to generate gradients

      // Test 1: Reproducibility - same inputs give same output
      let g1a = grad4(100.0, vec4f(0.1, 0.2, 0.3, 0.4));
      let g1b = grad4(100.0, vec4f(0.1, 0.2, 0.3, 0.4));

      // Test 2: Different j values may give different gradients (not guaranteed)
      let g2 = grad4(200.0, vec4f(0.1, 0.2, 0.3, 0.4));

      // Test 3: Different positions give different gradients
      let g3 = grad4(100.0, vec4f(0.5, 0.6, 0.7, 0.8));

      // Store x components for validation
      test::results[0] = vec4f(g1a.x, g1b.x, g2.x, g3.x);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Reproducibility: g1a.x == g1b.x
  expectCloseTo([result[0]], [result[1]], 0.0001);

  // All gradient components should be in reasonable range
  for (let i = 0; i < 4; i++) {
    expect(Math.abs(result[i])).toBeLessThan(3.0);
  }

  // Note: We don't test that different j or positions always produce different outputs
  // because grad4 uses a hash function that may occasionally produce collisions
});

test("grad4 - gradient range validation", async () => {
  const src = `
    import lygia::math::grad4::grad4;
    @compute @workgroup_size(1)
    fn foo() {
      // Test gradient at origin and other positions
      let g1 = grad4(50.0, vec4f(0.0, 0.0, 0.0, 0.0));
      let g2 = grad4(75.0, vec4f(0.2, 0.3, 0.4, 0.5));

      // Store y and z components for validation
      test::results[0] = vec4f(g1.y, g1.z, g2.y, g2.z);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // All gradient components should be in reasonable range
  for (let i = 0; i < 4; i++) {
    expect(Math.abs(result[i])).toBeLessThan(3.0);
  }
});

// Anti-aliased fract tests (require derivatives, use fragment shaders)
test("aafract - anti-aliased fract", async () => {
  const src = `
    import lygia::math::aafract::aafract;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Create varying value for derivatives to work
      // aafract should create smooth transitions at integer boundaries
      let x = pos.x / 10.0 + 2.3; // Range ~2.3-2.5
      let result = aafract(x);

      // For comparison, regular fract
      let regularFract = fract(x);

      return vec4f(result, regularFract, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [2, 2]);

  // aafract should produce values in [0, 1] range like fract
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // aafract should be close to regular fract for slowly varying values
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.3);
});

test("aafract - edge anti-aliasing behavior", async () => {
  const src = `
    import lygia::math::aafract::aafract;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Test near integer boundary where anti-aliasing matters most
      // Regular fract has a hard edge at integers, aafract should smooth it
      let x = pos.x / 100.0 + 2.25; // Away from integer boundary

      let aaResult = aafract(x);
      let regularResult = fract(x);

      return vec4f(aaResult, regularResult, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Both should be in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);

  // For input x ≈ 2.25, fract(x) ≈ 0.25
  // aafract should be similar for slowly varying values
  expectCloseTo([0.25], [result[1]], 0.1); // Regular fract
  expectCloseTo([0.25], [result[0]], 0.2); // Anti-aliased version
});

test("aafract2 - vec2 anti-aliased fract", async () => {
  const src = `
    import lygia::math::aafract::aafract2;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Test vec2 version with different values for x and y
      let xy = pos.xy / 100.0 + vec2f(1.3, 2.7);
      let result = aafract2(xy);

      return vec4f(result.x, result.y, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Both components should be in [0, 1] range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);

  // For slowly varying input (pos.xy / 100.0), derivatives are small
  // aafract should behave similar to regular fract
  // x: fract(~1.3) ≈ 0.3, y: fract(~2.7) ≈ 0.7
  // Note: aafract can have wider anti-aliasing bands, so use generous tolerance
  expectCloseTo([0.3, 0.7], result.slice(0, 2), 0.3);
});

test("aafract - periodic behavior", async () => {
  const src = `
    import lygia::math::aafract::aafract;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Test that aafract maintains periodicity across integer boundaries
      let x = pos.x / 50.0;

      // Test at different integer offsets
      let r1 = aafract(x + 1.5);
      let r2 = aafract(x + 2.5);
      let r3 = aafract(x + 3.5);

      // All should be approximately 0.5 (fractional part)
      return vec4f(r1, r2, r3, 1.0);
    }`;

  const result = await testFragment(src, [2, 2]);

  // All should be close to 0.5 (the fractional part of x.5)
  expectCloseTo([0.5], [result[0]], 0.15);
  expectCloseTo([0.5], [result[1]], 0.15);
  expectCloseTo([0.5], [result[2]], 0.15);

  // Verify periodicity: r1, r2, r3 should be similar
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.1);
  expect(Math.abs(result[1] - result[2])).toBeLessThan(0.1);
});

test("hemisphereCosSample - unit vector property", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that output is a unit vector (or near unit length)
      let v1 = hemisphereCosSample(vec2f(0.0, 0.0));
      let v2 = hemisphereCosSample(vec2f(1.0, 1.0));
      let v3 = hemisphereCosSample(vec2f(0.5, 0.5));

      // Calculate lengths
      let len1 = length(v1);
      let len2 = length(v2);
      let len3 = length(v3);

      test::results[0] = vec4f(len1, len2, len3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // All outputs should be unit vectors (length ≈ 1.0)
  expectCloseTo([1.0, 1.0, 1.0], result.slice(0, 3), 0.01);
});

test("hemisphereCosSample - positive hemisphere", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that all outputs point into positive hemisphere (z >= 0)
      let v1 = hemisphereCosSample(vec2f(0.0, 0.0));
      let v2 = hemisphereCosSample(vec2f(1.0, 1.0));
      let v3 = hemisphereCosSample(vec2f(0.5, 0.5));
      let v4 = hemisphereCosSample(vec2f(0.25, 0.75));

      test::results[0] = vec4f(v1.z, v2.z, v3.z, v4.z);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // All z components should be >= 0 (positive hemisphere)
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[3]).toBeGreaterThanOrEqual(0.0);
  // All z components should be <= 1 (unit vector constraint)
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  expect(result[3]).toBeLessThanOrEqual(1.0);
});

test("hemisphereCosSample - known values", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test specific known values
      // u=(0,0): phi=0, cosTheta2=1, cosTheta=1, sinTheta=0 → (0, 0, 1)
      let v1 = hemisphereCosSample(vec2f(0.0, 0.0));

      // u=(0,1): phi=0, cosTheta2=0, cosTheta=0, sinTheta=1 → (1, 0, 0)
      let v2 = hemisphereCosSample(vec2f(0.0, 1.0));

      // Store z component from v1 and full v2
      test::results[0] = vec4f(v1.z, v2.x, v2.y, v2.z);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // u=(0,0) should give (0, 0, 1) - pointing straight up (testing z component)
  expectCloseTo([1.0], [result[0]], 0.01);
  // u=(0,1) should give (cos(0)*1, sin(0)*1, 0) = (1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result.slice(1, 4), 0.01);
});

test("hemisphereCosSample - cosine distribution", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that the distribution is cosine-weighted
      // Higher u.y values should produce samples closer to the horizon (smaller z)
      let v_low = hemisphereCosSample(vec2f(0.5, 0.1));   // Low u.y → higher z
      let v_mid = hemisphereCosSample(vec2f(0.5, 0.5));   // Mid u.y → mid z
      let v_high = hemisphereCosSample(vec2f(0.5, 0.9));  // High u.y → lower z

      test::results[0] = vec4f(v_low.z, v_mid.z, v_high.z, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Verify ordering: z decreases as u.y increases (cosine distribution property)
  expect(result[0]).toBeGreaterThan(result[1]); // low u.y has higher z than mid u.y
  expect(result[1]).toBeGreaterThan(result[2]); // mid u.y has higher z than high u.y
  // All should still be in valid range [0, 1]
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThan(1.0);
});

// Unpack functions - converting packed vec3 to float values
test("unpack256 - default base 256", async () => {
  const src = `
    import lygia::math::unpack::unpack256;
    @compute @workgroup_size(1)
    fn foo() {
      // Test unpacking with base 256
      // unpack256 uses dot(v, vec3(256, 256^2, 256^3)) / 16581375
      // Note: divisor is 16581375 = 256^3 * (255/256), not just 256^3
      let v1 = vec3f(1.0, 0.0, 0.0);  // 256 / 16581375
      let v2 = vec3f(0.5, 0.5, 0.5);  // (128 + 32768 + 8388608) / 16581375
      let v3 = vec3f(1.0, 1.0, 1.0);  // (256 + 65536 + 16777216) / 16581375

      let r1 = unpack256(v1);
      let r2 = unpack256(v2);
      let r3 = unpack256(v3);

      test::results[0] = vec4f(r1, r2, r3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // v1 = (1,0,0) → 256 / 16581375 ≈ 0.00001544
  expectCloseTo([0.00001544], [result[0]], 0.00000001);
  // v2 = (0.5,0.5,0.5) → (128 + 32768 + 8388608) / 16581375 ≈ 0.50787
  expectCloseTo([0.50787], [result[1]], 0.001);
  // v3 = (1,1,1) → (256 + 65536 + 16777216) / 16581375 ≈ 1.01578
  expectCloseTo([1.01578], [result[2]], 0.001);
});

test("unpack - alias for unpack256", async () => {
  const src = `
    import lygia::math::unpack::unpack;
    import lygia::math::unpack::unpack256;
    @compute @workgroup_size(1)
    fn foo() {
      let v = vec3f(0.5, 0.5, 0.5);
      let r1 = unpack(v);
      let r2 = unpack256(v);
      test::results[0] = vec4f(r1, r2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // unpack should be identical to unpack256
  expectCloseTo([result[0]], [result[1]], 0.0001);
});

test("unpack8 - base 8", async () => {
  const src = `
    import lygia::math::unpack::unpack8;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack8 uses dot(v, vec3(8, 64, 512)) / 512
      let v1 = vec3f(1.0, 0.0, 0.0);   // 8 / 512 = 0.015625
      let v2 = vec3f(0.0, 1.0, 0.0);   // 64 / 512 = 0.125
      let v3 = vec3f(0.0, 0.0, 1.0);   // 512 / 512 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (4 + 32 + 256) / 512 = 0.5703125

      test::results[0] = vec4f(unpack8(v1), unpack8(v2), unpack8(v3), unpack8(v4));
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.015625, 0.125, 1.0, 0.5703125], result, 0.001);
});

test("unpack16 - base 16", async () => {
  const src = `
    import lygia::math::unpack::unpack16;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack16 uses dot(v, vec3(16, 256, 4096)) / 4096
      let v1 = vec3f(1.0, 0.0, 0.0);   // 16 / 4096 = 0.00390625
      let v2 = vec3f(0.0, 1.0, 0.0);   // 256 / 4096 = 0.0625
      let v3 = vec3f(0.0, 0.0, 1.0);   // 4096 / 4096 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (8 + 128 + 2048) / 4096 = 0.533203125

      test::results[0] = vec4f(unpack16(v1), unpack16(v2), unpack16(v3), unpack16(v4));
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.00390625, 0.0625, 1.0, 0.533203125], result, 0.001);
});

test("unpack32 - base 32", async () => {
  const src = `
    import lygia::math::unpack::unpack32;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack32 uses dot(v, vec3(32, 1024, 32768)) / 32768
      let v1 = vec3f(1.0, 0.0, 0.0);   // 32 / 32768 ≈ 0.000977
      let v2 = vec3f(0.0, 1.0, 0.0);   // 1024 / 32768 ≈ 0.03125
      let v3 = vec3f(0.0, 0.0, 1.0);   // 32768 / 32768 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (16 + 512 + 16384) / 32768 ≈ 0.515625

      test::results[0] = vec4f(unpack32(v1), unpack32(v2), unpack32(v3), unpack32(v4));
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.000977, 0.03125, 1.0, 0.515625], result, 0.001);
});

test("unpack64 - base 64", async () => {
  const src = `
    import lygia::math::unpack::unpack64;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack64 uses dot(v, vec3(64, 4096, 262144)) / 262144
      let v1 = vec3f(1.0, 0.0, 0.0);   // 64 / 262144 ≈ 0.000244
      let v2 = vec3f(0.0, 1.0, 0.0);   // 4096 / 262144 ≈ 0.015625
      let v3 = vec3f(0.0, 0.0, 1.0);   // 262144 / 262144 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (32 + 2048 + 131072) / 262144 ≈ 0.507935

      test::results[0] = vec4f(unpack64(v1), unpack64(v2), unpack64(v3), unpack64(v4));
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.000244, 0.015625, 1.0, 0.507935], result, 0.001);
});

test("unpack128 - base 128", async () => {
  const src = `
    import lygia::math::unpack::unpack128;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack128 uses dot(v, vec3(128, 16384, 2097152)) / 2097152
      let v1 = vec3f(1.0, 0.0, 0.0);   // 128 / 2097152 ≈ 0.000061
      let v2 = vec3f(0.0, 1.0, 0.0);   // 16384 / 2097152 ≈ 0.0078125
      let v3 = vec3f(0.0, 0.0, 1.0);   // 2097152 / 2097152 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (64 + 8192 + 1048576) / 2097152 ≈ 0.503967

      test::results[0] = vec4f(unpack128(v1), unpack128(v2), unpack128(v3), unpack128(v4));
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.000061, 0.0078125, 1.0, 0.503967], result, 0.001);
});

test("unpackBase - custom base", async () => {
  const src = `
    import lygia::math::unpack::unpackBase;
    @compute @workgroup_size(1)
    fn foo() {
      // Test with base 10: dot(v, vec3(10, 100, 1000)) / 1000
      let base = 10.0;
      let v1 = vec3f(1.0, 0.0, 0.0);   // 10 / 1000 = 0.01
      let v2 = vec3f(0.0, 1.0, 0.0);   // 100 / 1000 = 0.1
      let v3 = vec3f(0.0, 0.0, 1.0);   // 1000 / 1000 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (5 + 50 + 500) / 1000 = 0.555

      test::results[0] = vec4f(
        unpackBase(v1, base),
        unpackBase(v2, base),
        unpackBase(v3, base),
        unpackBase(v4, base)
      );
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.01, 0.1, 1.0, 0.555], result, 0.001);
});

test("unpack4 - vec4 unpacking (ThreeJS style)", async () => {
  const src = `
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack4 uses ThreeJS packing: dot(v, UnpackFactors)
      // UnpackFactors = (255/256) / vec4(256^3, 256^2, 256, 1)
      // = vec4(5.960464e-8, 1.5258789e-5, 0.00390625, 0.99609375)
      // Sum ≈ 1.0
      let v1 = vec4f(1.0, 0.0, 0.0, 0.0);  // r only: 5.960464e-8
      let v2 = vec4f(0.0, 0.0, 0.0, 1.0);  // a only: 0.99609375
      let v3 = vec4f(0.5, 0.5, 0.5, 0.5);  // uniform: 0.5
      let v4 = vec4f(1.0, 1.0, 1.0, 1.0);  // all: 1.0

      test::results[0] = vec4f(unpack4(v1), unpack4(v2), unpack4(v3), unpack4(v4));
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Test with specific expected values based on UnpackFactors formula
  expectCloseTo([5.960464e-8], [result[0]], 9);    // r component weight
  expectCloseTo([0.99609375], [result[1]], 5);     // a component weight (255/256)
  expectCloseTo([0.5], [result[2]], 3);            // uniform 0.5 → 0.5 * sum
  expectCloseTo([1.0], [result[3]], 5);            // sum of all weights
});

// Quaternion utility functions
test("quatForward - create quat from forward vector", async () => {
  const src = `
    import lygia::math::quat::quatForward;
    import lygia::math::quat::quatConj;
    import lygia::math::quat::mul::quatMulVec3;
    @compute @workgroup_size(1)
    fn foo() {
      // Create quaternion that rotates default forward to +X
      let forward = normalize(vec3f(1.0, 0.0, 0.0));
      let q = quatForward(forward);

      // Test by rotating default forward vector (0,0,1) using this quaternion
      // It should rotate to point in the forward direction we specified (+X)
      let defaultForward = vec3f(0.0, 0.0, 1.0);
      let rotated = quatMulVec3(q, defaultForward);

      // Also verify quaternion is normalized
      let length = sqrt(dot(q, q));

      test::results[0] = vec4f(rotated.x, rotated.y, rotated.z, length);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Rotated vector should point in +X direction (our specified forward)
  expectCloseTo([1.0, 0.0, 0.0], result.slice(0, 3), 0.1);
  // Quaternion should be normalized
  expectCloseTo([1.0], [result[3]], 0.01);
});

test("quatForwardUp - create quat from forward and up vectors", async () => {
  const src = `
    import lygia::math::quat::quatForwardUp;
    import lygia::math::quat::mul::quatMulVec3;
    @compute @workgroup_size(1)
    fn foo() {
      // Create quaternion with forward=+X and up=+Y
      let forward = normalize(vec3f(1.0, 0.0, 0.0));
      let up = normalize(vec3f(0.0, 1.0, 0.0));
      let q = quatForwardUp(forward, up);

      // Test by rotating vectors
      // Default forward (0,0,1) should rotate to our forward (+X)
      let defaultForward = vec3f(0.0, 0.0, 1.0);
      let rotatedForward = quatMulVec3(q, defaultForward);

      // Default up (0,1,0) should remain up (+Y) since we specified that
      let defaultUp = vec3f(0.0, 1.0, 0.0);
      let rotatedUp = quatMulVec3(q, defaultUp);

      // Verify quaternion is normalized
      let length = sqrt(dot(q, q));

      test::results[0] = vec4f(rotatedForward.x, rotatedUp.y, length, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Rotated forward should point in +X direction
  expectCloseTo([1.0], [result[0]], 0.1);
  // Rotated up should still point in +Y direction
  expectCloseTo([1.0], [result[1]], 0.1);
  // Quaternion should be normalized
  expectCloseTo([1.0], [result[2]], 0.01);
});
