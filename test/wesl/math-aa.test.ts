import { expect, test } from "vitest";
import { expectCloseTo, testFragment } from "./testUtil.ts";

// Anti-aliased functions requiring derivatives (use fragment shaders)

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

  const result = await testFragment(src, { size: [2, 2] });
  // aafloor should produce something close to 2.0
  expectCloseTo([2.0], [result[0]]);
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

  const result = await testFragment(src, { size: [2, 2] });

  expectCloseTo([2.0, 3.0], result.slice(0, 2));
});

test("aamirror - anti-aliased triangle wave", async () => {
  const src = `
    import lygia::math::aamirror::aamirror;
    import lygia::math::mirror::mirror;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Test triangle wave pattern at different positions
      // aamirror should create: 0�1�0�1�0 (triangle wave)
      let x = pos.x / 50.0;  // Slow variation for derivatives

      // Test at valley (x=0), peak (x=0.5), valley (x=1.0), peak (x=1.5)
      let valley1_aa = aamirror(x);
      let peak1_aa = aamirror(x + 0.5);
      let valley2_aa = aamirror(x + 1.0);
      let peak2_aa = aamirror(x + 1.5);

      return vec4f(valley1_aa, peak1_aa, valley2_aa, peak2_aa);
    }`;

  const result = await testFragment(src, { size: [2, 2] });

  // Verify triangle wave pattern: valleys near 0.0, peaks near 1.0
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThan(0.2); // Valley

  expect(result[1]).toBeGreaterThan(0.8);
  expect(result[1]).toBeLessThanOrEqual(1.0); // Peak

  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThan(0.2); // Valley (periodic)

  expect(result[3]).toBeGreaterThan(0.8);
  expect(result[3]).toBeLessThanOrEqual(1.0); // Peak (periodic)
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

  const result = await testFragment(src, { size: [2, 2] });

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
      let at0 = fcos(x);                   // cos(~0) H 1.0
      let atPi4 = fcos(x + PI * 0.25);     // cos(�/4) H INV_SQRT2
      let atPi2 = fcos(x + PI * 0.5);      // cos(�/2) H 0.0
      let atPi = fcos(x + PI);             // cos(�) H -1.0

      return vec4f(at0, atPi4, atPi2, atPi);
    }`;

  const result = await testFragment(src, { size: [2, 2] });

  // Verify cosine values
  expectCloseTo([1.0], [result[0]]); // cos(0) = 1.0
  // Loose precision: filtered cosine uses derivatives, introduces small error (~0.0003)
  expectCloseTo([Math.SQRT1_2], [result[1]], 0.001); // cos(�/4) = 2/2
  // Loose precision: filtered cosine uses derivatives, small error near zero (~0.0005)
  expectCloseTo([0.0], [result[2]], 0.001); // cos(�/2) = 0.0
  expectCloseTo([-1.0], [result[3]]); // cos(�) = -1.0
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

  const result = await testFragment(src, { size: [32, 32] });

  // High frequency should be heavily attenuated (close to 0)
  expect(Math.abs(result[0])).toBeLessThan(0.01);

  // Low frequency should have full amplitude (not attenuated)
  expect(Math.abs(result[1])).toBeGreaterThan(0.95);

  // Exact value check to catch regressions
  expectCloseTo([0.0, 0.9998], result.slice(0, 2));
});

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

  const result = await testFragment(src, { size: [2, 2] });

  // aafract should be close to regular fract for slowly varying values
  // Loose precision: anti-aliasing adds smoothing near integer boundaries
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.15);

  // Exact value checks to catch regressions
  expectCloseTo([0.4375, 0.35], result.slice(0, 2));
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

  const result = await testFragment(src, { size: [2, 2] });

  // For input x H 2.25, fract(x) H 0.25
  // aafract should be similar for slowly varying values
  // Loose precision: fragment shader position varies slightly per pixel (~0.005)
  expectCloseTo([0.25], [result[1]], 0.01); // Regular fract
  // Loose precision: anti-aliasing smooths transitions, may vary from exact value
  expectCloseTo([0.25], [result[0]], 0.2);
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

  const result = await testFragment(src, { size: [2, 2] });

  // For slowly varying input (pos.xy / 100.0), derivatives are small
  // aafract should behave similar to regular fract
  // x: fract(~1.3) H 0.3, y: fract(~2.7) H 0.7
  // Loose precision: anti-aliasing can create wider transition bands
  expectCloseTo([0.3, 0.7], result.slice(0, 2), 0.05);

  // Exact value check to catch regressions
  expectCloseTo([0.3112, 0.7194], result.slice(0, 2));
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

  const result = await testFragment(src, { size: [2, 2] });

  // Property checks: all should be close to 0.5 (the fractional part of x.5)
  // Loose precision: anti-aliasing smoothing varies with derivative magnitude
  expectCloseTo([0.5], [result[0]], 0.05);
  expectCloseTo([0.5], [result[1]], 0.05);
  expectCloseTo([0.5], [result[2]], 0.05);

  // Verify periodicity: r1, r2, r3 should be identical
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.01);
  expect(Math.abs(result[1] - result[2])).toBeLessThan(0.01);

  // Exact value check to catch regressions
  expectCloseTo([0.5312, 0.5312, 0.5312], result.slice(0, 3));
});
