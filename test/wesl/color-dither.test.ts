import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("ditherBayer - base function returns values in [0,1]", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer;

    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple positions in the 8x8 Bayer matrix
      let v1 = ditherBayer(vec2f(0.0, 0.0)); // Top-left corner
      let v2 = ditherBayer(vec2f(4.0, 4.0)); // Middle
      let v3 = ditherBayer(vec2f(7.0, 7.0)); // Bottom-right corner
      test::results[0] = vec4f(v1, v2, v3, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Based on the 8x8 Bayer matrix:
  // (0,0) -> index 0 -> 0.0/64.0 = 0.0
  expectCloseTo([0.0], [result[0]]);
  // (4,4) -> index 36 -> 1.0/64.0 ≈ 0.0156
  expectCloseTo([0.0156], [result[1]]);
  // (7,7) -> index 63 -> 21.0/64.0 ≈ 0.3281
  expectCloseTo([0.3281], [result[2]]);
});

test("ditherBayer - 8x8 pattern verification", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer;

    @compute @workgroup_size(1)
    fn foo() {
      // Test that the pattern repeats every 8 pixels
      let v1 = ditherBayer(vec2f(1.0, 1.0));
      let v2 = ditherBayer(vec2f(9.0, 9.0));  // Should be same as (1,1)
      let v3 = ditherBayer(vec2f(17.0, 17.0)); // Should be same as (1,1)

      // Test different positions give different values
      let v4 = ditherBayer(vec2f(2.0, 2.0));

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Pattern should repeat: v1 == v2 == v3
  expectCloseTo([result[0], result[0]], result.slice(1, 3));

  // Different position should give different value
  expect(Math.abs(result[0] - result[3])).toBeGreaterThan(0.05);
});

test("ditherBayerPrecision - f32 with precision control", async () => {
  const src = `
    import lygia::color::dither::bayer::{ditherBayer, ditherBayerPrecision};

    @compute @workgroup_size(1)
    fn foo() {
      // Test quantization formula: floor(value * precision + bayer) / precision
      let val = 0.5;
      let xy = vec2f(2.0, 3.0);
      let bayer = ditherBayer(xy);

      // With precision=4, value should snap to 0.0, 0.25, 0.5, 0.75, or 1.0
      let result4 = ditherBayerPrecision(val, xy, 4);
      // Expected: floor(0.5 * 4 + bayer) / 4 = floor(2.0 + bayer) / 4

      // With precision=8, finer quantization
      let result8 = ditherBayerPrecision(val, xy, 8);

      // Test another value that will quantize differently
      let val2 = 0.3;
      let result4_v2 = ditherBayerPrecision(val2, xy, 4);

      test::results[0] = vec4f(result4, result8, result4_v2, bayer);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // With precision=4, should be quantized to 0.25 increments
  expect(result[0] % 0.25).toBeCloseTo(0.0, 2);

  // With precision=8, should be quantized to 0.125 increments
  expect(result[1] % 0.125).toBeCloseTo(0.0, 2);

  // Different input value should produce different quantized output
  expect(Math.abs(result[0] - result[2])).toBeGreaterThan(0.1);

  // Results should be close to original values
  expect(result[0]).toBeCloseTo(0.5, 1);
  expect(result[2]).toBeCloseTo(0.3, 1);
});

test("ditherBayer3 - vec3 dithering", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer3;
    import lygia::color::dither::bayer::ditherBayer;

    @compute @workgroup_size(1)
    fn foo() {
      // Test that each channel is quantized independently
      let color = vec3f(0.8, 0.5, 0.2);
      let xy = vec2f(3.0, 4.0);

      let result = ditherBayer3(color, xy);

      // Test with gray to verify consistent dithering
      let gray = vec3f(0.5, 0.5, 0.5);
      let result_gray = ditherBayer3(gray, xy);

      test::results[0] = vec4f(result, 1.0);
      test::results[1] = vec4f(result_gray, 1.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f", size: 2 });

  // Orange should maintain R > G > B relationship after quantization
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);

  // Gray should have all channels equal (same dither threshold applied to all)
  expectCloseTo([result[4], result[4], result[4]], result.slice(4, 7));

  // Should be close to original values (coarse precision for dithered output)
  expectCloseTo([0.8, 0.5, 0.2], result.slice(0, 3), 0.1);
  expectCloseTo([0.5], [result[4]]);
});

test("ditherBayer4 - vec4 dithering preserves alpha", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer4;

    @compute @workgroup_size(1)
    fn foo() {
      // Test dithering with alpha channel
      let color = vec4f(0.6, 0.4, 0.2, 0.75);
      let xy = vec2f(5.0, 2.0);

      let result = ditherBayer4(color, xy);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Alpha should be preserved exactly
  expectCloseTo([0.75], [result[3]]);

  // Color relationship should be maintained
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);
});

test("ditherBayer - gradient banding reduction", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayerPrecision;

    @compute @workgroup_size(1)
    fn foo() {
      // Test that gradient values near a quantization boundary
      // get distributed across multiple quantized levels based on position
      let val = 0.5; // Right at midpoint

      // Sample 8 adjacent pixels - should show spatial distribution
      let d0 = ditherBayerPrecision(val, vec2f(0.0, 0.0), 16);
      let d1 = ditherBayerPrecision(val, vec2f(1.0, 0.0), 16);
      let d2 = ditherBayerPrecision(val, vec2f(2.0, 0.0), 16);
      let d3 = ditherBayerPrecision(val, vec2f(3.0, 0.0), 16);

      test::results[0] = vec4f(d0, d1, d2, d3);
      test::results[1] = vec4f(
        ditherBayerPrecision(val, vec2f(4.0, 0.0), 16),
        ditherBayerPrecision(val, vec2f(5.0, 0.0), 16),
        ditherBayerPrecision(val, vec2f(6.0, 0.0), 16),
        ditherBayerPrecision(val, vec2f(7.0, 0.0), 16)
      );
    }
  `;
  const result = await testCompute(src, { elem: "vec4f", size: 2 });

  // With precision=16, values should be quantized to 1/16 = 0.0625 increments
  // All values should snap to valid quantization levels
  for (let i = 0; i < 8; i++) {
    expect(result[i] % 0.0625).toBeCloseTo(0.0, 2);
  }

  // Not all values should be identical - spatial variation breaks up banding
  const uniqueValues = new Set(
    result.slice(0, 8).map((v) => Math.round(v * 16)),
  );
  expect(uniqueValues.size).toBeGreaterThan(1);

  // All values should be close to 0.5 (the input)
  result.slice(0, 8).forEach((v) => {
    expect(Math.abs(v - 0.5)).toBeLessThan(0.2);
  });
});

test("ditherBayer - quantization levels", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer3Precision;

    @compute @workgroup_size(1)
    fn foo() {
      // Test that precision parameter controls quantization levels
      let color = vec3f(0.5, 0.5, 0.5);
      let xy = vec2f(0.0, 0.0);

      // With precision=2, only 2 levels: 0.0 or 1.0 (step of 1.0)
      let result2 = ditherBayer3Precision(color, xy, 2);

      // With precision=4, levels: 0.0, 0.25, 0.5, 0.75, 1.0 (step of 0.25)
      let result4 = ditherBayer3Precision(color, xy, 4);

      // Test a darker value to see different quantization behavior
      let dark = vec3f(0.2, 0.2, 0.2);
      let dark2 = ditherBayer3Precision(dark, xy, 4);
      let dark256 = ditherBayer3Precision(dark, xy, 256);

      test::results[0] = vec4f(result2.r, result4.r, dark2.r, dark256.r);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Precision=2: should be either 0.0 or 1.0
  expect(result[0] === 0.0 || result[0] === 1.0).toBe(true);

  // Precision=4: should be multiple of 0.25
  expect(result[1] % 0.25).toBeCloseTo(0.0, 2);

  // Dark value with precision=4 should also be multiple of 0.25
  expect(result[2] % 0.25).toBeCloseTo(0.0, 2);

  // Higher precision should be closer to original
  expect(Math.abs(result[3] - 0.2)).toBeLessThan(0.02);

  // Lower precision has coarser steps
  expect(Math.abs(result[2] - 0.2)).toBeLessThan(0.15);
});
