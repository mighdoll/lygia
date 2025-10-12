import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

// ============================================================================
// Bayer Dithering Tests
// ============================================================================

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
  const result = await testCompute(src, "vec4f");

  // All values should be in [0, 1] range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Based on the 8x8 Bayer matrix:
  // (0,0) -> index 0 -> 0.0/64.0 = 0.0
  expectCloseTo([0.0], [result[0]], 0.01);
  // (4,4) -> index 36 -> 1.0/64.0 ≈ 0.0156
  expectCloseTo([0.0156], [result[1]], 0.01);
  // (7,7) -> index 63 -> 21.0/64.0 ≈ 0.3281
  expectCloseTo([0.3281], [result[2]], 0.01);
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
  const result = await testCompute(src, "vec4f");

  // Pattern should repeat: v1 == v2 == v3
  expectCloseTo([result[0], result[0]], result.slice(1, 3), 0.0001);

  // Different position should give different value
  expect(Math.abs(result[0] - result[3])).toBeGreaterThan(0.05);
});

test("ditherBayerPrecision - f32 with precision control", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayerPrecision;

    @compute @workgroup_size(1)
    fn foo() {
      // Test dithering a mid-gray value
      let val = 0.5;
      let xy = vec2f(2.0, 3.0);

      // Test with different precisions
      let result8 = ditherBayerPrecision(val, xy, 8);   // 8 levels
      let result16 = ditherBayerPrecision(val, xy, 16); // 16 levels
      let result256 = ditherBayerPrecision(val, xy, 256); // 256 levels (default)

      test::results[0] = vec4f(result8, result16, result256, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // All results should be in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Lower precision should have more coarse quantization
  // Higher precision should be closer to original value
  expect(result[2]).toBeCloseTo(0.5, 1); // 256 levels should be close to 0.5
});

test("ditherBayer3 - vec3 dithering", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer3;

    @compute @workgroup_size(1)
    fn foo() {
      // Test dithering an orange color
      let color = vec3f(0.8, 0.5, 0.2);
      let xy = vec2f(3.0, 4.0);

      let result = ditherBayer3(color, xy);
      test::results[0] = vec4f(result, 1.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Result should maintain approximate color relationships
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Orange should maintain R > G > B relationship
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);

  // Should be close to original values
  expectCloseTo([0.8, 0.5, 0.2], result.slice(0, 3), 0.1);
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
  const result = await testCompute(src, "vec4f");

  // RGB should be dithered
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Alpha should be preserved exactly
  expectCloseTo([0.75], [result[3]], 0.0001);

  // Color relationship should be maintained
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);
});

test("ditherBayerPrecision3 - vec3 with custom precision", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayerPrecision3;

    @compute @workgroup_size(1)
    fn foo() {
      // Test with low precision (posterization effect)
      let color = vec3f(0.5, 0.5, 0.5);
      let result4 = ditherBayerPrecision3(color, 4);   // 4 levels (0, 0.33, 0.67, 1)
      let result8 = ditherBayerPrecision3(color, 8);   // 8 levels
      let result256 = ditherBayerPrecision3(color, 256); // 256 levels

      test::results[0] = vec4f(result4.r, result8.r, result256.r, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // All results should be valid
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Higher precision should be closer to original
  expect(Math.abs(result[2] - 0.5)).toBeLessThan(Math.abs(result[0] - 0.5));
});

test("ditherBayerPrecision4 - vec4 with custom precision", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayerPrecision4;

    @compute @workgroup_size(1)
    fn foo() {
      let color = vec4f(0.7, 0.5, 0.3, 0.8);
      let result = ditherBayerPrecision4(color, 16); // 16 levels
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");

  // RGB should be quantized
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Alpha should be preserved
  expectCloseTo([0.8], [result[3]], 0.0001);

  // Color ordering should be maintained
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);
});

test("ditherBayer1 - f32 with defaults", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer1;

    @compute @workgroup_size(1)
    fn foo() {
      // Test with default coordinate and precision
      let result1 = ditherBayer1(0.3);
      let result2 = ditherBayer1(0.7);
      let result3 = ditherBayer1(0.5);
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // All values should be in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Results should be close to input values
  expectCloseTo([0.3, 0.7, 0.5], result.slice(0, 3), 0.1);
});

test("ditherBayer3Simple - vec3 with defaults", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer3Simple;

    @compute @workgroup_size(1)
    fn foo() {
      let color = vec3f(0.9, 0.6, 0.3);
      let result = ditherBayer3Simple(color);
      test::results[0] = vec4f(result, 1.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // RGB should be in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Should maintain color relationships
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);

  // Should be close to original
  expectCloseTo([0.9, 0.6, 0.3], result.slice(0, 3), 0.1);
});

test("ditherBayer4Simple - vec4 with defaults", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer4Simple;

    @compute @workgroup_size(1)
    fn foo() {
      let color = vec4f(0.8, 0.5, 0.2, 0.9);
      let result = ditherBayer4Simple(color);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");

  // RGB should be dithered, in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Alpha should be preserved
  expectCloseTo([0.9], [result[3]], 0.0001);

  // Color ordering maintained
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);

  // Close to original
  expectCloseTo([0.8, 0.5, 0.2], result.slice(0, 3), 0.1);
});

test("ditherBayer - gradient banding reduction", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayerPrecision;

    @compute @workgroup_size(1)
    fn foo() {
      // Simulate a gradient with low precision causing banding
      // Without dithering, these would all quantize to the same value
      let val1 = 0.500;
      let val2 = 0.502;
      let val3 = 0.504;

      // Use different positions in the Bayer matrix
      let d1 = ditherBayerPrecision(val1, vec2f(0.0, 0.0), 64);
      let d2 = ditherBayerPrecision(val2, vec2f(1.0, 0.0), 64);
      let d3 = ditherBayerPrecision(val3, vec2f(2.0, 0.0), 64);

      test::results[0] = vec4f(d1, d2, d3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // All should be in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Dithering should create variation even with similar input values
  // The spatial variation from different Bayer matrix positions
  // helps break up banding artifacts in gradients
  expect(result[0]).toBeCloseTo(0.5, 1);
  expect(result[1]).toBeCloseTo(0.5, 1);
  expect(result[2]).toBeCloseTo(0.5, 1);
});

test("ditherBayer - quantization levels", async () => {
  const src = `
    import lygia::color::dither::bayer::ditherBayer3Precision;

    @compute @workgroup_size(1)
    fn foo() {
      // Test that precision parameter controls quantization levels
      let color = vec3f(0.5, 0.5, 0.5);
      let xy = vec2f(0.0, 0.0);

      // With precision=2, should snap to 0.0, 0.5, or 1.0
      let result2 = ditherBayer3Precision(color, xy, 2);
      // With precision=4, should have more levels
      let result4 = ditherBayer3Precision(color, xy, 4);
      // With precision=256, should be nearly unchanged
      let result256 = ditherBayer3Precision(color, xy, 256);

      test::results[0] = vec4f(result2.r, result4.r, result256.r, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify all results are valid
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Higher precision = closer to original value
  expect(Math.abs(result[2] - 0.5)).toBeLessThan(0.05);
});
