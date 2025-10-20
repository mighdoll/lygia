import { expect, test } from "vitest";
import { expectCloseTo, expectDistribution, testCompute, testDistribution } from "./testUtil.ts";

/**
 * Generative Functions Test Suite
 *
 * This test suite uses a hybrid testing approach with three layers:
 *
 * 1. Property Tests (✅ Implemented)
 *    - Determinism: Same input produces same output
 *    - Continuity: Nearby points produce similar values
 *    - Periodicity: Periodic noise repeats correctly
 *    - Derivatives: Analytical derivatives match numerical derivatives
 *    - Hash properties: Avalanche effect, component independence
 *    - Mathematical properties: F1 ≤ F2 for Worley noise, tiling behavior
 *
 * 2. Range Tests (✅ Implemented)
 *    - Verify outputs are within expected bounds
 *    - Different functions have different ranges:
 *      - noise functions: [-1, 1]
 *      - random functions: [0, 1]
 *      - srandom functions: [-1, 1]
 *      - worley distances: [0, ~1.5] or [0, ~2.0]
 *
 * 3. GLSL Parity Tests (⏳ TODO - requires GLSL reference values)
 *    - Verify WESL outputs match GLSL outputs for specific inputs
 *    - Acts as regression test for conversion accuracy
 *    - Catches systematic biases or implementation bugs
 *
 *    To add GLSL parity tests:
 *    a) Run GLSL version with test inputs (e.g., vec2f(1.0, 2.0))
 *    b) Record exact output values
 *    c) Add specific value tests like:
 *       expectCloseTo([expected_glsl_value], [wesl_result], 3);
 *
 *    Priority order for adding parity tests:
 *    1. cnoise (classic noise - most commonly used)
 *    2. snoise (simplex noise - also very common)
 *    3. random (pseudo-random - frequently used)
 *    4. worley (cellular noise - specialty use)
 *    5. wavelet (wavelet noise - specialty use)
 */

test("cnoise2", async () => {
  const src = `
     import lygia::generative::cnoise::cnoise2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(1.01, 2.01); // Nearby point

       let n1 = cnoise2(p1);
       let n2 = cnoise2(p2);
       let n3 = cnoise2(p3);

       test::results[0] = vec4f(n1, n2, n3, abs(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test continuity: nearby points have similar values
  expect(result[3]).toBeLessThan(0.1); // difference should be small
  // Classic noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // TODO: Add GLSL parity test - verify result[0] matches GLSL cnoise2(vec2(1.0, 2.0))
  // Regression: exact output value
  expectCloseTo([0], [result[0]]);
});

test("cnoise3", async () => {
  const src = `
     import lygia::generative::cnoise::cnoise3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(1.01, 2.01, 3.01); // Nearby point

       let n1 = cnoise3(p1);
       let n2 = cnoise3(p2);
       let n3 = cnoise3(p3);

       test::results[0] = vec4f(n1, n2, n3, abs(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test continuity: nearby points have similar values
  expect(result[3]).toBeLessThan(0.1);
  // Classic noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0], [result[0]]);
});

test("cnoise4", async () => {
  const src = `
     import lygia::generative::cnoise::cnoise4;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0); // Same point
       let p3 = vec4f(1.01, 2.01, 3.01, 4.01); // Nearby point

       let n1 = cnoise4(p1);
       let n2 = cnoise4(p2);
       let n3 = cnoise4(p3);

       test::results[0] = vec4f(n1, n2, n3, abs(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test continuity: nearby points have similar values
  expect(result[3]).toBeLessThan(0.1);
  // Classic noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0], [result[0]]);
});

test("snoise2", async () => {
  const src = `
     import lygia::generative::snoise::snoise2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(1.01, 2.01); // Nearby point

       let n1 = snoise2(p1);
       let n2 = snoise2(p2);
       let n3 = snoise2(p3);

       test::results[0] = vec4f(n1, n2, n3, abs(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test continuity: nearby points have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Simplex noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // TODO: Add GLSL parity test - verify result[0] matches GLSL snoise2(vec2(1.0, 2.0))
  // Regression: exact output value
  expectCloseTo([0.36833828687667847], [result[0]]);
});

test("snoise3", async () => {
  const src = `
     import lygia::generative::snoise::snoise3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(1.01, 2.01, 3.01); // Nearby point

       let n1 = snoise3(p1);
       let n2 = snoise3(p2);
       let n3 = snoise3(p3);

       test::results[0] = vec4f(n1, n2, n3, abs(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test continuity: nearby points have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Simplex noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.7335153818130493], [result[0]]);
});

test("pnoise2", async () => {
  const src = `
     import lygia::generative::pnoise::pnoise2;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec2f(4.0, 4.0);
       let p = vec2f(1.0, 2.0);

       // Test periodicity: pnoise(p, period) == pnoise(p + period, period)
       let n1 = pnoise2(p, period);
       let n2 = pnoise2(p + period, period);
       let n3 = pnoise2(p + period * 2.0, period);

       test::results[0] = vec4f(n1, n2, n3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test periodicity property: noise repeats exactly after one period
  expectCloseTo([result[0], result[0]], [result[1], result[2]]);
  // Periodic noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0], [result[0]]);
});

test("pnoise3", async () => {
  const src = `
     import lygia::generative::pnoise::pnoise3;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec3f(4.0, 4.0, 4.0);
       let p = vec3f(1.0, 2.0, 3.0);

       // Test periodicity: pnoise(p, period) == pnoise(p + period, period)
       let n1 = pnoise3(p, period);
       let n2 = pnoise3(p + period, period);
       let n3 = pnoise3(p + period * 2.0, period);

       test::results[0] = vec4f(n1, n2, n3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test periodicity property: noise repeats exactly after one period
  expectCloseTo([result[0], result[0]], [result[1], result[2]]);
  // Periodic noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0], [result[0]]);
});

test("pnoise4", async () => {
  const src = `
     import lygia::generative::pnoise::pnoise4;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec4f(4.0, 4.0, 4.0, 4.0);
       let p = vec4f(1.0, 2.0, 3.0, 4.0);

       // Test periodicity: pnoise(p, period) == pnoise(p + period, period)
       let n1 = pnoise4(p, period);
       let n2 = pnoise4(p + period, period);
       let n3 = pnoise4(p + period * 2.0, period);

       test::results[0] = vec4f(n1, n2, n3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test periodicity property: noise repeats exactly after one period
  expectCloseTo([result[0], result[0]], [result[1], result[2]]);
  // Periodic noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0], [result[0]]);
});

test("srandom2", async () => {
  const src = `
     import lygia::generative::srandom::srandom2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let r1 = srandom2(p1);
       let r2 = srandom2(p2);
       let r3 = srandom2(p3);

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.7109375], [result[0]]);
});

test("worley2", async () => {
  const src = `
     import lygia::generative::worley::worley2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(1.5, 2.5); // Point in same cell

       let w1 = worley2(p1);
       let w2 = worley2(p2);
       let w3 = worley2(p3);

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different positions have different distances
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Worley noise returns distance in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.7470544576644897], [result[0]]);
});

// Noise with derivatives
test("noised2", async () => {
  const src = `
     import lygia::generative::noised::noised2;

     @compute @workgroup_size(1)
     fn foo() {
       let h = 0.001;
       let p = vec2f(1.0, 2.0);

       // Get noise and analytical derivatives
       let nd = noised2(p);
       let noise_val = nd.x;
       let dx_analytical = nd.y;
       let dy_analytical = nd.z;

       // Compute numerical derivatives via finite differences
       let dx_numerical = (noised2(p + vec2f(h, 0.0)).x - noised2(p - vec2f(h, 0.0)).x) / (2.0 * h);
       let dy_numerical = (noised2(p + vec2f(0.0, h)).x - noised2(p - vec2f(0.0, h)).x) / (2.0 * h);

       test::results[0] = vec4f(dx_analytical, dx_numerical, dy_analytical, dy_numerical);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test that analytical derivatives match numerical derivatives (with small tolerance for numerical error)
  expectCloseTo([result[0], result[2]], [result[1], result[3]], 2);
  // Derivatives should be in reasonable range
  expect(Math.abs(result[0])).toBeLessThan(5.0);
  expect(Math.abs(result[2])).toBeLessThan(5.0);
  // Regression: exact output value
  expectCloseTo([-0.3647780418395996], [result[0]]);
});

test("noised3", async () => {
  const src = `
     import lygia::generative::noised::noised3;

     @compute @workgroup_size(1)
     fn foo() {
       let h = 0.001;
       let p = vec3f(1.0, 2.0, 3.0);

       // Get noise and analytical derivatives
       let nd = noised3(p);
       let dx_analytical = nd.y;
       let dy_analytical = nd.z;
       let dz_analytical = nd.w;

       // Compute numerical derivatives via finite differences
       let dx_numerical = (noised3(p + vec3f(h, 0.0, 0.0)).x - noised3(p - vec3f(h, 0.0, 0.0)).x) / (2.0 * h);
       let dy_numerical = (noised3(p + vec3f(0.0, h, 0.0)).x - noised3(p - vec3f(0.0, h, 0.0)).x) / (2.0 * h);
       let dz_numerical = (noised3(p + vec3f(0.0, 0.0, h)).x - noised3(p - vec3f(0.0, 0.0, h)).x) / (2.0 * h);

       // Pack all results into a single vec4f (we only have one result slot)
       test::results[0] = vec4f(dx_analytical, dx_numerical, dy_analytical, dy_numerical);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test that analytical derivatives match numerical derivatives (with small tolerance for numerical error)
  expectCloseTo([result[0], result[2]], [result[1], result[3]], 2);
  // Derivatives should be in reasonable range
  expect(Math.abs(result[0])).toBeLessThan(5.0);
  expect(Math.abs(result[2])).toBeLessThan(5.0);
  // Regression: exact output value
  expectCloseTo([-0.59765625], [result[0]]);
});

test("wavelet2", async () => {
  const src = `
     import lygia::generative::wavelet::wavelet2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let w1 = wavelet2(p1);
       let w2 = wavelet2(p2);
       let w3 = wavelet2(p3);

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Wavelet noise should return values in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.19464324414730072], [result[0]]);
});

test("wavelet3", async () => {
  const src = `
     import lygia::generative::wavelet::wavelet3;

     @compute @workgroup_size(1)
     fn foo() {
       // Third component is the phase parameter
       let p = vec2f(1.0, 2.0);
       let phase1 = 0.0;
       let phase2 = 1.0;

       let w1 = wavelet3(vec3f(p, phase1));
       let w2 = wavelet3(vec3f(p, phase2));
       let w3 = wavelet3(vec3f(p, phase1)); // Same as w1

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same position and phase produce same output
  expectCloseTo([result[0]], [result[2]]);
  // Test that different phases produce different outputs
  expect(result[0]).not.toBeCloseTo(result[1], 1);
  // Wavelet noise should return values in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.19464324414730072], [result[0]]);
});

test("waveletScaled2", async () => {
  const src = `
     import lygia::generative::wavelet::waveletScaled2;

     @compute @workgroup_size(1)
     fn foo() {
       let p = vec2f(1.0, 2.0);
       let phase = 0.5;

       // Test with same phase but different scales - not frequency
       let w1 = waveletScaled2(p, phase);
       let w2 = waveletScaled2(p, phase); // Same
       let w3 = waveletScaled2(p * 2.0, phase); // Different position

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that scaling position changes output
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Wavelet noise should return values in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.11138363927602768], [result[0]]);
});

test("waveletScaled3 - with custom scale parameter", async () => {
  const src = `
     import lygia::generative::wavelet::waveletScaled3;

     @compute @workgroup_size(1)
     fn foo() {
       // waveletScaled3 takes vec3f(position.xy, phase) and scale parameter
       let p = vec2f(1.0, 2.0);
       let phase = 0.5;
       let scale1 = 1.0;
       let scale2 = 2.0;

       // Test with same position/phase but different scales
       let w1 = waveletScaled3(vec3f(p, phase), scale1);
       let w2 = waveletScaled3(vec3f(p, phase), scale1); // Same
       let w3 = waveletScaled3(vec3f(p, phase), scale2); // Different scale

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different scales produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Wavelet noise should return values in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.094538614153862], [result[0]]);
});

test("random", async () => {
  const src = `
     import lygia::generative::random::random;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random(1.0);
       let r2 = random(1.0); // Same input
       let r3 = random(2.0); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // TODO: Add GLSL parity test - verify result[0] matches GLSL random(1.0)
  // Regression: exact output value
  expectCloseTo([0.7629680633544922], [result[0]]);
});

test("random2", async () => {
  const src = `
     import lygia::generative::random::random2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let r1 = random2(p1);
       let r2 = random2(p1); // Same input
       let r3 = random2(vec2f(3.0, 4.0)); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.61529541015625], [result[0]]);
});

test("random3", async () => {
  const src = `
     import lygia::generative::random::random3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let r1 = random3(p1);
       let r2 = random3(p1); // Same input
       let r3 = random3(vec3f(4.0, 5.0, 6.0)); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.37200927734375], [result[0]]);
});

test("random4", async () => {
  const src = `
     import lygia::generative::random::random4;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let r1 = random4(p1);
       let r2 = random4(p1); // Same input
       let r3 = random4(vec4f(5.0, 6.0, 7.0, 8.0)); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.51806640625], [result[0]]);
});

test("random21 - basic output", async () => {
  const src = `
     import lygia::generative::random::random21;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random21(1.0);
       let r2 = random21(1.0); // Same input
       let r3 = random21(2.0); // Different input

       // Test determinism and range
       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.8786392211914062], [result[0]]);
});

test("random22 - basic output", async () => {
  const src = `
     import lygia::generative::random::random22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let r1 = random22(p1);
       let r2 = random22(p1); // Same input

       // Test determinism and range
       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.2332763671875], [result[0]]);
});

test("random23 - basic output", async () => {
  const src = `
     import lygia::generative::random::random23;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let r1 = random23(p1);
       let r2 = random23(p1); // Same input

       // Test determinism and range
       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.683746337890625], [result[0]]);
});

test("random31 - basic output", async () => {
  const src = `
     import lygia::generative::random::random31;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random31(1.0);
       let r2 = random31(1.0); // Same input

       // Test determinism and range (can only fit 3 components, test first 3)
       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output (test first component)
  expectCloseTo([result[0]], [result[3]]);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.8786392211914062], [result[0]]);
});

test("random32 - basic output", async () => {
  const src = `
     import lygia::generative::random::random32;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let r1 = random32(p1);
       let r2 = random32(p1); // Same input

       // Test determinism and range (can only fit 3 components, test first 3)
       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output (test first component)
  expectCloseTo([result[0]], [result[3]]);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.25337982177734375], [result[0]]);
});

test("random33 - basic output", async () => {
  const src = `
     import lygia::generative::random::random33;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let r1 = random33(p1);
       let r2 = random33(p1); // Same input

       // Test determinism and range (can only fit 3 components, test first 3)
       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output (test first component)
  expectCloseTo([result[0]], [result[3]]);
  // Random should return values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.45416259765625], [result[0]]);
});

test("random41 - determinism and range", async () => {
  const src = `
     import lygia::generative::random::random41;

     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random41(1.0);
     }
   `;
  const result1 = await testCompute(src, "vec4f");
  const result2 = await testCompute(src, "vec4f");

  // Test determinism: same input produces same output across runs
  expectCloseTo(result1, result2);

  // Range check: all components in [0, 1]
  result1.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Regression: exact output value
  expectCloseTo([0.382354736328125, 0.42840576171875, 0.5390167236328125, 0.4849395751953125], result1);
});

test("random42 - hash properties", async () => {
  // Test determinism
  const src1 = `
     import lygia::generative::random::random42;
     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0);  // Same input
       test::results[0] = random42(p1) - random42(p2);
     }
   `;
  const determinism = await testCompute(src1, "vec4f");
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism, 5); // Same input → same output

  // Test range and independence
  const src2 = `
     import lygia::generative::random::random42;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random42(vec2f(1.0, 2.0));
     }
   `;
  const result = await testCompute(src2, "vec4f");

  // All components in [0, 1]
  result.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Components should differ (not all identical)
  const allSame = result.every(v => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Test avalanche effect: small input change causes significant output change
  const src3 = `
     import lygia::generative::random::random42;
     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random42(vec2f(1.0, 2.0));
       let r3 = random42(vec2f(1.01, 2.0));  // Tiny 1% change in input
       test::results[0] = abs(r1 - r3);  // Difference magnitude
     }
   `;
  const avalanche = await testCompute(src3, "vec4f");
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.03); // Hash property: small input → significant output change

  // Regression: exact output value
  expectCloseTo([0.66876220703125, 0.996826171875, 0.603179931640625, 0.9088134765625], result);
});

test("random43 - hash properties", async () => {
  // Test determinism
  const src1 = `
     import lygia::generative::random::random43;
     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0);  // Same input
       test::results[0] = random43(p1) - random43(p2);
     }
   `;
  const determinism = await testCompute(src1, "vec4f");
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism, 5);

  // Test range and independence
  const src2 = `
     import lygia::generative::random::random43;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random43(vec3f(1.0, 2.0, 3.0));
     }
   `;
  const result = await testCompute(src2, "vec4f");

  // All components in [0, 1]
  result.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Components should differ (not all identical)
  const allSame = result.every(v => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Test avalanche effect
  const src3 = `
     import lygia::generative::random::random43;
     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random43(vec3f(1.0, 2.0, 3.0));
       let r3 = random43(vec3f(1.01, 2.0, 3.0));
       test::results[0] = abs(r1 - r3);
     }
   `;
  const avalanche = await testCompute(src3, "vec4f");
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.1);

  // Regression: exact output value
  expectCloseTo([0.407958984375, 0.216583251953125, 0.96063232421875, 0.4371337890625], result);
});

test("random44 - hash properties", async () => {
  // Test determinism
  const src1 = `
     import lygia::generative::random::random44;
     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0);  // Same input
       test::results[0] = random44(p1) - random44(p2);
     }
   `;
  const determinism = await testCompute(src1, "vec4f");
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism, 5);

  // Test range and independence
  const src2 = `
     import lygia::generative::random::random44;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random44(vec4f(1.0, 2.0, 3.0, 4.0));
     }
   `;
  const result = await testCompute(src2, "vec4f");

  // All components in [0, 1]
  result.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Components should differ (not all identical)
  const allSame = result.every(v => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Test avalanche effect
  const src3 = `
     import lygia::generative::random::random44;
     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random44(vec4f(1.0, 2.0, 3.0, 4.0));
       let r3 = random44(vec4f(1.01, 2.0, 3.0, 4.0));
       test::results[0] = abs(r1 - r3);
     }
   `;
  const avalanche = await testCompute(src3, "vec4f");
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.1);

  // Regression: exact output value
  expectCloseTo([0.81640625, 0.07281494140625, 0.7236328125, 0.70635986328125], result);
});

// Simplex noise variants - vector outputs
test("snoise22", async () => {
  const src = `
     import lygia::generative::snoise::snoise22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(1.01, 2.01); // Nearby point

       let n1 = snoise22(p1);
       let n2 = snoise22(p2);
       let n3 = snoise22(p3);

       test::results[0] = vec4f(n1.x, n1.y, n2.x, n2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Simplex noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
});

test("snoise33", async () => {
  const src = `
     import lygia::generative::snoise::snoise33;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(1.01, 2.01, 3.01); // Nearby point

       let n1 = snoise33(p1);
       let n2 = snoise33(p2);
       let n3 = snoise33(p3);

       // Test continuity by computing difference
       test::results[0] = vec4f(n1.x, n1.y, n1.z, length(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test continuity: nearby points should have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Simplex noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(-1.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.7335153818130493], [result[0]]);
});

test("snoise34", async () => {
  const src = `
     import lygia::generative::snoise::snoise34;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0); // Same point
       let p3 = vec4f(1.01, 2.01, 3.01, 4.01); // Nearby point

       let n1 = snoise34(p1);
       let n2 = snoise34(p2);
       let n3 = snoise34(p3);

       // Test continuity by computing difference
       test::results[0] = vec4f(n1.x, n1.y, n1.z, length(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test continuity: nearby points should have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Simplex noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(-1.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.37476846575737], [result[0]]);
});

test("snoise4", async () => {
  const src = `
     import lygia::generative::snoise::snoise4;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0); // Same point
       let p3 = vec4f(1.01, 2.01, 3.01, 4.01); // Nearby point

       let n1 = snoise4(p1);
       let n2 = snoise4(p2);
       let n3 = snoise4(p3);

       test::results[0] = vec4f(n1, n2, n3, abs(n3 - n1));
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test continuity: nearby points have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Simplex noise should return values in range [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.37476846575737], [result[0]]);
});

// Signed random variants
test("srandom", async () => {
  const src = `
     import lygia::generative::srandom::srandom;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = srandom(12.34);
       let r2 = srandom(12.34); // Same input
       let r3 = srandom(98.76); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.455078125], [result[0]]);
});

test("srandom22", async () => {
  const src = `
     import lygia::generative::srandom::srandom22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let r1 = srandom22(p1);
       let r2 = srandom22(p2);
       let r3 = srandom22(p3);

       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.3647780418395996], [result[0]]);
});

test("srandom3", async () => {
  const src = `
     import lygia::generative::srandom::srandom3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(4.0, 5.0, 6.0); // Different point

       let r1 = srandom3(p1);
       let r2 = srandom3(p2);
       let r3 = srandom3(p3);

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.1328125], [result[0]]);
});

test("srandom33", async () => {
  const src = `
     import lygia::generative::srandom::srandom33;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point

       let r1 = srandom33(p1);
       let r2 = srandom33(p2);

       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output (check first component)
  expectCloseTo([result[0]], [result[3]]);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(-1.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.58984375], [result[0]]);
});

test("srandom4", async () => {
  const src = `
     import lygia::generative::srandom::srandom4;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0); // Same point
       let p3 = vec4f(5.0, 6.0, 7.0, 8.0); // Different point

       let r1 = srandom4(p1);
       let r2 = srandom4(p2);
       let r3 = srandom4(p3);

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.6796875], [result[0]]);
});

test("srandom_tile22", async () => {
  const src = `
     import lygia::generative::srandom::srandom_tile22;

     @compute @workgroup_size(1)
     fn foo() {
       let tileLength = 4.0;
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(5.0, 6.0); // Should tile with period=4.0

       // p2 = p1 + (4.0, 4.0), so after tiling they should be the same
       let r1 = srandom_tile22(p1, tileLength);
       let r2 = srandom_tile22(p2, tileLength);

       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test tiling: points separated by tileLength should produce same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.3647780418395996], [result[0]]);
});

test("srandom_tile33", async () => {
  const src = `
     import lygia::generative::srandom::srandom_tile33;

     @compute @workgroup_size(1)
     fn foo() {
       let tileLength = 4.0;
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(5.0, 6.0, 7.0); // Should tile with period=4.0

       // p2 = p1 + (4.0, 4.0, 4.0), so after tiling they should be the same
       let r1 = srandom_tile33(p1, tileLength);
       let r2 = srandom_tile33(p2, tileLength);

       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test tiling: points separated by tileLength should produce same output (check first component)
  expectCloseTo([result[0]], [result[3]]);
  // Signed random returns values in [-1, 1]
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(-1.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(-1.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([-0.58984375], [result[0]]);
});

// Worley noise variants
test("worley22", async () => {
  const src = `
     import lygia::generative::worley::worley22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let w1 = worley22(p1);
       let w2 = worley22(p2);
       let w3 = worley22(p3);

       test::results[0] = vec4f(w1.x, w1.y, w2.x, w2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Worley noise returns distances (F1, F2) in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.5);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.5);
  // F1 should be less than or equal to F2 (closest point <= second closest)
  expect(result[0]).toBeLessThanOrEqual(result[1] + 0.001);
  // Regression: exact output value
  expectCloseTo([0.25294554233551025], [result[0]]);
});

test("worley3", async () => {
  const src = `
     import lygia::generative::worley::worley3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(4.0, 5.0, 6.0); // Different point

       let w1 = worley3(p1);
       let w2 = worley3(p2);
       let w3 = worley3(p3);

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Worley noise returns 1.0 - distance, so values in [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.38763654232025146], [result[0]]);
});

test("worley32", async () => {
  const src = `
     import lygia::generative::worley::worley32;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point

       let w1 = worley32(p1);
       let w2 = worley32(p2);

       test::results[0] = vec4f(w1.x, w1.y, w2.x, w2.y);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Worley noise returns distances (F1, F2) in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(2.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(2.0);
  // F1 should be less than or equal to F2 (closest point <= second closest)
  expect(result[0]).toBeLessThanOrEqual(result[1] + 0.001);
  // Regression: exact output value
  expectCloseTo([0.6123634576797485], [result[0]]);
});

test("wavelet - base function with custom phase and scale", async () => {
  const src = `
     import lygia::generative::wavelet::wavelet;

     @compute @workgroup_size(1)
     fn foo() {
       let p = vec2f(1.0, 2.0);
       let phase = 0.5;
       let scale = 1.5;

       // Test determinism: same inputs produce same output
       let w1 = wavelet(p, phase, scale);
       let w2 = wavelet(p, phase, scale);

       // Test that different phase affects output
       let w3 = wavelet(p, 1.0, scale);

       // Test that different scale affects output
       let w4 = wavelet(p, phase, 2.0);

       test::results[0] = vec4f(w1, w2, w3, w4);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different phase produces different output
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Test that different scale produces different output
  expect(result[0]).not.toBeCloseTo(result[3], 1);
  // Wavelet noise should return values in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(-1.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Regression: exact output value
  expectCloseTo([0.18840710818767548], [result[0]]);
});

// ============================================================================
// Distribution Tests
// ============================================================================
// These tests validate that random functions produce uniform distributions
// with correct statistical properties (mean, uniformity across range).

test("random - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        test::results[i] = random(f32(i));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = random2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random3 - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random3;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 16u);
        let y = f32((i / 16u) % 16u);
        let z = f32(i / 256u);
        test::results[i] = random3(vec3f(x, y, z));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("srandom - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        // Vary inputs more to avoid patterns
        test::results[i] = srandom(f32(i) * 1.234 + 0.567);
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("srandom2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = srandom2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("random22 - distribution (x component)", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random22;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        let sample = random22(vec2f(x, y));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random33 - distribution (x component)", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random33;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 16u);
        let y = f32((i / 16u) % 16u);
        let z = f32(i / 256u);
        let sample = random33(vec3f(x, y, z));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("srandom22 - distribution (x component)", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom22;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        // Vary inputs more to avoid patterns
        let x = f32(i % 32u) * 1.1 + 0.3;
        let y = f32(i / 32u) * 1.3 + 0.7;
        let sample = srandom22(vec2f(x, y));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});
