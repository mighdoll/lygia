import { expect, test } from "vitest";
import { expectCloseTo, testCompute, testFragment } from "./testUtil.ts";

// Draw utility functions

// TODO: Add image test - render strokes at various distances/widths in 4x4 or 8x8 texture
// stroke uses aastep which requires fwidth (fragment shader only)
test("stroke - basic behavior", async () => {
  const src = `
    import lygia::draw::stroke::stroke;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let size = 0.5;    // Stroke centered at distance 0.5
      let width = 0.2;   // Width 0.2 (inner edge: 0.4, outer edge: 0.6)

      // Case 1: At the center of the stroke (x = size = 0.5)
      // Expected: Maximum value (~1.0)
      let center = stroke(0.5, size, width);

      // Case 2: Far inside stroke region (x << size - width/2)
      // Expected: Close to 0.0 (outside the stroke band)
      let far_inside = stroke(0.2, size, width);

      // Case 3: Far outside stroke region (x >> size + width/2)
      // Expected: Close to 0.0 (outside the stroke band)
      let far_outside = stroke(0.8, size, width);

      // Case 4: Within stroke band but not at center (x = 0.55)
      // Expected: High value (within the stroke band from 0.4 to 0.6)
      let in_band = stroke(0.55, size, width);

      return vec4f(center, far_inside, far_outside, in_band);
    }
  `;
  const result = await testFragment(src, { size: [2, 2] });

  // Center of stroke should be close to 1.0
  expect(result[0]).toBeGreaterThan(0.9);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Outside regions (both inside and outside the stroke band) should be close to 0.0
  expect(result[1]).toBeLessThan(0.1); // Far inside
  expect(result[2]).toBeLessThan(0.1); // Far outside

  // Within stroke band should be high (close to 1.0)
  expect(result[3]).toBeGreaterThan(0.8);

  // Exact values to catch regressions
  expectCloseTo([1, 0, 0, 1], result);
});

// TODO: Add image test - render strokeEdge with various edge widths in 4x4 or 8x8 texture
test("strokeEdge - basic behavior", async () => {
  const src = `
    import lygia::draw::stroke::strokeEdge;

    @compute @workgroup_size(1)
    fn foo() {
      // Test stroke with different distances
      let size = 0.5;
      let width = 0.2;
      let edge = 0.01;

      // Case 1: Inside the stroke (distance < size-width/2)
      // stroke inner edge at 0.5-0.1=0.4, outer edge at 0.5+0.1=0.6
      let inside = strokeEdge(0.45, size, width, edge);

      // Case 2: Outside the stroke (distance > size+width/2)
      let outside = strokeEdge(0.7, size, width, edge);

      // Case 3: On the outer edge transition (distance ≈ size+width/2)
      let on_edge = strokeEdge(0.6, size, width, edge);

      test::results[0] = vec4f(inside, outside, on_edge, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Inside the stroke should be close to 1.0
  expect(result[0]).toBeGreaterThan(0.9);

  // Outside the stroke should be close to 0.0
  expect(result[1]).toBeLessThan(0.1);

  // On the edge should be intermediate value
  expect(result[2]).toBeGreaterThan(0.1);
  expect(result[2]).toBeLessThan(0.9);

  // Exact values to catch regressions
  expectCloseTo([1, 0, 0.5000007152557373, 0], result);
});
