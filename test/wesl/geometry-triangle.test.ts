import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("Triangle struct", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(3.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 4.0, 0.0);
      // Compute edge lengths to test struct functionality
      let ab = length(tri.b - tri.a);
      let bc = length(tri.c - tri.b);
      let ca = length(tri.a - tri.c);
      test::results[0] = vec3f(ab, bc, ca);
    }
  `;
  const result = await testCompute(src, "vec3f");
  // 3-4-5 right triangle: edges are 3, 5, 4
  expectCloseTo([3.0, 5.0, 4.0], result);
});

test("area", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::area::area;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(2.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 2.0, 0.0);
      let result = area(tri);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Area of triangle with base=2, height=2 is 0.5 * 2 * 2 = 2.0
  expectCloseTo([2.0], result);
});

test("barycentric - computes normalized coordinates", async () => {
  const src = `
    import lygia::geometry::triangle::barycentric::barycentric;

    @compute @workgroup_size(1)
    fn foo() {
      // Test with specific vectors - this tests a well-defined case
      let a = vec3f(1.0, 0.0, 0.0);
      let b = vec3f(0.0, 1.0, 0.0);
      let c = vec3f(0.0, 0.0, 1.0);

      let result = barycentric(a, b, c);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // The function should return specific coordinates for this configuration
  // Based on the Ericson algorithm, this returns (1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result, 0.01);

  // Barycentric coordinates must always sum to 1.0
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);
});

test("barycentric2 - Triangle struct wrapper", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric2;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(1.0, 0.0, 0.0);
      tri.b = vec3f(0.0, 1.0, 0.0);
      tri.c = vec3f(0.0, 0.0, 1.0);
      let result = barycentric2(tri);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Should produce same result as barycentric(a, b, c)
  expectCloseTo([1.0, 0.0, 0.0], result, 0.01);

  // Verify sum-to-1 property
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);
});

test("barycentric3 - point at vertex", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(1.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 1.0, 0.0);

      // Test point at vertex a - coordinate for a should be highest
      let result = barycentric3(tri, tri.a);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Point at vertex a should have dominant weight at a
  // Note: This function returns unnormalized coords (sum ≠ 1)
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  expect(result[1]).toBeLessThan(0.01);
  expect(result[2]).toBeLessThan(0.01);
});

test("barycentric3 - edge midpoint", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(2.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 2.0, 0.0);

      // Point at midpoint of edge a-b
      let midpoint = vec3f(1.0, 0.0, 0.0);
      let result = barycentric3(tri, midpoint);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Midpoint of a-b should have equal weights for a and b, zero for c
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.01);
  expect(result[2]).toBeLessThan(0.01);
});

test("centroid", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::centroid::centroid;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(3.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 3.0, 0.0);
      let result = centroid(tri);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");
  // Centroid should be at (1, 1, 0)
  expectCloseTo([1.0, 1.0, 0.0], result);
});

test("normal", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::normal::normal;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(1.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 1.0, 0.0);
      let result = normal(tri);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");
  // Normal of XY plane triangle should point in +Z direction
  expectCloseTo([0.0, 0.0, 1.0], result);
});
