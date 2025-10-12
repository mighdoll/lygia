# Test Improvement Plan: geometry-triangle.test.ts

**Date:** 2025-10-12
**Status:** Updated with specific expected values based on test-review.md guidance

## Summary of Findings

**Total Tests:** 7
**Good Tests:** 4 (57%)
**Trivial Tests:** 3 (43%)

### Good Tests (Keep As-Is)
1. ✅ **Triangle struct** - Tests struct functionality and edge length calculation (3-4-5 right triangle)
2. ✅ **area** - Tests area formula with right triangle (base=2, height=2, expects 2.0)
3. ✅ **centroid** - Tests centroid calculation (expects average of vertices: (1,1,0))
4. ✅ **normal** - Tests surface normal for XY-plane triangle (expects +Z direction)

### Trivial Tests (Need Improvement)
1. ⚠️ **barycentric** - RANGE-ONLY: Only validates that coordinates sum to 1.0, doesn't test specific values
2. ⚠️ **barycentric2** - RANGE-ONLY: Only validates that coordinates sum to 1.0, doesn't test actual computation
3. ⚠️ **barycentric3** - WEAK TEST: Tests centroid point but uses loose tolerance (0.1) and doesn't test edge cases

---

## Key Changes from test-review.md Guidance

1. **Specific expected values:** All improved tests now include exact expected barycentric coordinates (e.g., [1.0, 0.0, 0.0] for vertex, [0.333, 0.333, 0.333] for centroid)
2. **Tighter tolerance:** Changed from tolerance=1 (0.1) to tolerance=2 (0.01) for better precision
3. **Meaningful geometric cases:** Tests now use:
   - Origin at vertex (tests edge case)
   - Origin at centroid (tests symmetry)
   - Edge midpoints (tests interpolation)
   - Points outside triangle (tests generalized coordinates)
4. **Understanding function behavior:** Corrected understanding that `barycentric(a,b,c)` computes coords of **origin** relative to triangle, not arbitrary point
5. **Multiple test cases:** Each function now has 2-4 tests covering different geometric scenarios

---

## Detailed Improvement Plans

### 1. barycentric - RANGE-ONLY TEST

**Current Problem:**
- Only checks that barycentric coordinates sum to 1.0
- Only validates all coordinates are >= -0.1 (range check)
- Uses three arbitrary vectors (1,0,0), (0,1,0), (0,0,1) without testing specific barycentric values
- Doesn't verify the actual mathematical computation

**Function Signature:**
```typescript
fn barycentric(a: vec3f, b: vec3f, c: vec3f) -> vec3f
```

**What it does:**
From the WESL source code (geometry/triangle/barycentric.wesl), this function computes barycentric coordinates using dot products based on "Real-Time Collision Detection" by Christer Ericson. The function takes three vectors and returns weights (u, v, w) where u + v + w = 1.

**IMPORTANT NOTE:** This function computes barycentric coordinates **of the origin (0,0,0)** relative to the triangle formed by vectors a, b, c. It does NOT compute coordinates of a point on the triangle.

**Improved Test:**

```typescript
test("barycentric - origin inside triangle", async () => {
  const src = `
    import lygia::geometry::triangle::barycentric::barycentric;

    @compute @workgroup_size(1)
    fn foo() {
      // Create triangle with vertices around the origin
      // This tests when origin is inside the triangle
      let a = vec3f(1.0, 0.0, 0.0);
      let b = vec3f(-0.5, 0.866, 0.0);   // 120° from a
      let c = vec3f(-0.5, -0.866, 0.0);  // 240° from a

      let result = barycentric(a, b, c);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // For equilateral triangle centered at origin:
  // Barycentric coordinates should be approximately (1/3, 1/3, 1/3)
  expectCloseTo([0.333333, 0.333333, 0.333333], result, 2);

  // Verify the fundamental property: coordinates must sum to 1.0
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);
});

test("barycentric - origin at vertex", async () => {
  const src = `
    import lygia::geometry::triangle::barycentric::barycentric;

    @compute @workgroup_size(1)
    fn foo() {
      // Origin coincides with vertex 'a'
      let a = vec3f(0.0, 0.0, 0.0);
      let b = vec3f(1.0, 0.0, 0.0);
      let c = vec3f(0.0, 1.0, 0.0);

      let result = barycentric(a, b, c);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // When origin is at vertex 'a', expect (1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result, 2);
});
```

**Why this is better:**
- Tests specific mathematical output values with expected coordinates
- Tests two meaningful cases: origin inside triangle and origin at vertex
- Uses tighter tolerance (0.01 instead of 0.1)
- Correctly understands what the function computes (origin's barycentric coords, not arbitrary point)
- Still validates the sum-to-1 property

---

### 2. barycentric2 - RANGE-ONLY TEST

**Current Problem:**
- Only validates that coordinates sum to 1.0
- Only checks coordinates are >= -0.1 (range check)
- Doesn't verify the actual barycentric computation for the Triangle struct
- Uses arbitrary triangle without testing specific expected values

**Function Signature:**
```typescript
fn barycentric2(tri: Triangle) -> vec3f
```

**What it does:**
Wrapper function that calls `barycentric(tri.a, tri.b, tri.c)` using a Triangle struct. Computes barycentric coordinates of the origin relative to the triangle.

**Improved Test:**

```typescript
test("barycentric2 - origin at vertex", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric2;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      // Right triangle with vertex a at origin
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(1.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 1.0, 0.0);

      let result = barycentric2(tri);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Origin is at vertex a, expect (1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result, 2);
});

test("barycentric2 - origin centered in triangle", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric2;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      // Equilateral triangle centered at origin
      tri.a = vec3f(1.0, 0.0, 0.0);
      tri.b = vec3f(-0.5, 0.866, 0.0);
      tri.c = vec3f(-0.5, -0.866, 0.0);

      let result = barycentric2(tri);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Origin at centroid of equilateral triangle, expect (1/3, 1/3, 1/3)
  expectCloseTo([0.333333, 0.333333, 0.333333], result, 2);

  // Verify sum to 1.0 property
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);
});
```

**Why this is better:**
- Tests specific expected coordinate values with meaningful inputs
- Tests both edge case (origin at vertex) and symmetric case (origin at center)
- Uses tighter tolerance (0.01 instead of 0.1)
- Validates the Triangle struct wrapper works correctly
- Tests mathematical properties with specific values, not just "doesn't crash"

---

### 3. barycentric3 - WEAK TEST

**Current Problem:**
- Tests centroid point but only validates result is "close to" (1/3, 1/3, 1/3)
- Doesn't test edge cases like points at vertices
- Doesn't test points outside the triangle
- Uses low precision (tolerance=1, which is 0.1 relative error)

**Function Signature:**
```typescript
fn barycentric3(tri: Triangle, pos: vec3f) -> vec3f
```

**What it does:**
Computes barycentric coordinates of a point `pos` relative to triangle `tri` using area ratios (sub-triangle areas divided by total area).

**Improved Test:**

```typescript
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

      // Point at vertex a should have coords (1, 0, 0)
      let result = barycentric3(tri, tri.a);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Point at vertex a should have full weight at a
  expectCloseTo([1.0, 0.0, 0.0], result, 2);
});

test("barycentric3 - point at edge midpoint", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(2.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 2.0, 0.0);

      // Point at midpoint of edge a-b (halfway between a and b)
      let midpoint = vec3f(1.0, 0.0, 0.0);
      let result = barycentric3(tri, midpoint);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Midpoint of a-b should have coords (0.5, 0.5, 0)
  expectCloseTo([0.5, 0.5, 0.0], result, 2);
});

test("barycentric3 - centroid", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(3.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 3.0, 0.0);

      // Centroid is at (1, 1, 0) for this triangle
      let centroid = vec3f(1.0, 1.0, 0.0);
      let result = barycentric3(tri, centroid);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // Centroid should have equal barycentric coordinates (1/3, 1/3, 1/3)
  expectCloseTo([0.333333, 0.333333, 0.333333], result, 2);

  // Verify sum to 1.0
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);
});

test("barycentric3 - point outside triangle", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(1.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 1.0, 0.0);

      // Point outside the triangle at (2, 2, 0)
      let outside = vec3f(2.0, 2.0, 0.0);
      let result = barycentric3(tri, outside);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");

  // For points outside, at least one coordinate should be negative
  // and they should still sum to 1.0 (generalized barycentric coordinates)
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);

  // For point (2,2,0) relative to triangle at origin:
  // we expect negative weight for vertex a, positive for b and c
  expect(result[0]).toBeLessThan(0.0);
});
```

**Why this is better:**
- Tests multiple meaningful cases: vertex, edge midpoint, centroid, outside point
- Validates specific expected coordinate values, not just ranges
- Tests mathematical properties like sum-to-1 for both inside and outside points
- Uses tighter precision (tolerance=2, which is 0.01 relative error)
- Demonstrates that barycentric coordinates work for points outside the triangle (negative coordinates)

---

## Implementation Priority

Based on test-review.md guidance and the 57% good test rate (below 87.5% like AABB tests):

1. **High Priority:** barycentric3 - Most important as it tests point-in-triangle with actual positions, has multiple meaningful test cases
2. **High Priority:** barycentric - Tests core triangle barycentric computation, needed for barycentric2
3. **Medium Priority:** barycentric2 - Simple wrapper, but should be improved for consistency

---

## Testing Strategy

Each improved test should:
1. Use geometrically simple triangles (right triangles, equilateral triangles)
2. Test points at known locations (vertices, edge midpoints, centroid)
3. Validate specific mathematical properties (coordinates at vertices, sum-to-1)
4. Use tighter precision tolerance (0.01 instead of 0.1)
5. Test edge cases (outside points, degenerate cases if applicable)

---

## Additional Notes

- The `barycentric` function (3 vec3f arguments) computes barycentric coordinates using dot products
- The `barycentric3` function (Triangle + point) uses area ratios via cross products
- These are two different algorithms for computing barycentric coordinates
- Tests should verify both algorithms produce correct results for their use cases
