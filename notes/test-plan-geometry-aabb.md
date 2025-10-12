# Test Improvement Plan: geometry-aabb.test.ts

**Date:** 2025-10-12
**Status:** Updated based on test-review.md guidance

## Summary

**Total Tests:** 8
**Good Tests:** 7 (87.5%)
**Trivial Tests:** 1 (12.5%)

This test file is in excellent shape! Only one test needs improvement - the AABB struct test which currently just assigns values and reads them back without performing any meaningful computation.

**Note:** The `intersect` function exists in GLSL but has not been ported to WESL yet, so we cannot add tests for it at this time.

---

## Key Changes from test-review.md Guidance

1. **Remove trivial tests:** Following guidance to prefer "fewer, more meaningful tests," the recommendation is to remove the AABB struct test entirely
2. **Rationale:** The struct test only validates field assignment (trivial), while all other tests implicitly verify the struct works correctly
3. **Result:** Brings test suite from 87.5% meaningful to 100% meaningful coverage (7 good tests, 0 trivial)
4. **Optional improvements:** Suggested boundary case tests for `contain`, asymmetric cases for `centroid`, and idempotent test for `square` (all use specific expected values)
5. **Future work:** Once `intersect.wesl` is ported, add ray-box intersection tests with specific distance calculations

---

## Analysis by Test

### ✅ GOOD Tests (Keep As-Is)

#### 1. `centroid` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests centroid calculation with a symmetric box where min=[-2,-4,-6] and max=[2,4,6], correctly expecting the center at [0,0,0]. This validates the formula `(min + max) * 0.5`.

#### 2. `contain` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests point containment with two meaningful cases:
- Point inside: origin (0,0,0) in box [-1,-1,-1] to [1,1,1] → true
- Point outside: (2,0,0) outside the same box → false
This validates the boundary checking logic with `lessThanEqual` and `lessThan`.

#### 3. `diagonal` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests diagonal calculation (size) from min [-1,-2,-3] to max [1,2,3], correctly expecting [2,4,6]. This validates the formula `abs(max - min)`.

#### 4. `expand with scalar` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests uniform AABB expansion by scalar 0.5. Starting with box [-1,-1,-1] to [1,1,1], it verifies that min becomes -1.5 and max becomes 1.5, validating both min and max move outward symmetrically.

#### 5. `expand2 with point` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests AABB expansion to include an external point (2,-2,0.5). Validates that:
- max.x expands from 1.0 to 2.0 (to include point.x)
- min.y expands from -1.0 to -2.0 (to include point.y)
- Other components remain unchanged when already containing the point

#### 6. `expand3 with AABB` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests merging two AABBs (bounding box union). Box1 is [-1,-1,-1] to [1,1,1], Box2 is [0,-2,0] to [2,0,2]. The test validates that the merged box has:
- min.y = -2.0 (from Box2)
- max.x = 2.0 (from Box2)
- max.z = 2.0 (from Box2)
This tests non-trivial overlapping boxes.

#### 7. `square` - GOOD
**Status:** Keep as-is
**Why it's good:** Tests the "squaring" operation that makes all dimensions equal to the largest dimension. Starting with box dimensions [2, 4, 1] (from min [-1,-2,-0.5] to max [1,2,0.5]), it correctly validates that all dimensions become 4.0 (the largest). This tests centering preservation while expanding to a cube.

---

## ❌ TRIVIAL Tests (Need Improvement)

### Test 1: `AABB struct`

**Current Issue:** This test only assigns values to the struct and reads them back with basic arithmetic (size and volume calculation), but these are trivial operations that don't test any AABB-specific functionality.

**Current Code:**
```typescript
test("AABB struct", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -2.0, -3.0);
      box.max = vec3f(1.0, 2.0, 3.0);
      // Compute volume and size to test struct functionality
      let size = box.max - box.min;
      let volume = size.x * size.y * size.z;
      test::results[0] = vec3f(size.x, size.y, volume);
    }
  `;
  const result = await testCompute(src, "vec3f");
  // Size should be (2, 4, 6), volume = 2*4*6 = 48
  expectCloseTo([2.0, 4.0, 48.0], result);
});
```

**Problem:** This is just testing struct field access and basic arithmetic. It doesn't validate any meaningful AABB behavior.

---

## Recommendations

### Recommended: Option 1 - Remove the trivial test entirely
Since the struct definition is so simple (just two `vec3` fields), and all other tests implicitly validate that the struct works correctly, we should simply **remove this test**. The struct is tested indirectly by every other test in the file.

**Reasoning:**
- The AABB struct has no methods or complex initialization
- Every other test creates and uses AABB structs successfully
- The struct test doesn't add meaningful coverage beyond "can we assign fields?"
- Per test-review.md guidance: prefer fewer, more meaningful tests over trivial ones

### Alternative: Option 2 - Test boundary edge cases
If we want to keep a struct-focused test, make it test edge cases of AABB creation and validation:

```typescript
test("AABB struct - edge cases", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;

    @compute @workgroup_size(1)
    fn foo() {
      // Test 1: Inverted box (min > max) - invalid state
      var inverted: AABB;
      inverted.min = vec3f(1.0, 2.0, 3.0);
      inverted.max = vec3f(-1.0, -2.0, -3.0);
      let isInverted = any(greaterThan(inverted.min, inverted.max));

      // Test 2: Point box (min == max) - zero volume
      var point: AABB;
      point.min = vec3f(5.0, 5.0, 5.0);
      point.max = vec3f(5.0, 5.0, 5.0);
      let diagonal = point.max - point.min;
      let isPoint = all(equal(diagonal, vec3f(0.0)));

      // Test 3: Valid box with negative coordinates
      var negBox: AABB;
      negBox.min = vec3f(-10.0, -20.0, -30.0);
      negBox.max = vec3f(-5.0, -10.0, -15.0);
      let size = negBox.max - negBox.min;
      let validSize = all(greaterThan(size, vec3f(0.0)));

      test::results[0] = vec4f(
        select(0.0, 1.0, isInverted),
        select(0.0, 1.0, isPoint),
        select(0.0, 1.0, validSize),
        size.x
      );
    }
  `;
  const result = await testCompute(src, "vec4f");
  // isInverted=1, isPoint=1, validSize=1, size.x=5.0
  expectCloseTo([1.0, 1.0, 1.0, 5.0], result);
});
```

**What this tests:**
- **Inverted box detection:** Tests that we can identify invalid AABBs where min > max
- **Degenerate case:** Tests a point-sized box (zero volume)
- **Negative coordinates:** Validates that AABBs work correctly with negative coordinate spaces
- **Size calculation:** Verifies size is positive for valid boxes

### ~~Option 3: Replace with missing `intersect` test~~ (NOT APPLICABLE)

**Note:** The `intersect` function exists in GLSL (`geometry/aabb/intersect.glsl`) but **has not been ported to WESL yet**. Therefore, we cannot add tests for it at this time.

Once `intersect.wesl` is created, we should add comprehensive tests for ray-box intersection including:
- Ray passing through box (hit detection)
- Ray missing box (tNear > tFar)
- Ray starting inside box (negative tNear)

---

## Recommended Action

**I recommend Option 1: Remove the trivial AABB struct test**

**Rationale:**
1. The struct test is genuinely trivial and adds no value
2. Per test-review.md guidance: "prefer fewer, more meaningful tests over comprehensive coverage"
3. The AABB struct has no complex initialization or methods - it's just two `vec3` fields
4. All other tests implicitly validate that the struct works correctly
5. Removing it brings the test suite to **100% meaningful test coverage** (7 tests, all good)

**Implementation priority:** Low - The test suite is already excellent (87.5% good tests). This is a minor cleanup.

---

## Additional Test Opportunities (Optional)

While the current test suite is quite good, here are optional additions for comprehensive coverage:

### 1. Add `contain` boundary test
The current `contain` test validates interior and exterior points, but not boundary points. GLSL source shows it uses `lessThan` for min (exclusive) and `lessThanEqual` for max (inclusive):

```typescript
test("contain - boundary cases", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::contain::contain;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(0.0, 0.0, 0.0);
      box.max = vec3f(1.0, 1.0, 1.0);

      // Test boundary points
      let onMin = contain(box, vec3f(0.0, 0.5, 0.5));  // On min edge - should be FALSE (exclusive)
      let onMax = contain(box, vec3f(1.0, 0.5, 0.5));  // On max edge - should be TRUE (inclusive)
      let corner = contain(box, vec3f(1.0, 1.0, 1.0)); // On max corner - should be TRUE
      let justInside = contain(box, vec3f(0.001, 0.5, 0.5)); // Just inside min - should be TRUE

      test::results[0] = vec4f(
        select(0.0, 1.0, onMin),
        select(0.0, 1.0, onMax),
        select(0.0, 1.0, corner),
        select(0.0, 1.0, justInside)
      );
    }
  `;
  const result = await testCompute(src, "vec4f");
  // onMin=0 (false), onMax=1 (true), corner=1 (true), justInside=1 (true)
  expectCloseTo([0.0, 1.0, 1.0, 1.0], result);
});
```

This tests the **asymmetric boundary behavior** where min is exclusive but max is inclusive.

### 2. Add `centroid` asymmetric box test
Current test uses a perfectly symmetric box. Add test with asymmetric box:

```typescript
test("centroid - asymmetric box", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::centroid::centroid;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -3.0, -5.0);
      box.max = vec3f(3.0, 1.0, 5.0);
      let result = centroid(box);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec3f");
  // Centroid should be at (1.0, -1.0, 0.0)
  expectCloseTo([1.0, -1.0, 0.0], result);
});
```

### 3. Add `square` with already-square box
Test that squaring an already-square box is idempotent:

```typescript
test("square - already square box", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::square::square;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-2.0, -2.0, -2.0);
      box.max = vec3f(2.0, 2.0, 2.0);
      square(&box);
      let diag = box.max - box.min;
      // Also verify center is preserved
      let center = (box.min + box.max) * 0.5;
      test::results[0] = vec4f(diag.x, diag.y, diag.z, center.x);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // All dimensions should remain 4.0, center should remain at 0.0
  expectCloseTo([4.0, 4.0, 4.0, 0.0], result);
});
```

---

## Summary of Changes

### Immediate Action (Priority: Low):
- **Remove** trivial "AABB struct" test (or replace with edge cases test if desired)

### Optional Additions (Priority: Low):
- Add boundary case test for `contain` function (tests exclusive min, inclusive max)
- Add asymmetric box test for `centroid` function (verifies non-centered boxes)
- Add idempotent test for `square` function (verifies already-square boxes remain unchanged)

### Future Work:
- Once `intersect.wesl` is ported from GLSL, add comprehensive ray-box intersection tests

### Final Test Count:
- Current: 8 tests (7 good, 1 trivial) = 87.5% meaningful
- After removing struct test: 7 tests (7 good, 0 trivial) = 100% meaningful ✅
- With optional additions: 10 tests (all good, comprehensive coverage)

---

## Implementation Notes

1. All tests should use `testCompute` with shader validation
2. Use `expectCloseTo` for floating-point comparisons with default tolerance
3. Tests should include clear comments explaining the expected values
4. Each test should verify specific mathematical properties, not just "doesn't crash"
5. Prefer testing mathematical invariants (e.g., idempotent operations, inverse operations) when possible

---

## Conclusion

This test suite is already excellent with 87.5% good coverage (7 good tests, 1 trivial test). The recommended improvement is to **remove the trivial AABB struct test**, which would bring the suite to **100% meaningful test coverage** (7 good tests).

All core AABB functionality is well-tested:
- ✅ Centroid calculation
- ✅ Point containment
- ✅ Diagonal/size calculation
- ✅ Expansion (scalar, point, AABB)
- ✅ Square operation

The only AABB function not yet covered is `intersect` (ray-box intersection), which exists in GLSL but **has not been ported to WESL yet**. Once ported, comprehensive tests should be added for it.
