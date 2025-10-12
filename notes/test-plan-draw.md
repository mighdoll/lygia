# Test Plan: draw.test.ts

## Summary of Findings

**Total tests:** 2
- **Good tests:** 1 (50%) - `strokeEdge`
- **Needs improvement:** 1 (50%) - `stroke`

The draw test suite has minimal coverage with only two tests. The `strokeEdge` test validates real behavior with multiple distance cases using specific behavioral expectations. The `stroke` test validates a meaningful case (center of stroke) but relies on range-only validation instead of testing multiple cases with specific expected values.

---

## Test Analysis

### ✅ GOOD Tests (Keep As-Is)

#### `strokeEdge`
**Status:** Good - validates inside/outside/edge behavior with specific expectations

**Why it's good:**
- Tests three meaningful cases: inside stroke, outside stroke, and on edge
- Validates specific behavior expectations (inside > 0.9, outside < 0.1, edge in between)
- Uses the 4-parameter `strokeEdge` function which provides control over edge smoothing

**Current implementation:**
```typescript
test("strokeEdge", async () => {
  const src = `
    import lygia::draw::stroke::strokeEdge;

    @compute @workgroup_size(1)
    fn foo() {
      let size = 0.5;
      let width = 0.2;
      let edge = 0.01;

      // Case 1: Inside the stroke (distance < size-width/2)
      let inside = strokeEdge(0.45, size, width, edge);

      // Case 2: Outside the stroke (distance > size+width/2)
      let outside = strokeEdge(0.7, size, width, edge);

      // Case 3: On the outer edge transition
      let on_edge = strokeEdge(0.6, size, width, edge);

      test::results[0] = vec4f(inside, outside, on_edge, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  expect(result[0]).toBeGreaterThan(0.9);
  expect(result[1]).toBeLessThan(0.1);
  expect(result[2]).toBeGreaterThan(0.1);
  expect(result[2]).toBeLessThan(0.9);
});
```

---

## ⚠️ Tests Needing Improvement

### `stroke`

**Current Status:** ⚠️ RANGE-ONLY + SINGLE-CASE
**Issue:** Tests a meaningful case (center of stroke) but only validates range [0,1] and > 0.8, doesn't test multiple cases with specific expected values

**Current implementation:**
```typescript
test("stroke", async () => {
  const src = `
    import lygia::draw::stroke::stroke;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let sdf_dist = 0.5;
      let size = 0.5;
      let width = 0.2;
      let result = stroke(sdf_dist, size, width);
      return vec4f(result, result, result, 1.0);
    }
  `;
  const result = await testFragment(src, [2, 2]);

  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[0]).toBeGreaterThan(0.8);  // ← WEAK: Only checks "near 1.0"
});
```

**Why it needs improvement:**
1. Only tests one case (center of stroke), doesn't show stroke behavior at different distances
2. Uses range-only validation (> 0.8) instead of specific expected values
3. Comments explain expected behavior but test doesn't validate multiple behaviors
4. Per new guidance: Need specific value tests with expected outputs, not just range checks

**Note:** The test correctly uses a fragment shader since `stroke` uses `aastep` internally, which requires `fwidth` (derivatives).

**Mathematical behavior to test:**

The `stroke` function creates a ring/stroke pattern:
```
stroke(x, size, w) = aastep(size, x + w*0.5) - aastep(size, x - w*0.5)
```

- Inner edge: `size - w*0.5`
- Outer edge: `size + w*0.5`
- Maximum value (1.0) when `x = size` (center of stroke)
- Approaches 0.0 when `x < size - w*0.5` (inside) or `x > size + w*0.5` (outside)

**Improvement Plan:**

**Preferred approach: Test multiple specific cases in one test**

```typescript
test("stroke", async () => {
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

      // Case 4: At inner edge (x = size - width/2 = 0.4)
      // Expected: Transition value (~0.5 due to anti-aliasing)
      let inner_edge = stroke(0.4, size, width);

      return vec4f(center, far_inside, far_outside, inner_edge);
    }
  `;
  const result = await testFragment(src, [2, 2]);

  // Center of stroke should be close to 1.0
  expect(result[0]).toBeGreaterThan(0.9);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Outside regions (both inside and outside the stroke band) should be close to 0.0
  expect(result[1]).toBeLessThan(0.1);  // Far inside
  expect(result[2]).toBeLessThan(0.1);  // Far outside

  // Edge should be in transition range (anti-aliased)
  expect(result[3]).toBeGreaterThan(0.2);
  expect(result[3]).toBeLessThan(0.8);
});
```

**Key improvements:**
- Tests 4 specific cases showing stroke behavior: center, far inside, far outside, edge
- Each case has a clear expected behavior documented in comments
- Validates specific behavioral expectations rather than just "in range"
- Demonstrates the "band" nature of the stroke (1.0 at center, 0.0 outside the band)

**Alternative approach (not recommended):**

Splitting into multiple tests is possible but less efficient for this simple function. The combined test above is preferred because:
- All cases can fit in one vec4f output
- Fragment shader compilation overhead is reduced
- Stroke behavior is simple enough to test comprehensively in one test
- Easier to see the full behavior at a glance

If splitting is desired for documentation purposes:
- `test("stroke - center")` - validates maximum value at stroke center
- `test("stroke - outside band")` - validates zero outside the stroke region
- `test("stroke - edge transitions")` - validates anti-aliasing at boundaries

---

## Additional Test Suggestions

Currently, the draw category has minimal test coverage (only 2 tests). Per the new guidance, we prefer **fewer, more meaningful tests** over comprehensive coverage. The current tests are sufficient to validate basic functionality.

**Optional improvements (low priority):**

1. **Width variation test** - Show how stroke changes with different widths (one test with 2-3 widths)
2. **Edge parameter comparison** - Compare `strokeEdge` with different edge values to show smoothing effect
3. **Symmetry property** - Verify stroke is symmetric around the size parameter (property test)

**Not recommended:**
- Comprehensive parameter sweeps
- Redundant tests that don't add new behavioral validation
- Tests for trivial edge cases

---

## Implementation Priority

**High Priority:**
1. ✅ **Improve `stroke` test** - Replace range-only validation with multiple specific cases showing stroke behavior (center, outside regions, edge transition)

**Low Priority (optional):**
2. Width variation test - Show stroke behavior with different widths
3. Edge parameter comparison for `strokeEdge`
4. Symmetry property test

**Rationale:** Per new guidance, we prefer fewer meaningful tests. The current `strokeEdge` test is already good, and improving the `stroke` test to test multiple cases will provide sufficient coverage for this simple category.

---

## Notes

**Function behavior:**
- The `stroke` function requires a fragment shader because it uses `aastep` internally, which relies on `fwidth` (derivative function)
- The `strokeEdge` function uses `smoothstep` instead of `aastep`, providing explicit control over edge smoothing via the `edge` parameter
- Both functions create a "band" or "ring" pattern in SDF space, useful for drawing outlined shapes
- The difference of two step functions creates the stroke effect: `aastep(size, x + w/2) - aastep(size, x - w/2)`

**Testing approach (per new guidance):**
- Use specific value tests with expected outputs
- Test drawing behavior with meaningful inputs that show function characteristics
- Avoid trivial tests (zero cases, pass-through, dummy values)
- For anti-aliased functions, test: center (max value), far outside (min value), edge transitions (intermediate values)
- Prefer combining multiple cases in one test when they fit in vec4f output

---

## Checklist

- [x] Reviewed against new guidance in `test-review.md`
- [x] Updated summary to reflect new categorization (1 good, 1 needs improvement)
- [x] Clarified why `stroke` test needs improvement (single case + range-only validation)
- [x] Updated improvement plan to emphasize specific value tests with expected outputs
- [x] Simplified alternative approach section (combined test preferred)
- [x] Updated additional suggestions to align with "fewer, meaningful tests" philosophy
- [x] Adjusted priority section to reflect new guidance
- [x] Added testing approach notes from new guidance
- [x] Explained mathematical behavior to test
- [x] Provided specific improvement plans with example code
