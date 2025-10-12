# Test Plan: Animation Easing Functions

## Summary

**File:** `test/wesl/animation-easing.test.ts`

**Total Tests:** 32
- **Good Tests:** 29 (90.6%)
- **Trivial Tests:** 3 (9.4%)

The animation-easing test suite is in excellent shape overall. Most tests validate non-trivial mathematical behavior at meaningful input points. However, **three linear easing tests are trivial** because they only test three values (0.0, 0.5, 1.0) of an identity function.

**Updated Guidance (from test-review.md):**
Following the new testing guidance, the improved linear tests should:
- Test **specific values** with **expected outputs** (e.g., 0.0, 0.25, 0.5, 0.75, 1.0)
- Use more data points to establish the linear pattern
- Follow the same structure as other easing tests (backIn, cubicIn, etc.)
- Avoid complex slope calculations - instead verify the identity property f(t) = t directly

---

## Tests Requiring Improvement

### 1. linearIn (Line 187-199) - TRIVIAL ⚠️

**Current Implementation:**
```typescript
test("linearIn", async () => {
  const src = `
    import lygia::animation::easing::linearIn::linearIn;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = linearIn(0.0);
      test::results[1] = linearIn(0.5);
      test::results[2] = linearIn(1.0);
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0, 0.5, 1.0], result);
});
```

**Why It's Trivial:**
The `linearIn` function is defined as `float linearIn(in float t) { return t; }`. This is a pure identity function, so the test is just verifying pass-through behavior (0.0→0.0, 0.5→0.5, 1.0→1.0).

**Improvement Plan:**

Following the guidance in `test-review.md`, we should test specific values that verify the identity property AND the mathematical property that distinguishes linear easing from other functions (constant slope). The test should use multiple specific input values with their expected outputs.

**Recommended Test:**
```typescript
test("linearIn", async () => {
  const src = `
    import lygia::animation::easing::linearIn::linearIn;
    @compute @workgroup_size(1)
    fn foo() {
      // Test identity property at specific values
      test::results[0] = linearIn(0.0);
      test::results[1] = linearIn(0.25);
      test::results[2] = linearIn(0.5);
      test::results[3] = linearIn(0.75);
      test::results[4] = linearIn(1.0);
    }
  `;
  const result = await testCompute(src);

  // Verify specific expected values (identity function)
  expectCloseTo([0.0, 0.25, 0.5, 0.75, 1.0], result);
});
```

This improved test:
- Uses **specific value tests** with concrete expected outputs (0.25, 0.75, etc.)
- Tests more points to establish the linear pattern
- Verifies the identity property f(t) = t at multiple points
- Follows the same pattern as other easing tests (backIn, cubicIn, etc.)

---

### 2. linearOut (Line 203-215) - TRIVIAL ⚠️

**Current Implementation:**
```typescript
test("linearOut", async () => {
  const src = `
    import lygia::animation::easing::linearOut::linearOut;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = linearOut(0.0);
      test::results[1] = linearOut(0.5);
      test::results[2] = linearOut(1.0);
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0, 0.5, 1.0], result);
});
```

**Why It's Trivial:**
Same as `linearIn` - it's an identity function, so the test is just pass-through verification.

**Improvement Plan:**
Same approach as `linearIn` - test specific values with expected outputs to verify the identity property.

**Recommended Test:**
```typescript
test("linearOut", async () => {
  const src = `
    import lygia::animation::easing::linearOut::linearOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test identity property at specific values
      test::results[0] = linearOut(0.0);
      test::results[1] = linearOut(0.25);
      test::results[2] = linearOut(0.5);
      test::results[3] = linearOut(0.75);
      test::results[4] = linearOut(1.0);
    }
  `;
  const result = await testCompute(src);

  // Verify specific expected values (identity function)
  expectCloseTo([0.0, 0.25, 0.5, 0.75, 1.0], result);
});
```

This improved test:
- Uses **specific value tests** with concrete expected outputs
- Tests multiple points to establish the linear pattern
- Verifies the identity property f(t) = t at multiple points
- Maintains consistency with the linearIn test structure

---

### 3. linearInOut (Line 219-231) - TRIVIAL ⚠️

**Current Implementation:**
```typescript
test("linearInOut", async () => {
  const src = `
    import lygia::animation::easing::linearInOut::linearInOut;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = linearInOut(0.0);
      test::results[1] = linearInOut(0.5);
      test::results[2] = linearInOut(1.0);
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0, 0.5, 1.0], result);
});
```

**Why It's Trivial:**
Same as above - identity function with pass-through verification.

**Improvement Plan:**
Same approach as the other linear tests - test specific values with expected outputs to verify the identity property.

**Recommended Test:**
```typescript
test("linearInOut", async () => {
  const src = `
    import lygia::animation::easing::linearInOut::linearInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test identity property at specific values
      test::results[0] = linearInOut(0.0);
      test::results[1] = linearInOut(0.25);
      test::results[2] = linearInOut(0.5);
      test::results[3] = linearInOut(0.75);
      test::results[4] = linearInOut(1.0);
    }
  `;
  const result = await testCompute(src);

  // Verify specific expected values (identity function)
  expectCloseTo([0.0, 0.25, 0.5, 0.75, 1.0], result);
});
```

This improved test:
- Uses **specific value tests** with concrete expected outputs
- Tests multiple points to establish the linear pattern
- Verifies the identity property f(t) = t at multiple points
- Maintains consistency with the other linear test structures

---

## Good Tests to Keep (No Changes Needed)

The following 29 tests are **already good** and should remain as-is:

### Back Easing (3 tests)
- **backIn** (Line 5-13): Tests midpoint value (-0.375) which demonstrates the overshoot behavior
- **backOut** (Line 15-23): Tests midpoint value (1.375) showing overshoot in opposite direction
- **backInOut** (Line 25-33): Tests symmetry property at t=0.5 → 0.5

### Bounce Easing (3 tests)
- **bounceIn** (Line 35-43): Tests midpoint (0.28125) demonstrating bounce behavior
- **bounceOut** (Line 45-53): Tests midpoint (0.71875) showing opposite bounce
- **bounceInOut** (Line 55-63): Tests symmetry at t=0.5 → 0.5

### Circular Easing (3 tests)
- **circularIn** (Line 65-73): Tests sqrt-based curve at midpoint (0.134)
- **circularOut** (Line 75-83): Tests inverse circular curve (0.866)
- **circularInOut** (Line 85-93): Tests symmetry at t=0.5 → 0.5

### Cubic Easing (3 tests)
- **cubicIn** (Line 95-103): Tests t³ at midpoint (0.5³ = 0.125)
- **cubicOut** (Line 105-113): Tests 1-(1-t)³ at midpoint (0.875)
- **cubicInOut** (Line 115-123): Tests symmetry at t=0.5 → 0.5

### Elastic Easing (3 tests)
- **elasticIn** (Line 125-133): Tests spring oscillation (-0.022097)
- **elasticOut** (Line 135-143): Tests spring overshoot (1.022097)
- **elasticInOut** (Line 145-153): Tests symmetry at t=0.5 → 0.5

### Exponential Easing (3 tests)
- **exponentialIn** (Line 155-163): Tests 2^(-10t) at midpoint (0.03125)
- **exponentialOut** (Line 165-173): Tests inverse exponential (0.96875)
- **exponentialInOut** (Line 175-183): Tests symmetry at t=0.5 → 0.5

### Quadratic Easing (3 tests)
- **quadraticIn** (Line 233-241): Tests t² at midpoint (0.5² = 0.25)
- **quadraticOut** (Line 243-251): Tests 1-(1-t)² at midpoint (0.75)
- **quadraticInOut** (Line 253-261): Tests symmetry at t=0.5 → 0.5

### Quartic Easing (3 tests)
- **quarticIn** (Line 263-271): Tests t⁴ at midpoint (0.5⁴ = 0.0625)
- **quarticOut** (Line 273-281): Tests inverse quartic (0.9375)
- **quarticInOut** (Line 283-291): Tests symmetry at t=0.5 → 0.5

### Quintic Easing (3 tests)
- **quinticIn** (Line 293-301): Tests t⁵ at midpoint (0.5⁵ = 0.03125)
- **quinticOut** (Line 303-311): Tests 1-(1-t)⁵ at midpoint (1.03125) - **Note:** This shows overshoot beyond 1.0, which is mathematically correct for quintic out
- **quinticInOut** (Line 313-321): Tests complex quintic blend at midpoint (1.5) - **Note:** This also shows overshoot, demonstrating extreme easing behavior

### Sine Easing (3 tests)
- **sineIn** (Line 323-331): Tests 1-cos(t*π/2) at midpoint (0.2929)
- **sineOut** (Line 333-341): Tests sin(t*π/2) at midpoint (Math.SQRT1_2 ≈ 0.707)
- **sineInOut** (Line 343-351): Tests symmetry at t=0.5 → 0.5

---

## Mathematical Properties Validated by Good Tests

The current good tests validate several important properties:

1. **Specific Mathematical Formulas**: Most tests check that the easing function produces the correct output for a given input (e.g., cubicIn(0.5) = 0.125 = 0.5³)

2. **Symmetry Properties**: All `*InOut` functions are tested at t=0.5, where they should return 0.5 due to their symmetrical blend design

3. **Overshoot Behavior**: Tests like `backIn`, `backOut`, `elasticIn`, `elasticOut`, and `quinticOut` verify that certain easing functions intentionally produce values outside [0,1]

4. **Inverse Relationships**: Tests for `*In` and `*Out` pairs validate that they are complementary functions (e.g., backOut(t) = 1 - backIn(1-t))

---

## Implementation Priority

### High Priority (Improve Now)
1. **linearIn** - Add more test values (0.25, 0.75) to establish linear pattern
2. **linearOut** - Add more test values (0.25, 0.75) to establish linear pattern
3. **linearInOut** - Add more test values (0.25, 0.75) to establish linear pattern

These three tests are the only trivial tests in the suite. The improvement is simple: expand from 3 test points to 5 test points to better verify the identity property with specific expected values.

---

## Additional Test Ideas (Optional Enhancements)

While the current tests are good, here are some optional enhancements that could be added later:

### 1. Boundary Tests
Test all easing functions at t=0 and t=1 to verify they map to 0 and 1 respectively:

```typescript
test("easing boundary conditions", async () => {
  const src = `
    import lygia::animation::easing::cubicIn::cubicIn;
    import lygia::animation::easing::backIn::backIn;
    import lygia::animation::easing::elasticIn::elasticIn;

    @compute @workgroup_size(1)
    fn foo() {
      // All *In functions should map 0→0 and 1→1
      test::results[0] = cubicIn(0.0);    // Should be 0.0
      test::results[1] = cubicIn(1.0);    // Should be 1.0
      test::results[2] = backIn(0.0);     // Should be 0.0
      test::results[3] = backIn(1.0);     // Should be 1.0
      test::results[4] = elasticIn(0.0);  // Should be 0.0
      test::results[5] = elasticIn(1.0);  // Should be 1.0
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], result);
});
```

### 2. Monotonicity Tests
Verify that easing functions are monotonically increasing (except for overshoot functions like elastic and back):

```typescript
test("monotonicity - most easing functions increase", async () => {
  const src = `
    import lygia::animation::easing::cubicIn::cubicIn;
    import lygia::animation::easing::quadraticIn::quadraticIn;

    @compute @workgroup_size(1)
    fn foo() {
      // For monotonic functions: f(a) < f(b) when a < b
      let a = 0.3;
      let b = 0.7;

      let cubic_a = cubicIn(a);
      let cubic_b = cubicIn(b);
      let quad_a = quadraticIn(a);
      let quad_b = quadraticIn(b);

      // These should all be positive (function is increasing)
      test::results[0] = cubic_b - cubic_a;  // > 0
      test::results[1] = quad_b - quad_a;    // > 0
    }
  `;
  const result = await testCompute(src);

  // All differences should be positive
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
});
```

### 3. Comparative Tests
Compare different easing families at the same input:

```typescript
test("easing comparison - different curves at t=0.5", async () => {
  const src = `
    import lygia::animation::easing::linearIn::linearIn;
    import lygia::animation::easing::quadraticIn::quadraticIn;
    import lygia::animation::easing::cubicIn::cubicIn;
    import lygia::animation::easing::quarticIn::quarticIn;

    @compute @workgroup_size(1)
    fn foo() {
      let t = 0.5;

      // Higher powers should produce smaller values at t=0.5
      test::results[0] = linearIn(t);     // 0.5
      test::results[1] = quadraticIn(t);  // 0.25
      test::results[2] = cubicIn(t);      // 0.125
      test::results[3] = quarticIn(t);    // 0.0625
    }
  `;
  const result = await testCompute(src);

  // Verify increasing ease-in effect
  expectCloseTo([0.5, 0.25, 0.125, 0.0625], result);

  // Also verify ordering
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[1]).toBeGreaterThan(result[2]);
  expect(result[2]).toBeGreaterThan(result[3]);
});
```

---

## Summary of Actions

### Required Changes (3 tests)
Update the three linear easing tests to include more specific test values:
- Change from 3 test points (0.0, 0.5, 1.0) to 5 test points (0.0, 0.25, 0.5, 0.75, 1.0)
- This establishes the linear pattern more clearly
- Maintains consistency with the testing approach used for other easing functions
- Uses specific value tests with expected outputs (per test-review.md guidance)

### Optional Enhancements (Future)
Consider adding:
- Boundary condition tests (all functions at t=0 and t=1)
- Monotonicity tests (verify functions are increasing)
- Comparative tests (compare different easing families)

### Keep As-Is (29 tests)
The remaining 29 tests are already good and validate meaningful mathematical properties. No changes needed.

---

## Conclusion

The animation-easing test suite is **90.6% complete** with only 3 trivial tests that need improvement. The improvements are straightforward:

**Change:** Add more specific test values (0.25 and 0.75) to better establish the linear pattern
**Approach:** Use specific value tests with expected outputs (following test-review.md guidance)
**Result:** Tests that clearly verify f(t) = t at multiple points, consistent with other easing tests

**Estimated Implementation Time:** 5-10 minutes to update all three linear tests.
