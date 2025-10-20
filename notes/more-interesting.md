# Tests to Make More Interesting

This document lists tests across the LYGIA WESL test suite that need improvement to be more mathematically interesting and meaningful. Tests are categorized by severity and type of issue.

Generated: 2025-10-20

---

## Summary Statistics

- **Total test files reviewed**: 32
- **Files with trivial tests**: 15
- **Files with excellent tests**: 17
- **Total trivial tests identified**: ~63 tests

---

## High Priority Fixes

### animation-easing.test.ts (11 trivial tests)

**Issue**: Most InOut tests only validate at t=0.5 (midpoint), which returns 0.5 due to symmetry and doesn't test the actual easing behavior. Linear tests are pure identity.

**Trivial test for Trivial function:**
These are fine as trivial tests, tested function is trivial pass through. 
1. `linearIn` - Pass-through test (returns t unchanged)
2. `linearOut` - Pass-through test (identical to linearIn)
3. `linearInOut` - Pass-through test

**Trivial tests:**
4. `backInOut` - Only tests midpoint (0.5→0.5), missing overshoot behavior
5. `bounceInOut` - Only tests midpoint, missing bounce behavior
6. `circularInOut` - Only tests midpoint, missing circular curve
7. `cubicInOut` - Only tests midpoint (inflection point)
8. `elasticInOut` - Only tests midpoint, missing elastic oscillation
9. `exponentialInOut` - Only tests midpoint, missing exponential curve
10. `quadraticInOut` - Only tests midpoint
11. `quarticInOut` - Only tests midpoint
12. `sineInOut` - Only tests midpoint

**Fix**: Test multiple points (0.0, 0.25, 0.75, 1.0) to show actual easing curves and characteristic behaviors.

---

### filter-sharpen.test.ts (5 out of 6 tests trivial)

**Issue**: Range-only validation without verifying actual sharpening kernel math.

**Trivial tests:**
1. `sharpenAdaptive` - Range-only validation, no verification of kernel computation
2. `sharpenAdaptive4` - Range-only validation, no strength parameter testing
3. `sharpenContrastAdaptive` - Only tests checker ≠ solid, no contrast adaptation validation
4. `sharpenFast` - Range-only validation, no convolution kernel verification
5. `sharpenFast4` - Same as sharpenFast

**Good test:**
- `sharpendAdaptiveControl4` - Tests exact mathematical computation

**Fix**: Create synthetic test textures with known patterns where expected output can be calculated.

---

### math-minmax.test.ts (4 out of 4 tests trivial)

**Issue**: Single simple case per function, insufficient coverage.

**Trivial tests:**
1. `mmax2` - Single case, only positive values
2. `mmax3` - Single case, only positive values
3. `mmin2` - Single case, only positive values
4. `mmin3` - Single case, only positive values

**Fix**: Test multiple cases including negative values, mixed signs, equal values, and min/max in different positions.

---

### color-dither.test.ts (4 out of 7 tests trivial)

**Issue**: Range-only validation without verifying quantization behavior.

**Trivial tests:**
1. `ditherBayerPrecision - f32` - Range-only validation, doesn't verify quantization
2. `ditherBayer3 - vec3` - Range-only validation with weak relationship checks
3. `ditherBayer - gradient banding` - Claims to test banding reduction but only checks ranges
4. `ditherBayer - quantization levels` - Vague assertions, doesn't verify actual levels

**Good tests:**
- `ditherBayer - base function` - Tests specific Bayer matrix values
- `ditherBayer - 8x8 pattern` - Tests repeating pattern property
- `ditherBayer4 - alpha preservation` - Tests specific property

**Fix**: Calculate expected quantized values for specific inputs and verify them precisely.

---

## Medium Priority Fixes

### color-blend.test.ts (7 trivial tests)

**Issue**: Pass-through cases where blend operations return base unchanged.

**Trivial tests:**
1. `blendPinLight3` - Expected result matches base input exactly
2. `blendSaturation` - Expected result matches base input exactly
3. `blendLuminosity` - Expected result matches base input exactly
4. `blendPinLight - f32` - Returns input unchanged
5. `blendPinLight3Opacity` - Pass-through makes opacity test meaningless
6. `blendSaturationOpacity` - Result matches base with overly loose tolerance (0.1)
7. `blendLuminosityOpacity` - Result matches base with overly loose tolerance (0.1)

**Fix**: Choose input values where blend operations produce visibly different outputs.

---

### math.test.ts (7 trivial tests)

**Issue**: Mix of pass-through, symmetric, identity, and basic arithmetic cases.

**Trivial tests:**
1. `absi positive` - Pass-through (abs(5) = 5)
2. `cubicMix` - Symmetric case at t=0.5 returns 0.5
3. `smootherstep` - Symmetric case at t=0.5 returns 0.5
4. `taylorInvSqrt` - Identity case (1/sqrt(1) = 1)
5. `sum2` - Basic arithmetic (3 + 7 = 10)
6. `sum3` - Basic arithmetic (3 + 7 + 5 = 15)
7. `saturateMediump` - Range-only validation

**Fix**: Test asymmetric points, non-identity values, and edge cases.

---

### animation-spriteLoop.test.ts (3 out of 3 tests trivial)

**Issue**: Range-only validation without verifying sprite frame selection formula.

**Trivial tests:**
1. `spriteLoop - index 0` - Range-only, doesn't verify frame calculation
2. `spriteLoop - index 4` - Range-only, doesn't verify modulo arithmetic
3. `spriteLoop - time wrapping` - Partially valid but incomplete

**Fix**: Test that `frame = start_index + (time % (end_index - start_index))` is correct.

---

### geometry-triangle.test.ts (3 weak tests)

**Issue**: Pass-through/duplicate tests and range-only validation.

**Trivial tests:**
1. `barycentric` - Range-only validation, unclear what overload computes
2. `barycentric2` - Duplicate of previous test with struct wrapper
3. `barycentric3 - point at vertex` - Range-only validation with inequalities

**Fix**: Test multiple cases with exact expected values, document what each overload computes.

---

### math-sampling.test.ts (3 borderline tests)

**Issue**: Range-only or property-only validation.

**Borderline tests:**
1. `grad4 - gradient range validation` - Range-only check
2. `hemisphereCosSample - unit vector property` - Only tests magnitude
3. `hemisphereCosSample - positive hemisphere` - Range-only check

**Fix**: Verify specific gradient values or combine with other meaningful tests.

---

### space.test.ts (4 borderline tests)

**Issue**: Center point cases that don't demonstrate adjustment behavior.

**Borderline tests:**
1. `ratio` - Uses center point (0.5, 0.5) producing identity output
2. `sprite` - Range-only validation, doesn't verify cell calculation
3. `tbn` - Identity basis vectors create identity matrix
4. `unratio` - Center point case

**Fix**: Test with off-center points and edge cases to show actual transformations.

---

## Low Priority Fixes

### color-tonemap.test.ts (2 trivial tests)

**Issue**: Identity function tests (tonemapLinear is explicitly identity).

**Trivial tests:**
1. `tonemapLinear3` - Pass-through by design
2. `tonemapLinear4` - Pass-through by design

**Fix**: If keeping these, test identity property with multiple diverse inputs.

---

### math-matrix.test.ts (2 trivial tests)

**Issue**: Partial validation (only diagonal elements).

**Trivial tests:**
1. `toMat3` - Only checks 3 diagonal elements out of 9
2. `toMat4` - Only checks 4 diagonal elements out of 16

**Fix**: Verify all matrix elements, not just diagonal.

---

### geometry-aabb.test.ts (1 trivial test)

**Issue**: Incomplete validation.

**Trivial test:**
1. `square` - Only checks diagonal, not centroid preservation

**Fix**: Verify centroid remains unchanged and actual min/max values are correct.

---

### math-quat.test.ts (1 trivial test)

**Issue**: Constant value check without testing behavior.

**Trivial test:**
1. `quatIdentity` - Just checks QUAT_IDENTITY constant equals [0,0,0,1]

**Fix**: Test that identity quaternion behaves as multiplicative identity.

---

## Files with Excellent Tests (No Changes Needed)

The following 17 files have excellent, non-trivial tests and serve as examples:

1. **color-util.test.ts** - Excellent mathematical precision and property tests
2. **draw.test.ts** - Tests specific mathematical behavior of stroke functions
3. **lighting-diffuse.test.ts** - Tests physical properties and mathematical relationships
4. **color-composite.test.ts** - All tests verify specific Porter-Duff formulas
5. **color-layer.test.ts** - Validates blend mode formulas with exact values
6. **color-space.test.ts** - Comprehensive color space conversion tests with roundtrips
7. **lighting-misc.test.ts** - Outstanding physics-based validation
8. **sdf.test.ts** - Validates signed distance field computations precisely
9. **color-adjust.test.ts** - Multi-step pipelines with exact validation
10. **math-pack.test.ts** - Tests specific encoding/decoding formulas
11. **math-rotate.test.ts** - Validates rotation matrices with known angles
12. **sample.test.ts** - Real-world texture sampling scenario
13. **math-easing.test.ts** - Roundtrip tests and property validation
14. **math-aa.test.ts** - Anti-aliasing properties with multiple cases
15. **math-distance.test.ts** - Uses Pythagorean triples for validation
16. **lighting-common.test.ts** - Boundary conditions and physical properties
17. **generative.test.ts** - Sophisticated statistical and property tests

---

## Common Anti-Patterns to Avoid

### 1. Pass-through or Identity Cases
```typescript
// ❌ BAD
test("identity", async () => {
  let result = identity(0.5);
  expectCloseTo([0.5], result); // Just returns input
});
```

### 2. Zero-in, Zero-out Cases
```typescript
// ❌ BAD
test("function", async () => {
  let result = someFunc(0.0);
  expectCloseTo([0.0], result); // Trivial
});
```

### 3. Range-only Validation
```typescript
// ❌ BAD
test("function", async () => {
  let result = someFunc(0.5);
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  // Doesn't verify actual computation
});
```

### 4. Symmetric Midpoint Tests for InOut Functions
```typescript
// ❌ BAD
test("easingInOut", async () => {
  let result = easingInOut(0.5);
  expectCloseTo([0.5], result); // Always 0.5 due to symmetry
});
```

---

## Best Practices for Interesting Tests

### 1. Test Specific Mathematical Behavior
```typescript
// ✅ GOOD
test("rotate2d - 90 degree rotation", async () => {
  let mat = rotate2d(HALF_PI);
  let v = vec2f(1.0, 0.0);
  let result = mat * v;
  expectCloseTo([0.0, 1.0], result); // (1,0) → (0,1)
});
```

### 2. Test Roundtrip/Inverse Operations
```typescript
// ✅ GOOD
test("invCubic", async () => {
  let x = 0.3;
  let y = cubic(x);
  let xRecovered = invCubic(y);
  expectCloseTo([x, xRecovered]); // Should match
});
```

### 3. Test Mathematical Properties
```typescript
// ✅ GOOD
test("gain", async () => {
  // gain(0.5, k) should always equal 0.5 for any k
  let result1 = gain(0.5, 2.0);
  let result2 = gain(0.5, 10.0);
  expectCloseTo([0.5, 0.5], [result1, result2]);
});
```

### 4. Multiple Test Cases Showing Function Behavior
```typescript
// ✅ GOOD
test("fmod2", async () => {
  let result1 = fmod2(vec2f(5.0, 7.0), vec2f(3.0, 4.0));
  expectCloseTo([2.0, 3.0], result1); // Positive case

  let result2 = fmod2(vec2f(-5.0, -7.0), vec2f(3.0, 4.0));
  expectCloseTo([1.0, 1.0], result2); // Negative case
});
```

---

## Implementation Priority

1. **Start with high-priority files** (animation-easing, filter-sharpen, math-minmax, color-dither)
2. **Focus on most egregious cases** (InOut functions at midpoint, range-only validation)
3. **Use excellent test files as templates** (math-easing, lighting-common, generative)
4. **Add multiple test cases per function** to show behavior across different inputs
5. **Document expected values with inline math comments** to make tests self-explanatory
