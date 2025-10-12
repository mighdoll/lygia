# WESL Test Review Checklist

## Purpose

This document tracks the review of all 407 unit tests in the LYGIA WESL test suite. Our goal is to ensure each test validates **non-trivial cases** rather than pass-through values or zero cases.

At this stage of development, we prefer:
- **Fewer, more meaningful tests** over comprehensive coverage
- **One or two tests per function** that validate real behavior
- **Non-trivial inputs** that exercise the function's logic

## What Makes a Test Trivial?

### ❌ Trivial Tests (Need Improvement)

1. **Pass-through or identity cases**
   ```typescript
   // Bad: Just returns the input unchanged
   test("nyquist", async () => {
     nyquist(0.8, 0.0) // expects [0.8]
   });
   ```

2. **Zero cases that don't test functionality**
   ```typescript
   // Bad: Zero input produces zero output (trivial)
   test("hammersley", async () => {
     hammersley(0u, 10) // expects [0.0, 0.0]
   });
   ```

3. **Range-only validation without computation checks**
   ```typescript
   // Bad: Only checks output is in valid range
   test("cnoise2", async () => {
     let result = cnoise2(vec2f(1.0, 2.0));
     expect(result[0]).toBeGreaterThanOrEqual(-1.0);
     expect(result[0]).toBeLessThanOrEqual(1.0);
   });
   ```

4. **Tests that only verify "doesn't crash"**
   ```typescript
   // Bad: Returns dummy value, doesn't validate behavior
   test("lookAt", async () => {
     let viewMatrix = lookAt(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0));
     test::results[0] = vec4f(1.0, 1.0, 1.0, 1.0); // ← dummy value
   });
   ```

### ✅ Non-Trivial Tests (Good!)

1. **Tests that verify specific mathematical behavior**
   ```typescript
   // Good: Tests actual rotation behavior
   test("rotate2d - 90 degree rotation", async () => {
     let mat = rotate2d(1.57079632679); // π/2 radians
     let v = vec2f(1.0, 0.0);
     let result = mat * v;
     expectCloseTo([0.0, 1.0], result); // (1,0) → (0,1)
   });
   ```

2. **Tests with meaningful inputs and expected outputs**
   ```typescript
   // Good: Tests clamping behavior with specific values
   test("saturate", async () => {
     let result = saturate(-0.5);
     expectCloseTo([0.0], result); // Negative clamped to 0
   });
   ```

3. **Roundtrip tests that verify inverse operations**
   ```typescript
   // Good: Tests that inverse function reverses original
   test("invCubic", async () => {
     let x = 0.3;
     let y = cubic(x);
     let xRecovered = invCubic(y);
     expectCloseTo([0.3, 0.3], [x, xRecovered]); // Should match
   });
   ```

4. **Tests that verify function properties or edge cases**
   ```typescript
   // Good: Tests mathematical property of gain function
   test("gain", async () => {
     // gain(0.5, k) should always equal 0.5 for any k
     test::results[0] = vec4f(gain(0.5, 2.0), gain(0.25, 2.0), gain(0.75, 2.0), 0.0);
     expect(result[0]).toBeCloseTo(0.5, 2); // Property verified
   });
   ```

5. **Tests with multiple cases showing function behavior**
   ```typescript
   // Good: Tests both positive and negative values to verify fmod behavior
   test("fmod2", async () => {
     let result1 = fmod2(vec2f(5.0, 7.0), vec2f(3.0, 4.0));
     expectCloseTo([2.0, 3.0], result1); // Positive case

     let result2 = fmod2(vec2f(-5.0, -7.0), vec2f(3.0, 4.0));
     expectCloseTo([1.0, 1.0], result2); // Negative case (floored, not truncated)
   });
   ```

## Review Instructions

For each test:

1. **Read the test code** in the corresponding test file
2. **Identify if it's trivial** using the criteria above
3. **Check the box** when reviewed (change `- [ ]` to `- [x]`)
4. **Add notes** after the checkbox if improvements are needed

### Note Format

When marking a test as needing improvement, add a note like this:

```markdown
- [x] testName - ⚠️ TRIVIAL: Just tests zero case, needs meaningful input
- [x] testName - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify computation
- [x] testName - ⚠️ PASS-THROUGH: Input equals output, no transformation tested
- [x] testName - ✅ GOOD: Tests specific mathematical behavior
```

### Special Guidelines for Noise and Random Functions

For deterministic functions like noise generators and pseudo-random functions:

**Use a hybrid approach combining:**

1. **Specific value tests** - Verify GLSL/WESL parity with known outputs
   ```typescript
   // Good: Tests that WESL produces same values as GLSL
   test("cnoise2", async () => {
     let result1 = cnoise2(vec2f(0.0, 0.0));
     expectCloseTo([0.0], [result1[0]]); // Known value at origin

     let result2 = cnoise2(vec2f(1.0, 2.0));
     expectCloseTo([-0.2847], [result2[0]]); // Known value from GLSL version
   });
   ```

2. **Property tests** - Verify mathematical behavior
   ```typescript
   // Good: Tests determinism property
   test("cnoise2 - deterministic", async () => {
     let result1 = cnoise2(vec2f(3.5, 7.2));
     let result2 = cnoise2(vec2f(3.5, 7.2));
     expectCloseTo([result1[0]], [result2[0]]); // Same input → same output
   });

   // Good: Tests continuity/smoothness
   test("cnoise2 - smoothness", async () => {
     let v1 = cnoise2(vec2f(1.0, 2.0));
     let v2 = cnoise2(vec2f(1.01, 2.01)); // Very close point
     let diff = abs(v1 - v2);
     expect(diff).toBeLessThan(0.1); // Should be smooth
   });
   ```

3. **Bounded output** (combined with other tests, not alone)
   ```typescript
   // OK when combined with specific values
   expect(result1[0]).toBeGreaterThanOrEqual(-1.0);
   expect(result1[0]).toBeLessThanOrEqual(1.0);
   ```

**Why this approach?**
- **Specific values** act as regression tests and verify GLSL↔WESL parity
- **Property tests** document expected behavior (determinism, smoothness, periodicity)
- **Avoid** range-only tests that just verify "doesn't crash"

**Examples of good property tests for noise:**
- Determinism: Same input always produces same output
- Continuity: Nearby points produce similar values
- Periodicity: `pnoise2(p, period) == pnoise2(p + period, period)`
- Symmetry: Functions with symmetric properties (if applicable)

---
