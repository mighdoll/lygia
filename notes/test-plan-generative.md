# Test Improvement Plan: generative.test.ts

## Summary of Findings

**Total Tests:** 60 tests
**Good Tests:** 59 tests (98%)
**Needs Improvement:** 1 test (2%) - random41

The generative test suite is **one of the best-designed test suites in the entire LYGIA project**. Nearly all tests validate meaningful mathematical properties beyond just ranges:
- **Determinism checks** - Same input produces same output
- **Continuity checks** - Nearby points produce similar values
- **Periodicity checks** - Periodic noise repeats correctly
- **Derivative validation** - Analytical derivatives match numerical derivatives
- **Hash properties** - Avalanche effect, component independence
- **Mathematical properties** - F1 ≤ F2 for Worley noise, tiling behavior

## What's Missing: Specific Value Tests for GLSL/WESL Parity

While the property tests are excellent, we should **add specific value tests** to verify GLSL↔WESL parity. This is the hybrid approach recommended in test-review.md:

1. **Property tests** (already present) - verify mathematical behavior
2. **Specific value tests** (missing) - verify known outputs match GLSL

### Excellent Tests (Keep As-Is)
These tests verify advanced mathematical properties:

1. **pnoise2, pnoise3, pnoise4** - Test periodicity property (noise repeats after one period)
2. **noised2, noised3** - Test analytical derivatives match numerical derivatives via finite differences
3. **random42, random43, random44** - Test hash properties: determinism, component independence, and avalanche effect
4. **worley22, worley32** - Test F1 ≤ F2 property (closest point distance ≤ second closest)
5. **srandom_tile22, srandom_tile33** - Test tiling property (points separated by tileLength produce same output)
6. **wavelet** - Tests determinism and that different phase/scale parameters produce different outputs

### Good Tests (Keep As-Is)
These tests verify fundamental properties (determinism, continuity, variation):

- **cnoise2, cnoise3, cnoise4** - Test determinism and continuity
- **snoise2, snoise3, snoise4** - Test determinism and continuity
- **snoise22, snoise33, snoise34** - Test determinism and continuity for vector outputs
- **srandom, srandom2, srandom22, srandom3, srandom33, srandom4** - Test determinism and variation
- **random, random2, random3, random4** - Test determinism and variation
- **random21, random22, random23, random31, random32, random33** - Test determinism for vector outputs
- **worley2, worley3** - Test determinism and variation
- **wavelet2, wavelet3, waveletScaled2, waveletScaled3** - Test determinism and parameter effects

### Needs Improvement
1. **random41** - Only checks range, missing determinism test

---

## Recommended Improvements

### 1. Add Specific Value Tests for GLSL/WESL Parity

**What's Missing:** While property tests are excellent, we need **specific value tests** to verify GLSL↔WESL parity. This acts as a regression test.

**Example for cnoise2:**

```typescript
test("cnoise2 - GLSL parity", async () => {
  const src = `
     import lygia::generative::cnoise::cnoise2;

     @compute @workgroup_size(1)
     fn foo() {
       // Test known values that match GLSL implementation
       let v1 = cnoise2(vec2f(0.0, 0.0));
       let v2 = cnoise2(vec2f(1.0, 2.0));
       let v3 = cnoise2(vec2f(3.5, 7.2));
       let v4 = cnoise2(vec2f(-2.0, 5.0));

       test::results[0] = vec4f(v1, v2, v3, v4);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // These values should match GLSL output exactly
  // Run GLSL version to get expected values, then add them here
  expectCloseTo([0.0], [result[0]], 3);  // origin typically returns 0
  expectCloseTo([-0.2847], [result[1]], 3); // example value from GLSL
  // Add more expected values from GLSL testing

  // Verify range
  result.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(-1.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });
});
```

**Apply this pattern to:**
- cnoise2, cnoise3, cnoise4
- snoise2, snoise3, snoise4
- random functions (random, random2, random3, random4)
- srandom functions

**Why this matters:**
- Acts as regression test for WESL implementation
- Verifies GLSL↔WESL parity with concrete values
- Property tests alone can't catch systematic biases or implementation bugs
- Combines well with existing property tests for comprehensive coverage

---

### 2. Fix random41 - Add Determinism Check

**Current Issue:** Only checks range, doesn't test determinism

**Improved Test:**

```typescript
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

  // Test determinism across runs
  expectCloseTo(result1, result2);

  // Range check
  result1.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });
});
```

---

## Implementation Plan

### Phase 1: Fix random41 (Immediate)
- Add determinism check to random41 test

### Phase 2: Add GLSL Parity Tests (When GLSL values are available)
For each noise/random function family:
1. Run GLSL version with test inputs
2. Record exact output values
3. Add specific value tests to WESL tests
4. Verify WESL outputs match GLSL outputs

**Priority order:**
1. **cnoise** (classic noise - most commonly used)
2. **snoise** (simplex noise - also very common)
3. **random** (pseudo-random - frequently used)
4. **worley** (cellular noise - specialty use)
5. **wavelet** (wavelet noise - specialty use)

### Phase 3: Document Test Coverage
Update test documentation to note:
- ✅ Property tests (determinism, continuity, etc.)
- ✅ GLSL parity tests (specific values)
- ✅ Range validation (bounds checking)

---

## Summary Statistics

**Current State:**
- **Total Tests:** 60
- **Excellent Tests:** 11 (18%) - Test advanced properties
- **Good Tests:** 48 (80%) - Test fundamental properties
- **Weak Tests:** 1 (2%) - random41
- **Approval Rate:** 98%

**After Improvements:**
- **Total Tests:** ~90-100 (adding GLSL parity tests)
- **Excellent Tests:** ~30-40 (including parity tests)
- **Good Tests:** 58-60
- **Weak Tests:** 0
- **Approval Rate:** 100%

---

## Conclusion

The generative test suite is **already excellent** with comprehensive property testing. The recommended improvements are:

1. **Add specific value tests** for GLSL/WESL parity (regression testing)
2. **Fix random41** to include determinism check
3. **Keep all existing property tests** - they are well-designed and valuable

The hybrid approach (property tests + specific value tests) provides:
- **Property tests** verify mathematical behavior (determinism, continuity, periodicity)
- **Specific value tests** verify GLSL↔WESL parity and catch implementation bugs
- **Range tests** verify bounds (already included in current tests)
