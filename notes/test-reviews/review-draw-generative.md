# WESL Test Review: draw.test.ts and generative.test.ts

## Review Date
2025-10-10

---

## draw.test.ts (1 test)

- [x] strokeEdge - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify stroke computation behavior. Test uses meaningful parameters (sdf_dist=0.5, size=0.6, width=0.2, edge=0.01) but doesn't validate expected output. Could verify stroke behavior at different distances: inside (sdf < size-width), on edge (sdf ≈ size), or outside (sdf > size).

---

## generative.test.ts (16 tests)

- [x] cnoise2 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify classic noise properties. Could test continuity (nearby points have similar values), or specific known noise characteristics.

- [x] cnoise3 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify classic noise behavior. Same issue as cnoise2.

- [x] snoise2 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify simplex noise computation. Could test that it differs from classic noise or has expected frequency characteristics.

- [x] snoise3 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify simplex noise properties. Same issue as snoise2.

- [x] pnoise2 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify periodicity. Should test that pnoise2(p, period) == pnoise2(p + period, period) to validate the periodic property.

- [x] srandom2 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify signed random properties. Could verify determinism (same input = same output) or distribution characteristics.

- [x] worley2 - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify Worley/cellular noise properties. Could test cell boundaries or compare outputs at different scales.

- [x] noised2 - ⚠️ RANGE-ONLY: Only checks noise value is in [-2,2] and that derivatives exist, doesn't verify derivative correctness. Could validate that derivatives match finite differences: (noise(p+h) - noise(p-h))/(2h).

- [x] noised3 - ⚠️ RANGE-ONLY: Only checks noise value is in [-2,2] and that derivatives exist, doesn't verify derivative computation. Same issue as noised2 - should validate derivative values.

- [x] wavelet2 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify wavelet noise characteristics. Could test band-limited properties or frequency content.

- [x] wavelet3 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify phase parameter effect. Could test that different phase values produce different outputs at same position.

- [x] waveletScaled2 - ⚠️ RANGE-ONLY: Only checks output is in [-1,1], doesn't verify scaling behavior. Should test that scale parameter affects output frequency or verify relationship to non-scaled version.

- [x] random - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify randomness properties. Could test determinism (random(1.0) always returns same value) or distribution.

- [x] random2 - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify random function behavior. Same issue as random.

- [x] random3 - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify random function properties. Same issue as random.

- [x] random4 - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify random function characteristics. Same issue as random.

---

## Summary

**draw.test.ts:**
- Total tests: 1
- Range-only tests: 1
- Trivial tests: 0
- Good tests: 0

**generative.test.ts:**
- Total tests: 16
- Range-only tests: 16
- Trivial tests: 0
- Good tests: 0

**Combined totals:**
- Total tests reviewed: 17
- Tests needing improvement: 17 (100%)
- Tests approved: 0 (0%)

---

## Recommendations

### For draw.test.ts

The `strokeEdge` test should validate actual stroke behavior:

```typescript
test("strokeEdge", async () => {
  const src = `
    import lygia::draw::stroke::strokeEdge;

    @compute @workgroup_size(1)
    fn foo() {
      // Test three cases: inside, on edge, outside
      let size = 0.5;
      let width = 0.1;
      let edge = 0.01;

      // Case 1: Inside stroke (should be 1.0)
      let inside = strokeEdge(0.4, size, width, edge);

      // Case 2: Outside stroke (should be 0.0)
      let outside = strokeEdge(0.7, size, width, edge);

      // Case 3: On outer edge (should be ~0.5, depending on antialiasing)
      let on_edge = strokeEdge(0.6, size, width, edge);

      test::results[0] = vec4f(inside, outside, on_edge, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 0.0, 0.5], [result[0], result[1], result[2]], 1);
});
```

### For generative.test.ts

**Noise functions (cnoise, snoise, pnoise, wavelet):**
- Test continuity: nearby points should have similar values
- Test determinism: same input produces same output
- For pnoise: test periodicity property

**Random functions:**
- Test determinism: verify specific known outputs
- Test that different inputs produce different outputs
- Example: `expect(random(1.0)).not.toEqual(random(2.0))`

**noised functions:**
- Validate derivatives with finite differences
- Test that derivative magnitude relates to noise variation

**Example improved test:**
```typescript
test("pnoise2 - periodicity", async () => {
  const src = `
    import lygia::generative::pnoise::pnoise2;

    @compute @workgroup_size(1)
    fn foo() {
      let period = vec2f(4.0, 4.0);
      let p = vec2f(1.0, 2.0);

      // Should be equal due to periodicity
      let noise1 = pnoise2(p, period);
      let noise2 = pnoise2(p + period, period);

      test::results[0] = vec4f(noise1, noise2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([result[0]], [result[1]]); // Periodic property
});
```

---

## Notes

All 17 tests are **range-only** tests that only verify outputs don't crash and fall within expected ranges. While these tests serve as basic smoke tests, they don't validate the mathematical correctness or characteristic properties of the functions.

Given the nature of noise and random functions (pseudo-random, complex mathematical formulas), improving these tests is challenging but important:
1. **Determinism tests** are the easiest to add (same input = same output)
2. **Periodicity/symmetry tests** work well for functions with known mathematical properties
3. **Derivative validation** can be done with finite differences for noised functions
4. **Relationship tests** between function variants (e.g., waveletScaled vs wavelet)

These improvements would make the tests much more valuable for catching regression bugs during future development.
