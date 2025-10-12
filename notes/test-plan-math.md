# Math Test Improvement Plan

## Summary of Findings

**Total Tests Analyzed:** 84 tests in `test/wesl/math.test.ts`

**Good Tests (Keep As-Is):** 73 tests (86.9%)
**Tests Needing Improvement:** 11 tests (13.1%)

The math test suite is already quite strong, with most tests following the principles in test-review.md:
- ✅ Most tests use specific value tests with expected outputs
- ✅ Most tests use meaningful inputs (not zero, not pass-through)
- ✅ Most tests verify mathematical formulas, not just ranges

However, 11 tests need improvement according to test-review.md criteria:
- **4 tests** have incorrect expected values (hammersley, atan2Custom, powFast need fixes)
- **3 tests** are range-only (aamirror, aastep, fcos need specific value validation)
- **2 tests** are trivial/pass-through (nyquist, permute need non-trivial inputs)
- **2 tests** need better validation (saturateMediump, grad4 need property tests)

---

## Tests That Need Improvement

### 1. **hammersley** (Line 123-146)
**Current Issue:** ⚠️ TRIVIAL: Zero case test doesn't validate bit reversal
The current test uses index 0, which produces (0.0, 0.0) - a trivial case that doesn't test the interesting mathematical behavior.

**Why It's Weak:**
The radical inverse (bit-reversed) part is the key algorithm, but zero input produces zero output without exercising the bit reversal logic.

**Improvement Plan:**
Test with non-zero indices to verify the actual bit reversal formula with known expected values.

```typescript
test("hammersley", async () => {
  const src = `
    import lygia::math::hammersley::hammersley;
    @compute @workgroup_size(1)
    fn foo() {
      // Test known Hammersley sequence values
      // hammersley(i, N) = (i/N, radicalInverse_VdC(i))
      // For radicalInverse_VdC:
      //   0 -> 0.0
      //   1 -> 0.5 (binary 1 -> reversed -> 0.1 binary = 0.5)
      //   2 -> 0.25 (binary 10 -> reversed -> 0.01 binary = 0.25)
      //   3 -> 0.75 (binary 11 -> reversed -> 0.11 binary = 0.75)
      //   4 -> 0.125 (binary 100 -> reversed -> 0.001 binary = 0.125)
      let h0 = hammersley(0u, 8);  // (0/8, 0.0) = (0.0, 0.0)
      let h1 = hammersley(1u, 8);  // (1/8, 0.5) = (0.125, 0.5)
      let h2 = hammersley(2u, 8);  // (2/8, 0.25) = (0.25, 0.25)
      let h3 = hammersley(3u, 8);  // (3/8, 0.75) = (0.375, 0.75)

      test::results[0] = vec4f(h0.x, h0.y, h1.x, h1.y);
      test::results[1] = vec4f(h2.x, h2.y, h3.x, h3.y);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify linear x component
  expectCloseTo([0.0, 0.125, 0.25, 0.375],
    [result[0], result[2], result[4], result[6]], 0.01);

  // Verify radical inverse y component (bit reversal)
  expectCloseTo([0.0, 0.5, 0.25, 0.75],
    [result[1], result[3], result[5], result[7]], 0.01);
});
```

---

### 2. **nyquist** (Line 148-171)
**Current Issue:** ⚠️ PASS-THROUGH: Value 0.8 with width 0.0 just returns 0.8
Tests with width=0.0 which causes pass-through behavior without testing actual filtering.

**Why It's Weak:**
Nyquist filtering has specific mathematical behavior - it should attenuate signals when width exceeds the Nyquist limit. The test with width=0.0 doesn't exercise this filtering logic.

**Improvement Plan:**
Test with meaningful width values that trigger actual filtering behavior with specific expected attenuation.

```typescript
test("nyquist", async () => {
  const src = `
    import lygia::math::nyquist::nyquist;
    @compute @workgroup_size(1)
    fn foo() {
      // nyquist filters based on frequency width:
      // - width < 0.5: no filtering (pass-through)
      // - width > 0.5: filters toward 0.5 (Nyquist limit)
      // The function attenuates signals based on how far they are from 0.5

      // Test 1: No filtering (width well below Nyquist)
      let v1 = nyquist(0.8, 0.1);  // width < 0.5, should pass through

      // Test 2: Partial filtering (width near Nyquist)
      let v2 = nyquist(0.8, 0.5);  // At Nyquist limit

      // Test 3: Heavy filtering (width above Nyquist)
      let v3 = nyquist(0.8, 1.0);  // Full filtering

      // Test 4: Value at midpoint (should always stay at 0.5)
      let v4 = nyquist(0.5, 0.8);  // 0.5 is unaffected by filtering

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // No filtering: should pass through (0.8)
  expectCloseTo([0.8], [result[0]], 0.05);

  // Partial filtering: should be between 0.5 and 0.8
  expect(result[1]).toBeGreaterThan(0.5);
  expect(result[1]).toBeLessThan(0.8);

  // Heavy filtering: should be very close to 0.5 (heavily attenuated)
  expectCloseTo([0.5], [result[2]], 0.1);

  // Midpoint invariant: 0.5 should remain 0.5
  expectCloseTo([0.5], [result[3]], 0.01);
});
```

---

### 3. **permute** (Line 226-248)
**Current Issue:** ⚠️ RANGE-ONLY
The test checks reproducibility and that different inputs produce different outputs, but only validates `result > 10.0`. It doesn't verify the actual permutation formula: `mod289(((x * 34.0) + 1.0) * x)`.

**Why It's Weak:**
The permute function is used in noise generation and has a specific mathematical formula. We should verify this formula produces the expected output.

**Improvement Plan:**
Test with known input values and verify the exact formula output.

```typescript
test("permute", async () => {
  const src = `
    import lygia::math::permute::permute;
    @compute @workgroup_size(1)
    fn foo() {
      // permute(x) = mod289(((x * 34.0) + 1.0) * x)
      // Test with hand-calculated values:

      // permute(1.0) = mod289(((1 * 34) + 1) * 1) = mod289(35) = 35
      let p1 = permute(1.0);

      // permute(10.0) = mod289(((10 * 34) + 1) * 10) = mod289(3410) = 3410 % 289 = 275
      let p2 = permute(10.0);

      // permute(100.0) = mod289(((100 * 34) + 1) * 100) = mod289(340100) = 340100 % 289 = 189
      let p3 = permute(100.0);

      // Verify reproducibility
      let p1_repeat = permute(1.0);

      test::results[0] = vec4f(p1, p2, p3, p1_repeat);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify exact formula outputs
  expectCloseTo([35.0], [result[0]], 0.1);
  expectCloseTo([275.0], [result[1]], 1.0);
  expectCloseTo([189.0], [result[2]], 1.0);

  // Verify reproducibility
  expectCloseTo([result[0]], [result[3]], 0.0001);
});
```

---

### 4. **aamirror** (Line 757-780)
**Current Issue:** ⚠️ RANGE-ONLY: Only checks output is in [0,1]
Uses fragment shader with derivatives but only validates output is in valid range. Doesn't verify the triangle wave pattern or anti-aliasing behavior.

**Why It's Weak:**
The key property of `aamirror` is producing a smooth triangle wave (0→1→0→1) with anti-aliased transitions at peaks and valleys. The test doesn't verify this mathematical behavior at all.

**Improvement Plan:**
Test the triangle wave pattern by checking values at specific positions (valleys at 0 and 1, peaks at 0.5 and 1.5).

```typescript
test("aamirror - anti-aliased triangle wave", async () => {
  const src = `
    import lygia::math::aamirror::aamirror;
    import lygia::math::mirror::mirror;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Test triangle wave pattern at different positions
      // aamirror should create: 0→1→0→1→0 (triangle wave)
      let x = pos.x / 50.0;  // Slow variation for derivatives

      // Test at valley (x=0), peak (x=0.5), valley (x=1.0), peak (x=1.5)
      let valley1_aa = aamirror(x);
      let peak1_aa = aamirror(x + 0.5);
      let valley2_aa = aamirror(x + 1.0);
      let peak2_aa = aamirror(x + 1.5);

      // Compare with non-anti-aliased versions
      let valley1_regular = mirror(x);
      let peak1_regular = mirror(x + 0.5);

      return vec4f(valley1_aa, peak1_aa, valley2_aa, peak2_aa);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Verify triangle wave pattern: valleys near 0.0, peaks near 1.0
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThan(0.2);  // Valley

  expect(result[1]).toBeGreaterThan(0.8);
  expect(result[1]).toBeLessThanOrEqual(1.0);  // Peak

  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThan(0.2);  // Valley (periodic)

  expect(result[3]).toBeGreaterThan(0.8);
  expect(result[3]).toBeLessThanOrEqual(1.0);  // Peak (periodic)
});
```

---

### 5. **aastep** (Line 782-802)
**Current Issue:** ⚠️ RANGE-ONLY: Only checks output is in [0,1]
Uses fragment shader but only validates output is in valid range. Doesn't verify the smooth transition behavior that distinguishes it from regular step.

**Why It's Weak:**
The defining property of `aastep` is providing a smooth gradient around the threshold (not a hard edge like `step`). The test doesn't verify this gradient exists.

**Improvement Plan:**
Test with values on both sides of threshold to verify smooth transition zone exists.

```typescript
test("aastep - smooth transition near threshold", async () => {
  const src = `
    import lygia::math::aastep::aastep;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 100.0;  // Very slow variation for minimal derivatives
      let threshold = 0.5;

      // Test values on both sides of threshold
      let below = aastep(threshold, x + 0.3);      // Well below (0.3 < 0.5)
      let justBelow = aastep(threshold, x + 0.48); // Just below (0.48 < 0.5)
      let justAbove = aastep(threshold, x + 0.52); // Just above (0.52 > 0.5)
      let above = aastep(threshold, x + 0.7);      // Well above (0.7 > 0.5)

      return vec4f(below, justBelow, justAbove, above);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Well below: should be close to 0.0
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThan(0.1);

  // Just below: should be in transition zone (0.1 to 0.5)
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[1]).toBeLessThan(0.6);

  // Just above: should be in transition zone (0.5 to 0.9)
  expect(result[2]).toBeGreaterThan(0.4);
  expect(result[2]).toBeLessThan(1.0);

  // Well above: should be close to 1.0
  expect(result[3]).toBeGreaterThan(0.9);
  expect(result[3]).toBeLessThanOrEqual(1.0);

  // Verify smooth gradient: justBelow < justAbove
  expect(result[1]).toBeLessThan(result[2]);
});
```

---

### 6. **fcos** (Line 804-829)
**Current Issue:** ⚠️ RANGE-ONLY: Only checks output is in [-1,1] with loose tolerance
Tests band-limited cosine at known values but only validates outputs are roughly in valid range with tolerance of 0.1, doesn't verify specific mathematical behavior.

**Why It's Weak:**
The test has overly loose tolerances and doesn't verify the band-limiting property (the key feature that prevents aliasing at high frequencies).

**Improvement Plan:**
Split into two tests: one with tighter tolerances for slow variation, one that verifies band-limiting at high frequencies.

```typescript
test("fcos - filtered cosine at known angles", async () => {
  const src = `
    import lygia::math::fcos::fcos;
    import lygia::math::consts::PI;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 1000.0;  // Very slow variation to minimize filtering

      // At very slow variation, fcos should match regular cos closely
      let at0 = fcos(x);                   // cos(~0) ≈ 1.0
      let atPi4 = fcos(x + PI * 0.25);     // cos(π/4) ≈ 0.707
      let atPi2 = fcos(x + PI * 0.5);      // cos(π/2) ≈ 0.0
      let atPi = fcos(x + PI);             // cos(π) ≈ -1.0

      return vec4f(at0, atPi4, atPi2, atPi);
    }`;

  const result = await testFragment(src, [2, 2]);

  // Verify cosine values with tighter tolerance for slow variation
  expectCloseTo([1.0], [result[0]], 0.05);      // cos(0) = 1.0
  expectCloseTo([0.707], [result[1]], 0.05);    // cos(π/4) = √2/2
  expectCloseTo([0.0], [result[2]], 0.05);      // cos(π/2) = 0.0
  expectCloseTo([-1.0], [result[3]], 0.05);     // cos(π) = -1.0
});

test("fcos - band limiting at high frequency", async () => {
  const src = `
    import lygia::math::fcos::fcos;
    import lygia::math::consts::PI;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x;  // Fast variation (per pixel)

      // At high frequency, fcos should attenuate (approach 0)
      // while regular cos would alias
      let highFreq = fcos(x * 100.0 * PI);  // Very high frequency

      // At low frequency, should still oscillate normally
      let lowFreq = fcos(x * 0.01 * PI);    // Very low frequency

      return vec4f(highFreq, lowFreq, 0.0, 1.0);
    }`;

  const result = await testFragment(src, [32, 32]);

  // High frequency should be heavily attenuated (close to 0)
  expect(Math.abs(result[0])).toBeLessThan(0.3);

  // Low frequency should have full amplitude (not attenuated)
  expect(Math.abs(result[1])).toBeGreaterThan(0.3);
});
```

---

### 7. **atan2Custom** (Line 847-871)
**Current Issue:** ⚠️ EXPECTED VALUES INCORRECT
The test validates specific angle values, but the expected values are mathematically wrong. The function normalizes to [0, 2π] via `fmod(atan2(y,x) + TAU, TAU)`, but the test expects incorrect values.

**Why It's Weak:**
The test has incorrect expected values that don't match the actual normalization formula, so it's testing against wrong answers.

**Improvement Plan:**
Calculate and use correct expected values based on the actual normalization formula.

```typescript
test("atan2Custom", async () => {
  const src = `
    import lygia::math::atan2::atan2Custom;
    import lygia::math::consts::PI;
    import lygia::math::consts::TAU;
    @compute @workgroup_size(1)
    fn foo() {
      // atan2Custom normalizes angles to [0, 2π] range
      // Formula: fmod(atan2(y, x) + TAU, TAU)
      // Note: This shifts negative angles to positive

      // Test common angles
      let angle1 = atan2Custom(1.0, 0.0);   // atan2(1,0) = π/2, normalized = π/2
      let angle2 = atan2Custom(0.0, 1.0);   // atan2(0,1) = 0, normalized = 0
      let angle3 = atan2Custom(-1.0, 0.0);  // atan2(-1,0) = -π/2, normalized = 3π/2
      let angle4 = atan2Custom(0.0, -1.0);  // atan2(0,-1) = π, normalized = π
      let angle5 = atan2Custom(1.0, 1.0);   // atan2(1,1) = π/4, normalized = π/4

      test::results[0] = vec4f(angle1, angle2, angle3, angle4);
      test::results[1] = angle5;
    }
  `;
  const result = await testCompute(src, "vec4f");

  const PI = Math.PI;
  const TAU = 2 * PI;

  // Verify normalized angles [0, 2π]
  expectCloseTo([PI/2], [result[0]], 0.01);     // π/2 ≈ 1.5708
  expectCloseTo([0.0], [result[1]], 0.01);      // 0
  expectCloseTo([3*PI/2], [result[2]], 0.01);   // 3π/2 ≈ 4.7124
  expectCloseTo([PI], [result[3]], 0.01);       // π ≈ 3.1416
  expectCloseTo([PI/4], [result[4]], 0.01);     // π/4 ≈ 0.7854

  // All angles should be in [0, 2π) range
  for (let i = 0; i < 5; i++) {
    expect(result[i]).toBeGreaterThanOrEqual(0.0);
    expect(result[i]).toBeLessThan(TAU);
  }
});
```

---

### 8. **powFast** (Line 1046-1067)
**Current Issue:** ⚠️ EXPECTED VALUES INCORRECT
The test includes detailed formula analysis but the expected values don't match the actual formula `a / ((1-b)*a + b)`. The mathematical derivation in comments is wrong.

**Why It's Weak:**
Incorrect expected values mean the test validates against wrong answers, defeating the purpose of testing.

**Improvement Plan:**
Recalculate correct expected values using the actual formula with hand-calculated examples.

```typescript
test("powFast", async () => {
  const src = `
    import lygia::math::powFast::powFast;
    @compute @workgroup_size(1)
    fn foo() {
      // powFast is a fast approximation: powFast(a, b) = a / ((1-b)*a + b)
      // This approximates pow(a, b) but is not exact

      // Test case 1: powFast(0.5, 0.5)
      // = 0.5 / ((1-0.5)*0.5 + 0.5)
      // = 0.5 / (0.5*0.5 + 0.5)
      // = 0.5 / (0.25 + 0.5)
      // = 0.5 / 0.75 = 0.6667
      // (True pow(0.5, 0.5) = 0.7071)
      let fast1 = powFast(0.5, 0.5);

      // Test case 2: powFast(0.8, 0.3)
      // = 0.8 / ((1-0.3)*0.8 + 0.3)
      // = 0.8 / (0.7*0.8 + 0.3)
      // = 0.8 / (0.56 + 0.3)
      // = 0.8 / 0.86 = 0.9302
      // (True pow(0.8, 0.3) = 0.9438)
      let fast2 = powFast(0.8, 0.3);

      // Test case 3: powFast(0.25, 0.75)
      // = 0.25 / ((1-0.75)*0.25 + 0.75)
      // = 0.25 / (0.25*0.25 + 0.75)
      // = 0.25 / (0.0625 + 0.75)
      // = 0.25 / 0.8125 = 0.3077
      // (True pow(0.25, 0.75) = 0.3536)
      let fast3 = powFast(0.25, 0.75);

      // Test edge case: powFast(1.0, x) should always be 1.0
      let edge1 = powFast(1.0, 0.5);

      test::results[0] = vec4f(fast1, fast2, fast3, edge1);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify approximation values (not exact pow, but close)
  expectCloseTo([0.6667], [result[0]], 0.01);
  expectCloseTo([0.9302], [result[1]], 0.01);
  expectCloseTo([0.3077], [result[2]], 0.01);

  // Edge case: powFast(1, x) = 1 for any x
  expectCloseTo([1.0], [result[3]], 0.01);
});
```

---

### 9. **saturateMediump** (Line 1198-1220)
**Current Issue:** ⚠️ RANGE-ONLY: Only checks result > 0
Tests with large values (65504.0) but only validates results are positive. Doesn't verify actual clamping behavior.

**Why It's Weak:**
The test is platform-dependent and ambiguous. On desktop (highp), 65504 doesn't clamp. On mobile (mediump), it might. The test doesn't clearly validate the saturate behavior [0,1].

**Improvement Plan:**
Test with values in normal range to verify standard saturate behavior: negative→0, middle→middle, >1→1.

```typescript
test("saturateMediump", async () => {
  const src = `
    import lygia::math::saturateMediump::saturateMediump;
    @compute @workgroup_size(1)
    fn foo() {
      // saturateMediump clamps to [0, min(1, MEDIUMP_FLT_MAX)]
      // MEDIUMP_FLT_MAX = 65504.0 (max value for half-float)
      // In practice, this acts like saturate() for values we care about

      // Test normal saturate behavior in safe range
      let v1 = saturateMediump(-0.5);   // Should clamp to 0
      let v2 = saturateMediump(0.5);    // Should pass through
      let v3 = saturateMediump(1.5);    // Should clamp to 1

      // Test with large values (within mediump range)
      let v4 = saturateMediump(1000.0); // Should clamp to 1 (not 1000)

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Test saturate behavior
  expectCloseTo([0.0], [result[0]], 0.01);  // Negative clamped to 0
  expectCloseTo([0.5], [result[1]], 0.01);  // Middle value unchanged
  expectCloseTo([1.0], [result[2]], 0.01);  // > 1 clamped to 1
  expectCloseTo([1.0], [result[3]], 0.01);  // Large value clamped to 1
});
```

---

### 10. **grad4** (Line 1292-1315)
**Current Issue:** ⚠️ RANGE-ONLY: Only checks reproducibility and output range
Tests gradient calculation but only validates same input produces same output and result is bounded. Doesn't verify gradient computation correctness.

**Why It's Weak:**
Just checking "doesn't crash" and "results are similar" doesn't validate the noise gradient algorithm works correctly. Need to verify different inputs produce different gradients.

**Improvement Plan:**
Add tests showing different permutation values and different positions produce meaningfully different gradient vectors.

```typescript
test("grad4 - noise gradient helper", async () => {
  const src = `
    import lygia::math::grad4::grad4;
    @compute @workgroup_size(1)
    fn foo() {
      // grad4 computes gradient vectors for 4D noise
      // It uses permutation value j and position p to generate gradients

      // Test 1: Reproducibility - same inputs give same output
      let g1a = grad4(100.0, vec4f(0.1, 0.2, 0.3, 0.4));
      let g1b = grad4(100.0, vec4f(0.1, 0.2, 0.3, 0.4));

      // Test 2: Different j values give different gradients
      let g2 = grad4(200.0, vec4f(0.1, 0.2, 0.3, 0.4));

      // Test 3: Different positions give different gradients
      let g3 = grad4(100.0, vec4f(0.5, 0.6, 0.7, 0.8));

      // Test 4: Gradient at origin
      let g4 = grad4(50.0, vec4f(0.0, 0.0, 0.0, 0.0));

      // Store x and y components for validation
      test::results[0] = vec4f(g1a.x, g1b.x, g2.x, g3.x);
      test::results[1] = vec4f(g1a.y, g2.y, g3.y, g4.x);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Reproducibility: g1a.x == g1b.x
  expectCloseTo([result[0]], [result[1]], 0.0001);

  // Different j values produce different gradients
  expect(Math.abs(result[0] - result[2])).toBeGreaterThan(0.1);

  // Different positions produce different gradients
  expect(Math.abs(result[0] - result[3])).toBeGreaterThan(0.1);

  // All gradient components should be in reasonable range
  for (let i = 0; i < 8; i++) {
    expect(Math.abs(result[i])).toBeLessThan(3.0);
  }
});
```

---

### 11. **unpack functions** (Lines 1532-1689)
**Current Issue:** ✅ MOSTLY GOOD, but could be improved
The unpack tests (unpack256, unpack8, unpack16, unpack32, unpack64, unpack128, unpackBase, unpack4) are generally good, but they don't test roundtrip behavior with corresponding pack functions.

**Why It Could Be Better:**
Testing roundtrips (pack then unpack) would validate that the encoding/decoding is actually invertible.

**Improvement Plan:**
Add roundtrip tests where applicable.

```typescript
test("pack/unpack roundtrip - multiple bases", async () => {
  const src = `
    import lygia::math::pack::pack;
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that pack/unpack are inverse operations
      let values = array<f32, 4>(0.0, 0.25, 0.5, 0.75);

      for (var i = 0; i < 4; i++) {
        let original = values[i];
        let packed = pack(original);
        let unpacked = unpack4(packed);
        test::results[i] = unpacked;
      }
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Verify roundtrip accuracy (limited by packing precision)
  expectCloseTo([0.0], [result[0]], 0.001);
  expectCloseTo([0.25], [result[1]], 0.001);
  expectCloseTo([0.5], [result[2]], 0.001);
  expectCloseTo([0.75], [result[3]], 0.001);
});
```

---

## Implementation Priority

### High Priority (Core Math Functions)
1. **permute** - Used in noise generation, should have exact formula verification
2. **hammersley** - Important for sampling, bit reversal is the interesting part
3. **atan2Custom** - Fix incorrect expected values

### Medium Priority (Anti-Aliasing Functions)
4. **aastep** - Test smooth transition property
5. **aamirror** - Verify triangle wave pattern
6. **fcos** - Test band-limiting property

### Lower Priority (Approximation Functions)
7. **powFast** - Fix formula and expected values
8. **grad4** - More thorough gradient testing
9. **nyquist** - Better frequency filtering validation
10. **saturateMediump** - Simplify test to focus on saturate behavior

### Nice to Have
11. **unpack roundtrips** - The existing tests are already good

---

## Summary and Recommendations

The math test suite is already quite strong (86.9% of tests are good) and mostly follows the principles in test-review.md. The 11 tests that need improvement fall into clear categories:

### Priority 1: Fix Incorrect Expected Values (Critical)
These tests are currently validating against **wrong answers**:
1. **hammersley** - Using trivial zero case instead of testing bit reversal
2. **atan2Custom** - Expected values don't match normalization formula
3. **powFast** - Formula analysis in comments is mathematically wrong

**Action:** Calculate correct expected values using hand calculation or reference implementation

### Priority 2: Replace Range-Only Tests with Specific Values (Important)
These tests only check output is in valid range:
4. **aamirror** - Should test triangle wave pattern (valleys/peaks at specific positions)
5. **aastep** - Should test smooth transition zone around threshold
6. **fcos** - Should split into two tests: tight values + band-limiting property

**Action:** Add specific expected values and tighten tolerances

### Priority 3: Replace Trivial Tests with Meaningful Inputs (Important)
These tests use pass-through or zero inputs:
7. **nyquist** - width=0.0 causes pass-through, should test actual filtering
8. **permute** - Only checks range, should verify permutation formula

**Action:** Use non-trivial inputs that exercise the algorithm

### Priority 4: Strengthen Property Tests (Enhancement)
These tests work but could be more thorough:
9. **saturateMediump** - Platform-dependent, should focus on standard saturate behavior
10. **grad4** - Only tests reproducibility, should verify gradients differ with different inputs
11. **unpack functions** - Good tests, could add roundtrip validation

**Action:** Add tests for mathematical properties (determinism, inversion, etc.)

### Alignment with test-review.md

After these improvements, the math test suite will fully exemplify the principles:
- ✅ **Specific value tests** with hand-calculable expected outputs
- ✅ **Meaningful inputs** that exercise actual computation logic
- ✅ **Mathematical properties** verified, not just "doesn't crash"
- ✅ **No trivial tests** (zero cases, pass-through, range-only)

See `/Users/lee/wesl/lygia/notes/test-plan-math-quat.md` for an exemplar test file (100% good tests).
