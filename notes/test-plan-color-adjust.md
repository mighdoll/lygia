# Test Improvement Plan: color-adjust.test.ts

## Summary

**Total Tests:** 35 (including blend mode tests at the end)
**Tests Analyzed for color-adjust functions:** 35
**Good Tests:** 32 (91%)
**Tests Needing Improvement:** 3 (9%)

## Alignment with test-review.md Guidance

### ✅ Strengths
- Most tests use **specific value validation** with `expectCloseTo`
- Detailed **step-by-step calculations** in comments
- Non-trivial inputs that exercise function logic
- Good coverage of edge cases

### ⚠️ Areas for Improvement
- 3 tests need enhancement to better validate behavior:
  1. `tonemapLinear3` - Should explicitly test non-clamping characteristic
  2. `ditherBlueNoise` - Should test spatial distribution properties
  3. `vibrance` - Could add complementary test showing selective saturation

The test suite is overall very strong! Most tests validate specific mathematical behavior with detailed calculations documented in comments. However, there are 5 tests that need improvement:

### Tests Needing Improvement:
1. `levels3Float` - Identity operation, returns input unchanged
2. `tonemapLinear3` - Identity operation (intentional design but test doesn't validate this)
3. `vibrance` - Only checks specific output without validating vibrance behavior
4. `ditherBayer` - Only checks range, doesn't verify dithering quantization
5. `ditherBlueNoise - core noise function` - Range-only validation

### Approved Tests (30):
All other tests are excellent - they test specific mathematical formulas with meaningful inputs and expected outputs, with detailed calculations in comments.

---

## Detailed Improvement Plans

### 1. levels3Float - Identity Operation

**Current Test (Lines 141-167):**
```typescript
test("levels3Float", async () => {
  const src = `
     import lygia::color::levels::levels3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.3, 0.5, 0.7);
       // Remap input [0.2, 0.8] to output [0.1, 0.9] with gamma 2.0
       let result = levels3Float(color, 0.2, 2.0, 0.8, 0.1, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // Step 1: inputRange: (v - 0.2) / (0.8 - 0.2) = (v - 0.2) / 0.6
  //   r: (0.3 - 0.2) / 0.6 = 0.1667
  //   g: (0.5 - 0.2) / 0.6 = 0.5
  //   b: (0.7 - 0.2) / 0.6 = 0.8333
  // Step 2: gamma: pow(v, 1/2.0) = sqrt(v)
  //   r: sqrt(0.1667) = 0.4082
  //   g: sqrt(0.5) = 0.7071
  //   b: sqrt(0.8333) = 0.9129
  // Step 3: outputRange: mix(0.1, 0.9, v) = 0.1 + v * 0.8
  //   r: 0.1 + 0.4082 * 0.8 = 0.4266
  //   g: 0.1 + 0.7071 * 0.8 = 0.6657
  //   b: 0.1 + 0.9129 * 0.8 = 0.8303
  expectCloseTo([0.4266, 0.6657, 0.8303], result);
});
```

**Problem:**
The test calculation is actually correct and non-trivial! This was marked incorrectly in earlier review. The test performs a complete levels adjustment with input range remapping, gamma correction, and output range mapping. This is a **GOOD TEST** - no changes needed.

**Status:** ✅ **Already Good** - Removed from improvement list

**Alignment with test-review.md:** This test uses specific values, shows step-by-step calculations, and validates the complete 3-stage transformation. Exemplary test.

---

### 2. tonemapLinear3 - Identity Operation

**Current Test (Lines 509-525):**
```typescript
test("tonemapLinear3", async () => {
  const src = `
     import lygia::color::tonemap::linear::tonemapLinear3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test with HDR values (>1.0) to verify passthrough behavior
       let hdr = vec3f(1.5, 2.0, 0.5);
       let result = tonemapLinear3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // Linear tonemap is identity function - HDR values pass through unchanged
  // This is intentional: no tone mapping is applied (useful as a baseline)
  expectCloseTo([1.5, 2.0, 0.5], result);
});
```

**Problem:**
The test correctly validates that `tonemapLinear3` is an identity function (no tone mapping). However, it doesn't test the *reason* this function exists - as a baseline or placeholder. The test should validate that it intentionally does NOT clamp values to [0,1] like other tonemappers.

**Improved Test:**
```typescript
test("tonemapLinear3 - identity baseline", async () => {
  const src = `
     import lygia::color::tonemap::linear::tonemapLinear3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: HDR values remain unchanged (identity function)
       let hdr = vec3f(1.5, 2.0, 0.5);
       let result1 = tonemapLinear3(hdr);

       // Test 2: Values >1.0 are NOT clamped (unlike other tonemappers)
       let bright = vec3f(5.0, 10.0, 100.0);
       let result2 = tonemapLinear3(bright);

       // Test 3: Negative values also pass through (no clamping)
       let negative = vec3f(-0.5, 0.0, 1.0);
       let result3 = tonemapLinear3(negative);

       test::results[0] = vec4f(result1.r, result2.r, result2.g, result3.r);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Verify passthrough behavior:
  // result1.r = 1.5 (unchanged)
  // result2.r = 5.0 (not clamped to 1.0)
  // result2.g = 10.0 (not clamped)
  // result3.r = -0.5 (negative preserved)
  expectCloseTo([1.5, 5.0, 10.0, -0.5], result);
});
```

**Why This Improves:**
- Tests the *characteristic* of linear tonemap (no clamping) vs other tonemappers
- Validates edge cases: very bright values, negative values
- Documents why this "trivial" function exists in the library

---

### 3. vibrance - Specific Output Without Behavior Validation

**Current Test (Lines 545-568):**
```typescript
test("vibrance", async () => {
  const src = `
     import lygia::color::vibrance::vibrance3;

     @compute @workgroup_size(1)
     fn foo() {
       // Orange color with low saturation (muted)
       let rgb = vec3f(0.6, 0.5, 0.4);
       // Increase vibrance by 0.5 (should increase saturation of muted colors)
       let result = vibrance3(rgb, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // Vibrance formula: mix(vec3(luma), color, 1.0 + (v * 1.0 - sign(v) * sat))
  // max_color = 0.6, min_color = 0.4, sat = 0.2
  // luma ≈ 0.6*0.2126 + 0.5*0.7152 + 0.4*0.0722 = 0.5141
  // mix factor = 1.0 + (0.5 * 1.0 - sign(0.5) * 0.2) = 1.0 + 0.5 - 0.2 = 1.3
  // mix(0.5141, color, 1.3) means interpolate/extrapolate
  // r: 0.5141 + (0.6 - 0.5141) * 1.3 = 0.5141 + 0.1117 = 0.6258
  // g: 0.5141 + (0.5 - 0.5141) * 1.3 = 0.5141 - 0.0183 = 0.4958
  // b: 0.5141 + (0.4 - 0.5141) * 1.3 = 0.5141 - 0.1483 = 0.3658
  expectCloseTo([0.6258, 0.4958, 0.3658], result, 0.001);
});
```

**Problem:**
Actually, this test IS good! It tests the vibrance formula with detailed calculations. The issue in test-review.md was incorrect. This validates:
- Increased saturation of muted colors
- The mathematical formula with detailed step-by-step calculation
- Non-trivial input and output

However, we could add a second test case to show the key characteristic of vibrance: it affects muted colors MORE than saturated colors.

**Improved Test (add complementary test):**
```typescript
test("vibrance - selective saturation boost", async () => {
  const src = `
     import lygia::color::vibrance::vibrance3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: Muted color (low saturation) - should change significantly
       let muted = vec3f(0.6, 0.5, 0.4);  // sat ≈ 0.2
       let muted_boosted = vibrance3(muted, 0.5);

       // Test 2: Saturated color (high saturation) - should change less
       let saturated = vec3f(1.0, 0.1, 0.0);  // sat ≈ 0.9
       let saturated_boosted = vibrance3(saturated, 0.5);

       // Test 3: Negative vibrance should desaturate
       let color = vec3f(0.8, 0.4, 0.2);
       let desaturated = vibrance3(color, -0.5);

       // Calculate saturation change for each
       let muted_sat_change = (muted_boosted.r - muted_boosted.b) / (muted.r - muted.b);
       let saturated_sat_change = (saturated_boosted.r - saturated_boosted.g) / (saturated.r - saturated.g);

       test::results[0] = vec4f(muted_sat_change, saturated_sat_change, desaturated.r, desaturated.g);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Vibrance should increase muted saturation more than saturated colors
  // muted_sat_change should be > saturated_sat_change
  expect(result[0]).toBeGreaterThan(result[1]);

  // Muted color should have increased saturation (change > 1.0)
  expect(result[0]).toBeGreaterThan(1.0);

  // Negative vibrance should move colors toward gray (desaturated)
  // For vec3f(0.8, 0.4, 0.2), luma ≈ 0.52, so desaturated should be closer to gray
  expectCloseTo([0.66, 0.46], [result[2], result[3]], 0.1);
});
```

**Why This Improves:**
- Tests the *characteristic* of vibrance: affects muted colors more than saturated ones
- Validates both positive and negative vibrance
- Shows that vibrance is "smart saturation" (selective adjustment)

**Status:** Current test is good, but add complementary test above

---

### 4. ditherBayer - Range-Only Validation

**Current Test (Lines 570-603):**
```typescript
test("ditherBayer", async () => {
  const src = `
     import lygia::color::dither::bayer::{ditherBayer, ditherBayer3Precision};

     @compute @workgroup_size(1)
     fn foo() {
       // Test Bayer dithering with quantization to 16 levels
       // Use a value between quantization levels to see dithering effect
       let color = vec3f(0.53, 0.53, 0.53); // Between 8/16 (0.5) and 9/16 (0.5625)
       let xy = vec2f(2.0, 3.0);

       // Get the Bayer threshold value for this pixel position
       let bayerValue = ditherBayer(xy);

       // Get the dithered color (quantized to 16 levels)
       let dithered = ditherBayer3Precision(color, xy, 16);

       test::results[0] = vec4f(dithered, bayerValue);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // xy=(2,3): x % 8 = 2, y % 8 = 3, index = 2 + 3*8 = 26
  // Bayer matrix[26] = 52.0/64.0 = 0.8125
  const expectedBayerValue = 0.8125;

  // For color 0.53 quantized to 16 levels:
  // decimated = floor(0.53 * 16) / 16 = floor(8.48) / 16 = 8/16 = 0.5
  // diff = (0.53 - 0.5) * 16 = 0.48
  // step(0.8125, 0.48) = 0.0 (since 0.48 < 0.8125)
  // result = decimate3(0.53 + 0.0/16, vec3(16)) = decimate3(0.53, vec3(16))
  //        = floor(0.53 * 16) / 16 = floor(8.48) / 16 = 8/16 = 0.5
  expectCloseTo([0.5, 0.5, 0.5, expectedBayerValue], result, 0.001);
});
```

**Problem:**
Actually, this test IS comprehensive! It:
- Tests the Bayer matrix value calculation
- Validates quantization behavior with detailed step-by-step calculation
- Uses a value between quantization levels to demonstrate dithering
- Verifies the threshold comparison logic

This was marked incorrectly in earlier review. The test is **GOOD** as-is.

**Status:** ✅ **Already Good** - Removed from improvement list

**Alignment with test-review.md:** Uses specific expected values (0.8125 for Bayer, 0.5 for quantized), shows detailed calculations, validates actual dithering behavior.

---

### 5. ditherBlueNoise - core noise function - Range-Only Validation

**Current Test (Lines 605-637):**
```typescript
test("ditherBlueNoise - core noise function", async () => {
  const src = `
     import lygia::color::dither::blueNoise::ditherBlueNoise;

     @compute @workgroup_size(1)
     fn foo() {
       // Test the core blue noise generation function
       // It should return a value in [0, 1] range
       let xy1 = vec2f(2.0, 3.0);
       let xy2 = vec2f(5.0, 7.0);
       let xy3 = vec2f(10.0, 15.0);

       let noise1 = ditherBlueNoise(xy1);
       let noise2 = ditherBlueNoise(xy2);
       let noise3 = ditherBlueNoise(xy3);

       test::results[0] = vec4f(noise1, noise2, noise3, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify all noise values are in valid [0, 1] range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeLessThanOrEqual(1.0);

  // Different coordinates should produce different values
  expect(result[0]).not.toBeCloseTo(result[1], 3);
  expect(result[1]).not.toBeCloseTo(result[2], 3);
});
```

**Problem:**
The test validates range and uniqueness but doesn't test the key property of blue noise: **low frequency energy** (values should be well-distributed without clumping). We should test that adjacent pixels have different values (high-frequency characteristics).

**Improved Test:**
```typescript
test("ditherBlueNoise - spatial distribution", async () => {
  const src = `
     import lygia::color::dither::blueNoise::ditherBlueNoise;

     @compute @workgroup_size(1)
     fn foo() {
       // Test spatial distribution characteristics of blue noise

       // Test 1: Adjacent pixels should have different noise values
       let noise_0_0 = ditherBlueNoise(vec2f(0.0, 0.0));
       let noise_1_0 = ditherBlueNoise(vec2f(1.0, 0.0));
       let noise_0_1 = ditherBlueNoise(vec2f(0.0, 1.0));
       let noise_1_1 = ditherBlueNoise(vec2f(1.0, 1.0));

       // Calculate variance of this 2x2 block (should be high for good distribution)
       let mean = (noise_0_0 + noise_1_0 + noise_0_1 + noise_1_1) * 0.25;
       let variance = (
         pow(noise_0_0 - mean, 2.0) +
         pow(noise_1_0 - mean, 2.0) +
         pow(noise_0_1 - mean, 2.0) +
         pow(noise_1_1 - mean, 2.0)
       ) * 0.25;

       // Test 2: Deterministic - same coordinates produce same values
       let repeat1 = ditherBlueNoise(vec2f(5.0, 7.0));
       let repeat2 = ditherBlueNoise(vec2f(5.0, 7.0));

       test::results[0] = vec4f(variance, repeat1, repeat2, noise_0_0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Variance should be reasonably high (> 0.01) for well-distributed noise
  expect(result[0]).toBeGreaterThan(0.01);

  // Deterministic: same input should produce same output
  expectCloseTo([result[1]], [result[2]], 0.0001);

  // All values should be in [0, 1] range
  expect(result[3]).toBeGreaterThanOrEqual(0.0);
  expect(result[3]).toBeLessThanOrEqual(1.0);
});
```

**Why This Improves:**
- Tests spatial distribution characteristics (key property of blue noise)
- Validates that adjacent pixels have different values (high-frequency content)
- Tests determinism (same input → same output)
- Quantifies distribution quality with variance measurement

---

## Summary of Changes

### Tests Already Good (Remove from Improvement List):
1. ✅ `levels3Float` - Comprehensive 3-stage levels adjustment (was incorrectly marked)
2. ✅ `ditherBayer` - Complete quantization validation with step-by-step math (was incorrectly marked)
3. ✅ `vibrance` - Validates formula with detailed calculations (but could add complementary test)

### Tests That Need Improvement:
1. **tonemapLinear3** - Add test for non-clamping behavior (identity characteristic)
2. **ditherBlueNoise - core noise function** - Add spatial distribution test

### Optional Enhancement:
3. **vibrance** - Add complementary test showing selective saturation boost

---

## Implementation Priority

### High Priority (Real Issues):
1. `ditherBlueNoise - core noise function` - Replace with spatial distribution test

### Medium Priority (Enhancements):
2. `tonemapLinear3` - Enhance to validate non-clamping characteristic
3. `vibrance` - Add complementary test (existing test is already good)

---

## Notes

The color-adjust test suite is **excellent overall**. Most tests have:
- Detailed mathematical calculations in comments
- Non-trivial inputs that exercise the function
- Expected outputs with step-by-step derivation
- Good coverage of edge cases

The issues identified in test-review.md were mostly incorrect for this file. Only 1-2 tests genuinely need improvement, and those are minor enhancements rather than fundamental rewrites.
