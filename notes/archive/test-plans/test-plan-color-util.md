# Test Improvement Plan: color-util.test.ts

## Summary

**File:** `/Users/lee/wesl/lygia/test/wesl/color-util.test.ts`
**Total Tests:** 42
**Good Tests:** 35 (83%)
**Needs Improvement:** 7 (17%)

## Alignment with test-review.md Guidance

### ✅ Strengths
- Most tests use **specific expected values** with `expectCloseTo`
- Many include **step-by-step calculations** in comments
- Good use of **non-trivial inputs** that exercise functions
- Several tests validate **mathematical properties**

### ⚠️ Areas Needing Improvement
After detailed re-analysis, 7 tests need enhancement:
1. `luma` (scalar) - Trivial pass-through, needs consistency test
2. `whiteBalance3` - Range-only, needs specific temperature values
3. `whiteBalance4` - Weak validation, needs tint parameter test
4. `colorDistanceLABCIE94` - Range-only, should use known value
5. `colorDistanceOKLAB` - Range-only, needs comparative test
6. `colorDistanceYCbCr` - Range-only, should test luma independence
7. `colorDistanceYPbPr` - Range-only, should test symmetry

### Breakdown by Quality

#### ✅ Good Tests (27)
Tests that validate specific mathematical behavior with meaningful inputs and expected outputs:
- luminance, luminance4
- luma, luma3, luma4
- mixOklab, mixOklab4
- brightnessContrast3, brightnessContrast4
- exposure3, exposure4
- heatmap
- hueDefault, paletteHue
- mixSpectral (excellent test with detailed validation)
- mixSpectral_linear_to_reflectance, mixSpectral_reflectance_to_xyz
- saturationMatrix
- levelsOutputRange3
- colorDistance, colorDistance4
- colorDistanceYUV
- hueShift4
- hueShiftRYB4
- vibrance4
- mixSpectral4

#### ⚠️ Needs Improvement (15)
Tests that are trivial, range-only, or don't validate actual behavior:
- colorDistance (line 36) - Currently range-only
- luma (line 534) - Pass-through test
- hueShiftRYB (line 160) - Range-only validation
- whiteBalance3 (line 286) - Range-only validation
- whiteBalance4 (line 308) - Weak validation
- colorDistanceLABCIE94 (line 406) - Range-only
- colorDistanceOKLAB (line 425) - Range-only
- colorDistanceYCbCr (line 443) - Range-only
- colorDistanceYPbPr (line 461) - Range-only
- hueShift4 (line 498) - Already good actually!
- hueShiftRYB4 (line 515) - Already good actually!
- luma (scalar version, line 534) - Trivial pass-through

---

## Detailed Improvement Plans

### 1. colorDistance (Line 36)
**Current Issue:** Range-only validation - only checks distance > 10.0
**Current Code:**
```typescript
test("colorDistance", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let color1 = vec3f(1.0, 0.0, 0.0); // Red
       let color2 = vec3f(0.0, 0.0, 1.0); // Blue
       let result = colorDistanceLAB(color1, color2);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // LAB Euclidean distance between red and blue in LAB color space
  // This is a perceptual color distance metric
  // Empirically measured: red to blue distance ~16
  expectCloseTo([16.07], result, 0.5);
});
```

**Why It's Already Good:** Wait, this test actually validates a specific expected value (16.07 ± 0.5). This is a **GOOD** test.

**Action:** ✅ Keep as-is

**Alignment with test-review.md:** Uses specific expected value, validates perceptual distance metric with meaningful colors. Exemplary test.

---

### 2. luma (scalar - Line 534)
**Current Issue:** Trivial pass-through - f32 input just returns the input unchanged
**Current Code:**
```typescript
test("luma", async () => {
  const src = `
     import lygia::color::luma::luma;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.75;
       let result = luma(value);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // f32 luma is passthrough (identity function)
  expectCloseTo([0.75], result);
});
```

**Why It's Trivial:** This is testing an identity function - input equals output. While this might be the correct behavior for the scalar overload, it doesn't test anything meaningful.

**Improvement Plan:**
- **Option A:** Remove this test entirely (scalar luma is trivial)
- **Option B:** Test that it's consistent with vec3 luma when given a grayscale color

**Recommended Code:**
```typescript
test("luma - grayscale consistency", async () => {
  const src = `
     import lygia::color::luma::luma;
     import lygia::color::luma::luma3;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.75;
       let gray = vec3f(0.75, 0.75, 0.75);

       let scalarLuma = luma(value);
       let vectorLuma = luma3(gray);

       // For grayscale, both should give same result
       test::results[0] = vec4f(scalarLuma, vectorLuma, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Grayscale color should have luma equal to its value
  expectCloseTo([0.75, 0.75], [result[0], result[1]], 0.01);
});
```

---

### 3. hueShiftRYB (Line 160)
**Current Issue:** Test validates specific expected output [1.0, 1.0, 0.0] with tolerance - this is actually GOOD
**Current Code:**
```typescript
test("hueShiftRYB", async () => {
  const src = `
     import lygia::color::hueShiftRYB::hueShiftRYB;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.0, 0.0); // Red
       let angle = 2.0944; // 120 degrees (1/3 turn) - shift red toward yellow in RYB space
       let result = hueShiftRYB(color, angle);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // RYB hue shift: Red shifted by 120° in RYB space
  // After RGB->RYB->hue shift->RGB conversion
  // Red shifted 120° in RYB color wheel goes toward yellow
  expectCloseTo([1.0, 1.0, 0.0], result, 0.15);
});
```

**Why It's Actually Good:** This validates that red shifted 120° in RYB space produces yellow [1.0, 1.0, 0.0]. This is meaningful color transformation validation.

**Action:** ✅ Keep as-is

**Alignment with test-review.md:** Uses specific expected output [1.0, 1.0, 0.0], validates RYB color wheel behavior. Good test.

---

### 4. whiteBalance3 (Line 286)
**Current Issue:** Range-only validation - only checks R > B and rough range, doesn't verify specific values
**Current Code:**
```typescript
test("whiteBalance3", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.5, 0.5, 0.5);
       let temperature = 0.1; // Warmer (shift toward yellow/orange)
       let tint = 0.0;
       let result = whiteBalance3(color, temperature, tint);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // White balance with temperature=0.1 (warmer) should shift gray toward orange/yellow
  // Positive temperature makes image warmer (more red/yellow tones)
  // Gray should become slightly yellowish: expect R slightly higher than B
  expect(result[0]).toBeGreaterThan(result[2]); // Red > Blue for warm shift
  expect(result[0]).toBeGreaterThan(0.48); // Should maintain brightness
  expect(result[0]).toBeLessThan(0.55);
});
```

**Why It Needs Improvement:** Only validates directional behavior (R>B) but not specific transformation

**Improvement Plan:** Test multiple temperature values and verify specific color temperature behavior

**Recommended Code:**
```typescript
test("whiteBalance3", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance3;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec3f(0.5, 0.5, 0.5);

       // Test warm, neutral, and cool temperatures
       let warm = whiteBalance3(gray, 0.2, 0.0);    // Warm (orange/yellow)
       let neutral = whiteBalance3(gray, 0.0, 0.0); // Neutral (unchanged)
       let cool = whiteBalance3(gray, -0.2, 0.0);   // Cool (blue)

       test::results[0] = vec4f(warm.r, neutral.g, cool.b, 0.0);
       test::results[1] = vec4f(warm.b, neutral.r, cool.r, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Warm temperature should increase red
  expect(result[0]).toBeGreaterThan(0.5); // warm.r > 0.5

  // Neutral should preserve gray
  expectCloseTo([0.5], [result[1]], 0.01); // neutral.g == 0.5

  // Cool temperature should increase blue
  const coolB = result[2];
  const warmB = result[4]; // From results[1]
  expect(coolB).toBeGreaterThan(warmB); // cool.b > warm.b

  // Warm should have higher red than cool
  const warmR = result[0];
  const coolR = result[6]; // From results[1]
  expect(warmR).toBeGreaterThan(coolR); // warm.r > cool.r
});
```

---

### 5. whiteBalance4 (Line 308)
**Current Issue:** Only validates alpha preservation and directional behavior, not specific values
**Current Code:**
```typescript
test("whiteBalance4", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.5, 0.5, 0.5, 0.8);
       let temperature = 0.1;
       let tint = 0.0;
       let result = whiteBalance4(color, temperature, tint);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");
  // Alpha should be preserved, and RGB should show warm shift like whiteBalance3
  expectCloseTo([0.8], [result[3]], 0.01); // Alpha preserved
  expect(result[0]).toBeGreaterThan(result[2]); // Red > Blue for warm shift
  expect(result[0]).toBeGreaterThan(0.48); // Should maintain brightness
});
```

**Improvement Plan:** Test tint parameter in addition to temperature

**Recommended Code:**
```typescript
test("whiteBalance4", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance4;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec4f(0.5, 0.5, 0.5, 0.8);

       // Test temperature shift (warm)
       let tempShift = whiteBalance4(gray, 0.2, 0.0);

       // Test tint shift (magenta/green)
       let tintMagenta = whiteBalance4(gray, 0.0, 0.1);  // Positive = magenta
       let tintGreen = whiteBalance4(gray, 0.0, -0.1);   // Negative = green

       test::results[0] = tempShift;
       test::results[1] = vec4f(tintMagenta.g, tintGreen.g, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Temperature shift: warm should have R > B
  expect(result[0]).toBeGreaterThan(result[2]); // tempShift.r > tempShift.b

  // Alpha preserved
  expectCloseTo([0.8], [result[3]], 0.01);

  // Tint behavior: magenta tint reduces green, green tint increases green
  const magentaG = result[4]; // From results[1][0]
  const greenG = result[5];   // From results[1][1]
  expect(greenG).toBeGreaterThan(magentaG); // green tint increases G
  expect(magentaG).toBeLessThan(0.5);       // magenta tint decreases G
  expect(greenG).toBeGreaterThan(0.5);      // green tint increases G
});
```

---

### 6. colorDistanceLABCIE94 (Line 406)
**Current Issue:** Range-only validation (2.0 < distance < 5.0)
**Current Code:**
```typescript
test("colorDistanceLABCIE94", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLABCIE94;

     @compute @workgroup_size(1)
     fn foo() {
       let green = vec3f(0.0, 1.0, 0.0);
       let yellow = vec3f(1.0, 1.0, 0.0);
       let distance = colorDistanceLABCIE94(green, yellow);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // CIE94 distance between green and yellow
  // Actual value is around 3.04 (relatively close colors)
  expect(result[0]).toBeGreaterThan(2.0);
  expect(result[0]).toBeLessThan(5.0);
});
```

**Improvement Plan:** Use the known approximate value from the comment

**Recommended Code:**
```typescript
test("colorDistanceLABCIE94", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLABCIE94;

     @compute @workgroup_size(1)
     fn foo() {
       let green = vec3f(0.0, 1.0, 0.0);
       let yellow = vec3f(1.0, 1.0, 0.0);
       let distance = colorDistanceLABCIE94(green, yellow);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // CIE94 distance between green and yellow
  // These are relatively close colors in perceptual space
  expectCloseTo([3.04], result, 0.5);
});
```

---

### 7. colorDistanceOKLAB (Line 425)
**Current Issue:** Range-only validation (0.05 < distance < 0.3)
**Current Code:**
```typescript
test("colorDistanceOKLAB", async () => {
  const src = `
     import lygia::color::distance::colorDistanceOKLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let orange = vec3f(1.0, 0.5, 0.0);
       let distance = colorDistanceOKLAB(red, orange);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // OKLAB distance between red and orange (should be relatively small)
  expect(result[0]).toBeGreaterThan(0.05);
  expect(result[0]).toBeLessThan(0.3);
});
```

**Improvement Plan:** Test comparative distances (red-orange < red-blue) to validate perceptual uniformity

**Recommended Code:**
```typescript
test("colorDistanceOKLAB", async () => {
  const src = `
     import lygia::color::distance::colorDistanceOKLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let orange = vec3f(1.0, 0.5, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);

       // Perceptual distances
       let redToOrange = colorDistanceOKLAB(red, orange);
       let redToBlue = colorDistanceOKLAB(red, blue);

       test::results[0] = vec4f(redToOrange, redToBlue, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  const redToOrange = result[0];
  const redToBlue = result[1];

  // Orange is perceptually closer to red than blue is
  expect(redToOrange).toBeLessThan(redToBlue);

  // Red-orange should be relatively small (similar hues)
  expect(redToOrange).toBeGreaterThan(0.05);
  expect(redToOrange).toBeLessThan(0.3);

  // Red-blue should be larger (opposite hues)
  expect(redToBlue).toBeGreaterThan(0.3);
});
```

---

### 8. colorDistanceYCbCr (Line 443)
**Current Issue:** Range-only validation (0.5 < distance < 1.5)
**Current Code:**
```typescript
test("colorDistanceYCbCr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYCbCr;

     @compute @workgroup_size(1)
     fn foo() {
       let color1 = vec3f(0.8, 0.2, 0.2);
       let color2 = vec3f(0.2, 0.2, 0.8);
       let distance = colorDistanceYCbCr(color1, color2);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // YCbCr chroma distance (ignores Y/luma, only uses CbCr)
  expect(result[0]).toBeGreaterThan(0.5);
  expect(result[0]).toBeLessThan(1.5);
});
```

**Improvement Plan:** Test that luma is actually ignored (two colors with same chroma but different luma should have distance ~0)

**Recommended Code:**
```typescript
test("colorDistanceYCbCr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYCbCr;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: Different chrominance (red vs blue)
       let red = vec3f(0.8, 0.2, 0.2);
       let blue = vec3f(0.2, 0.2, 0.8);
       let chromaDist = colorDistanceYCbCr(red, blue);

       // Test 2: Same chrominance, different luma (should be ~0)
       // Dark gray vs light gray (same neutral chroma)
       let darkGray = vec3f(0.3, 0.3, 0.3);
       let lightGray = vec3f(0.7, 0.7, 0.7);
       let lumaDist = colorDistanceYCbCr(darkGray, lightGray);

       test::results[0] = vec4f(chromaDist, lumaDist, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  const chromaDist = result[0];
  const lumaDist = result[1];

  // Different chrominance should produce measurable distance
  expect(chromaDist).toBeGreaterThan(0.5);
  expect(chromaDist).toBeLessThan(1.5);

  // Same chrominance (grays) should have near-zero distance
  // (YCbCr distance ignores Y/luma)
  expectCloseTo([0.0], [lumaDist], 0.05);
});
```

---

### 9. colorDistanceYPbPr (Line 461)
**Current Issue:** Range-only validation (0.5 < distance < 2.0)
**Current Code:**
```typescript
test("colorDistanceYPbPr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYPbPr;

     @compute @workgroup_size(1)
     fn foo() {
       let magenta = vec3f(1.0, 0.0, 1.0);
       let cyan = vec3f(0.0, 1.0, 1.0);
       let distance = colorDistanceYPbPr(magenta, cyan);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // YPbPr chroma distance
  expect(result[0]).toBeGreaterThan(0.5);
  expect(result[0]).toBeLessThan(2.0);
});
```

**Improvement Plan:** Test symmetry and that complementary colors have larger distance than similar colors

**Recommended Code:**
```typescript
test("colorDistanceYPbPr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYPbPr;

     @compute @workgroup_size(1)
     fn foo() {
       // Complementary colors (magenta vs cyan)
       let magenta = vec3f(1.0, 0.0, 1.0);
       let cyan = vec3f(0.0, 1.0, 1.0);
       let complementaryDist = colorDistanceYPbPr(magenta, cyan);

       // Similar colors (cyan vs blue)
       let blue = vec3f(0.0, 0.0, 1.0);
       let similarDist = colorDistanceYPbPr(cyan, blue);

       // Test symmetry
       let dist1 = colorDistanceYPbPr(magenta, cyan);
       let dist2 = colorDistanceYPbPr(cyan, magenta);

       test::results[0] = vec4f(complementaryDist, similarDist, 0.0, 0.0);
       test::results[1] = vec4f(dist1, dist2, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  const complementaryDist = result[0];
  const similarDist = result[1];
  const dist1 = result[4];
  const dist2 = result[5];

  // Complementary colors should have larger distance than similar colors
  expect(complementaryDist).toBeGreaterThan(similarDist);

  // Distance should be symmetric
  expectCloseTo([dist1], [dist2], 0.01);

  // Complementary colors should have significant distance
  expect(complementaryDist).toBeGreaterThan(0.5);
});
```

---

### 10. paletteHue (Line 196)
**Current Issue:** Actually validates specific output - this test is GOOD
**Current Code:**
```typescript
test("paletteHue", async () => {
  const src = `
     import lygia::color::palette::hue::hue;

     @compute @workgroup_size(1)
     fn foo() {
       let x = 0.5;
       let ratio = 0.333; // neon ratio (1/3)
       let result = hue(x, ratio);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // Physical hue palette at x=0.5 with ratio=1/3
  // Formula: v = abs(fmod(x + [0,1,2]*ratio, 1) * 2 - 1)
  // Then smoothstep: v*v*(3-2*v)
  // For x=0.5, ratio=1/3: [0.5, 0.833, 0.167] -> fmod -> [0.5, 0.833, 0.167]
  // -> *2-1 -> [0, 0.666, -0.666] -> abs -> [0, 0.666, 0.666] -> smoothstep
  expectCloseTo([0.0, 0.740, 0.743], result, 0.05);
});
```

**Action:** ✅ Keep as-is - this validates specific mathematical output

---

### 11. saturationMatrix (Line 328)
**Current Issue:** Actually validates specific behavior - this test is GOOD
**Current Code:**
```typescript
test("saturationMatrix", async () => {
  const src = `
     import lygia::color::saturationMatrix::saturationMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let amount = 1.5; // Increase saturation by 50%
       let mat = saturationMatrix(amount);
       // Test matrix by applying to an orange color
       let color = vec3f(0.8, 0.5, 0.3);
       let result = mat * vec4f(color, 1.0);
       test::results[0] = result.xyz;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // Saturation matrix at 1.5 should increase color saturation
  // Original: (0.8, 0.5, 0.3) -> more saturated orange
  // Luma ~0.57, with 1.5 saturation should push values further from luma
  // Expected: R increases (>0.8), G stays similar, B decreases (<0.3)
  expect(result[0]).toBeGreaterThan(0.8); // Red should increase
  expect(result[2]).toBeLessThan(0.3); // Blue should decrease
  expectCloseTo([0.95, 0.53, 0.17], result, 0.1); // Approximate expected values
});
```

**Action:** ✅ Keep as-is - this validates saturation increase with specific values

---

## Additional Tests to Consider

### Test: Color distance metric comparison
```typescript
test("colorDistance - metric comparison", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLAB;
     import lygia::color::distance::colorDistanceOKLAB;
     import lygia::color::distance::colorDistanceYUV;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let green = vec3f(0.0, 1.0, 0.0);

       // Same color pair, different metrics
       let labDist = colorDistanceLAB(red, green);
       let oklabDist = colorDistanceOKLAB(red, green);
       let yuvDist = colorDistanceYUV(red, green);

       test::results[0] = vec4f(labDist, oklabDist, yuvDist, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // All metrics should detect red-green as significantly different
  expect(result[0]).toBeGreaterThan(0.5); // LAB
  expect(result[1]).toBeGreaterThan(0.3); // OKLAB
  expect(result[2]).toBeGreaterThan(0.5); // YUV
});
```

### Test: Brightness/Contrast edge cases
```typescript
test("brightnessContrast3 - edge cases", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast3;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec3f(0.5, 0.5, 0.5);

       // Zero contrast should force everything to middle gray
       let noContrast = brightnessContrast3(vec3f(0.8, 0.3, 0.6), 0.0, 0.0);

       // High contrast should amplify differences from midpoint
       let highContrast = brightnessContrast3(vec3f(0.6, 0.5, 0.4), 0.0, 2.0);

       test::results[0] = vec4f(noContrast.r, highContrast.r, highContrast.g, highContrast.b);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Zero contrast forces to 0.5 (middle gray)
  expectCloseTo([0.5], [result[0]], 0.01);

  // High contrast (2.0): (0.6-0.5)*2+0.5=0.7, (0.5-0.5)*2+0.5=0.5, (0.4-0.5)*2+0.5=0.3
  expectCloseTo([0.7, 0.5, 0.3], [result[1], result[2], result[3]], 0.01);
});
```

### Test: Exposure stops
```typescript
test("exposure3 - photographic stops", async () => {
  const src = `
     import lygia::color::exposure::exposure3;

     @compute @workgroup_size(1)
     fn foo() {
       let midGray = vec3f(0.5, 0.5, 0.5);

       // +1 stop = 2x brighter
       let plusOne = exposure3(midGray, 1.0);

       // -1 stop = 0.5x darker
       let minusOne = exposure3(midGray, -1.0);

       // +2 stops = 4x brighter
       let plusTwo = exposure3(midGray, 2.0);

       test::results[0] = vec4f(plusOne.r, minusOne.r, plusTwo.r, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");

  // +1 stop: 0.5 * 2^1 = 1.0
  expectCloseTo([1.0], [result[0]], 0.01);

  // -1 stop: 0.5 * 2^-1 = 0.25
  expectCloseTo([0.25], [result[1]], 0.01);

  // +2 stops: 0.5 * 2^2 = 2.0 (clamped to 1.0 in some implementations)
  expect(result[2]).toBeGreaterThan(1.5);
});
```

---

## Implementation Priority

1. **High Priority** (Clear improvements with specific values):
   - luma (scalar) - Test grayscale consistency
   - whiteBalance3 - Test temperature range
   - whiteBalance4 - Test tint parameter
   - colorDistanceLABCIE94 - Use specific expected value
   - colorDistanceYCbCr - Test luma independence
   - colorDistanceYPbPr - Test symmetry and comparative distances

2. **Medium Priority** (Good tests that could be enhanced):
   - colorDistanceOKLAB - Add comparative distance test
   - Add edge case tests for brightness/contrast
   - Add photographic stops test for exposure

3. **Already Good** (No changes needed):
   - luminance, luminance4
   - luma3, luma4
   - mixOklab, mixOklab4
   - brightnessContrast3, brightnessContrast4
   - exposure3, exposure4
   - hueShiftRYB, hueShiftRYB4
   - hueShift4
   - heatmap, paletteHue, hueDefault
   - mixSpectral (excellent test!)
   - saturationMatrix
   - levelsOutputRange3
   - colorDistance, colorDistance4
   - colorDistanceYUV
   - vibrance4
   - mixSpectral4, mixSpectral_linear_to_reflectance, mixSpectral_reflectance_to_xyz

---

## Summary

The color-util test file is generally in good shape with 64% of tests being non-trivial and meaningful. The main improvements needed are:

1. **Strengthen range-only tests** by adding specific expected values
2. **Test edge cases and properties** (symmetry, luma independence, etc.)
3. **Remove or enhance trivial tests** (scalar luma passthrough)
4. **Add comparative tests** to validate relative behavior between metrics

These improvements will increase confidence that the color utility functions work correctly across different inputs and edge cases.
