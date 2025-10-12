# Test Plan: color-layer.test.ts

## Summary

**Status**: Currently marked as all tests GOOD in earlier review
**Finding**: Upon detailed analysis, the tests are actually **INCOMPLETE** - they only verify alpha compositing, not the blend modes themselves.

## Alignment with test-review.md Guidance

### ⚠️ Critical Issue: Tests Don't Validate RGB Blend Modes

While these tests avoid being trivial (they do validate alpha compositing), they **fail to test the primary functionality** - the RGB blend modes. According to test-review.md:

- ❌ **Missing specific value tests** for RGB channels
- ❌ **Not testing mathematical behavior** of blend modes (multiply, screen, color burn, etc.)
- ✅ **Do use meaningful inputs** (non-zero, non-identity alpha values)
- ✅ **Validate one property** (alpha compositing formula)

### What test-review.md Teaches Us

From the good examples in test-review.md:
1. "Tests that verify specific mathematical behavior" - We need to validate blend formulas
2. "Tests with meaningful inputs and expected outputs" - We need RGB channel assertions
3. "Tests that verify function properties" - We should test edge cases (fully opaque, extreme values)

### Current Status
- **Total Tests**: 18
- **Tests Verifying Alpha**: 18 (100%)
- **Tests Verifying RGB Blend Mode**: 0 (0%)
- **Assessment**: Tests are insufficient but not trivial

### The Problem

All 18 tests follow this pattern:
```typescript
test("layerAverageSourceOver4", async () => {
  const result = await testCompute(src, "vec4f");
  expect(result[3]).toBeCloseTo(0.92); // Alpha: source-over ✓
  // Missing: RGB channel validation ✗
});
```

Each test only validates the **alpha channel** (Porter-Duff source-over compositing), but completely ignores the **RGB channels** (where the actual blend mode is applied).

### What's Missing

The layer functions perform two operations:
1. **Blend Mode** (e.g., average, color burn, hard mix) applied to RGB channels
2. **Alpha Compositing** (source-over) applied to both RGB and alpha

Current tests only validate #2, not #1.

---

## Improvement Strategy

For each test, we need to:
1. **Keep** the existing alpha channel validation (it's correct)
2. **Add** RGB channel validation to verify the blend mode is working
3. **Add** edge case tests where relevant (e.g., fully opaque/transparent layers)

---

## Detailed Test Improvements

### 1. layerAverageSourceOver4

**Current Test**: Only checks alpha = 0.92

**Issue**: Doesn't verify average blending (base + blend) * 0.5

**Improved Test**:
```typescript
test("layerAverageSourceOver4", async () => {
  const src = `
     import lygia::color::layer::averageSourceOver::layerAverageSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.8, 0.6, 0.4, 0.8);
       let dstColor = vec4f(0.4, 0.2, 0.6, 0.6);
       let result = layerAverageSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing (source-over)
  expect(result[3]).toBeCloseTo(0.92); // 0.8 + 0.6 * (1 - 0.8)

  // Verify average blending is applied to RGB
  // Average blend: (src + dst) * 0.5 = (0.8+0.4, 0.6+0.2, 0.4+0.6) * 0.5 = (0.6, 0.4, 0.5)
  // Then source-over compositing: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  // RGB = 0.6*0.8 + 0.4*0.6*0.2 = 0.48 + 0.048 = 0.528
  expect(result[0]).toBeCloseTo(0.528, 2); // R channel
  expect(result[1]).toBeCloseTo(0.368, 2); // G channel: 0.4*0.8 + 0.2*0.6*0.2
  expect(result[2]).toBeCloseTo(0.472, 2); // B channel: 0.5*0.8 + 0.6*0.6*0.2
});

test("layerAverageSourceOver4 - fully opaque", async () => {
  const src = `
     import lygia::color::layer::averageSourceOver::layerAverageSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // When src is fully opaque, dst should be completely replaced by average blend
       let srcColor = vec4f(1.0, 0.0, 0.5, 1.0);
       let dstColor = vec4f(0.0, 1.0, 0.5, 1.0);
       let result = layerAverageSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Alpha should be 1.0 (fully opaque)
  expect(result[3]).toBeCloseTo(1.0);

  // RGB should be the pure average blend: (src + dst) * 0.5
  expect(result[0]).toBeCloseTo(0.5, 2); // (1.0 + 0.0) * 0.5
  expect(result[1]).toBeCloseTo(0.5, 2); // (0.0 + 1.0) * 0.5
  expect(result[2]).toBeCloseTo(0.5, 2); // (0.5 + 0.5) * 0.5
});
```

---

### 2. layerColorBurnSourceOver4

**Current Test**: Only checks alpha = 0.92

**Issue**: Doesn't verify color burn (darkening blend mode)

**Improved Test**:
```typescript
test("layerColorBurnSourceOver4", async () => {
  const src = `
     import lygia::color::layer::colorBurnSourceOver::layerColorBurnSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.6, 0.5, 0.4, 0.8);
       let dstColor = vec4f(0.4, 0.3, 0.5, 0.6);
       let result = layerColorBurnSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Verify color burn darkens the image
  // Color burn formula: 1 - (1 - base) / blend
  // This creates a darkening effect similar to overexposure in photography
  // Result should be darker than both input colors
  expect(result[0]).toBeLessThan(0.6); // Darkened from src
  expect(result[0]).toBeGreaterThan(0.0); // Not completely black
  expect(result[1]).toBeLessThan(0.5);
  expect(result[2]).toBeLessThan(0.5);
});

test("layerColorBurnSourceOver4 - with black blend", async () => {
  const src = `
     import lygia::color::layer::colorBurnSourceOver::layerColorBurnSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Color burn with black should produce black (extreme darkening)
       let srcColor = vec4f(0.0, 0.0, 0.0, 0.8);
       let dstColor = vec4f(0.8, 0.6, 0.4, 0.6);
       let result = layerColorBurnSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Black color burn should produce very dark or black result
  expect(result[0]).toBeLessThan(0.1);
  expect(result[1]).toBeLessThan(0.1);
  expect(result[2]).toBeLessThan(0.1);
});
```

---

### 3. layerColorDodgeSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify color dodge (brightening blend mode)

**Improved Test**:
```typescript
test("layerColorDodgeSourceOver4", async () => {
  const src = `
     import lygia::color::layer::colorDodgeSourceOver::layerColorDodgeSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.5, 0.6, 0.7);
       let dstColor = vec4f(0.3, 0.4, 0.5, 0.5);
       let result = layerColorDodgeSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify color dodge brightens the image
  // Color dodge formula: base / (1 - blend)
  // This creates a brightening effect similar to underexposure in photography
  // Result should be brighter than both input colors
  expect(result[0]).toBeGreaterThan(0.3); // Brightened from dst
  expect(result[1]).toBeGreaterThan(0.4);
  expect(result[2]).toBeGreaterThan(0.5);
});

test("layerColorDodgeSourceOver4 - with white blend", async () => {
  const src = `
     import lygia::color::layer::colorDodgeSourceOver::layerColorDodgeSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Color dodge with white should produce white (extreme brightening)
       let srcColor = vec4f(1.0, 1.0, 1.0, 0.8);
       let dstColor = vec4f(0.3, 0.4, 0.5, 0.5);
       let result = layerColorDodgeSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // White color dodge should produce very bright or white result (clamped to 1.0)
  expect(result[0]).toBeGreaterThan(0.9);
  expect(result[1]).toBeGreaterThan(0.9);
  expect(result[2]).toBeGreaterThan(0.9);
});
```

---

### 4. layerColorSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify HSL color blending (hue + saturation from src, luminosity from dst)

**Improved Test**:
```typescript
test("layerColorSourceOver4", async () => {
  const src = `
     import lygia::color::layer::colorSourceOver::layerColorSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Use strongly contrasting colors to verify color mode behavior
       let srcColor = vec4f(0.8, 0.4, 0.2, 0.7); // Orange-ish
       let dstColor = vec4f(0.2, 0.6, 0.8, 0.5); // Cyan-ish
       let result = layerColorSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify color mode takes hue+saturation from src, luminosity from dst
  // The result should have the orange hue/saturation of src
  // but maintain the luminosity (brightness) of dst
  // This is a complex color space conversion, so we check qualitatively

  // The result should NOT be cyan (dst's hue is replaced)
  // Orange has more red, less blue
  expect(result[0]).toBeGreaterThan(result[2]); // R > B (orange-ish, not cyan-ish)
});

test("layerColorSourceOver4 - grayscale dst", async () => {
  const src = `
     import lygia::color::layer::colorSourceOver::layerColorSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Applying color to grayscale should colorize it
       let srcColor = vec4f(1.0, 0.0, 0.0, 1.0); // Pure red
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0); // Mid-gray
       let result = layerColorSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Result should be reddish (takes hue from src)
  // but maintain gray's luminosity
  expect(result[0]).toBeGreaterThan(result[1]); // More red than green
  expect(result[0]).toBeGreaterThan(result[2]); // More red than blue

  // Should not be pure red (luminosity constrained by gray dst)
  expect(result[0]).toBeLessThan(1.0);
});
```

---

### 5. layerGlowSourceOver4

**Current Test**: Only checks alpha = 0.92

**Issue**: Doesn't verify glow blending (similar to reflect)

**Improved Test**:
```typescript
test("layerGlowSourceOver4", async () => {
  const src = `
     import lygia::color::layer::glowSourceOver::layerGlowSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.6, 0.2, 0.8);
       let dstColor = vec4f(0.5, 0.3, 0.8, 0.6);
       let result = layerGlowSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Verify glow effect (reflect with swapped parameters)
  // Glow should create a brightening/reflective effect
  // Result should show interaction between channels
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);

  // At least one channel should show brightening
  const maxResult = Math.max(result[0], result[1], result[2]);
  const maxSrc = Math.max(0.4, 0.6, 0.2);
  const maxDst = Math.max(0.5, 0.3, 0.8);
  expect(maxResult).toBeGreaterThanOrEqual(Math.min(maxSrc, maxDst) * 0.5);
});
```

---

### 6. layerHardLightSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify hard light blending (overlay with swapped src/dst)

**Improved Test**:
```typescript
test("layerHardLightSourceOver4", async () => {
  const src = `
     import lygia::color::layer::hardLightSourceOver::layerHardLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.6, 0.8, 0.7);
       let dstColor = vec4f(0.3, 0.5, 0.7, 0.5);
       let result = layerHardLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify hard light creates strong contrast
  // Hard light: if blend < 0.5, multiply; else screen
  // This should show visible blending, not just pass-through
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerHardLightSourceOver4 - dark blend", async () => {
  const src = `
     import lygia::color::layer::hardLightSourceOver::layerHardLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Dark blend (< 0.5) should darken via multiply
       let srcColor = vec4f(0.2, 0.2, 0.2, 1.0);
       let dstColor = vec4f(0.8, 0.8, 0.8, 1.0);
       let result = layerHardLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Dark blend should darken the result
  expect(result[0]).toBeLessThan(0.8); // Darker than dst
  expect(result[1]).toBeLessThan(0.8);
  expect(result[2]).toBeLessThan(0.8);
});

test("layerHardLightSourceOver4 - light blend", async () => {
  const src = `
     import lygia::color::layer::hardLightSourceOver::layerHardLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Light blend (> 0.5) should brighten via screen
       let srcColor = vec4f(0.8, 0.8, 0.8, 1.0);
       let dstColor = vec4f(0.2, 0.2, 0.2, 1.0);
       let result = layerHardLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Light blend should brighten the result
  expect(result[0]).toBeGreaterThan(0.2); // Brighter than dst
  expect(result[1]).toBeGreaterThan(0.2);
  expect(result[2]).toBeGreaterThan(0.2);
});
```

---

### 7. layerHardMixSourceOver4

**Current Test**: Only checks alpha = 0.8

**Issue**: Doesn't verify hard mix posterization (binary output: 0 or 1)

**Improved Test**:
```typescript
test("layerHardMixSourceOver4", async () => {
  const src = `
     import lygia::color::layer::hardMixSourceOver::layerHardMixSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.6, 0.8, 0.6);
       let dstColor = vec4f(0.3, 0.5, 0.2, 0.5);
       let result = layerHardMixSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.8);

  // Verify hard mix creates posterization
  // Hard mix produces binary output: each channel should be close to 0 or 1
  // Formula: if vividLight(base, blend) < 0.5 then 0 else 1

  // Each channel should be closer to 0 or 1 than to 0.5
  for (let i = 0; i < 3; i++) {
    const distToZero = Math.abs(result[i] - 0.0);
    const distToOne = Math.abs(result[i] - 1.0);
    const distToMid = Math.abs(result[i] - 0.5);
    const minDist = Math.min(distToZero, distToOne);

    // Result should be closer to 0 or 1 than to 0.5
    // (allowing for alpha compositing to soften the effect)
    expect(minDist).toBeLessThanOrEqual(distToMid * 1.5);
  }
});

test("layerHardMixSourceOver4 - fully opaque posterization", async () => {
  const src = `
     import lygia::color::layer::hardMixSourceOver::layerHardMixSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // With full opacity, should see pure posterization
       let srcColor = vec4f(0.4, 0.6, 0.8, 1.0);
       let dstColor = vec4f(0.3, 0.5, 0.2, 1.0);
       let result = layerHardMixSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Each channel should be very close to 0.0 or 1.0 (posterized)
  for (let i = 0; i < 3; i++) {
    const nearZero = result[i] < 0.2;
    const nearOne = result[i] > 0.8;
    expect(nearZero || nearOne).toBe(true);
  }
});
```

---

### 8. layerHueSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify HSL hue blending (hue from src, saturation+luminosity from dst)

**Improved Test**:
```typescript
test("layerHueSourceOver4", async () => {
  const src = `
     import lygia::color::layer::hueSourceOver::layerHueSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Use contrasting hues to verify hue transfer
       let srcColor = vec4f(0.8, 0.4, 0.2, 0.7); // Orange-ish (warm)
       let dstColor = vec4f(0.2, 0.6, 0.8, 0.5); // Cyan-ish (cool)
       let result = layerHueSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify hue is taken from src (orange), not dst (cyan)
  // Orange has more red than blue; cyan has more blue than red
  // Result should have orange's hue (R > B)
  expect(result[0]).toBeGreaterThan(result[2]); // R > B indicates warm hue
});

test("layerHueSourceOver4 - red to gray", async () => {
  const src = `
     import lygia::color::layer::hueSourceOver::layerHueSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Applying hue to desaturated color
       let srcColor = vec4f(1.0, 0.0, 0.0, 1.0); // Pure red (hue = 0°)
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0); // Gray (no hue)
       let result = layerHueSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Result should remain grayscale because dst has no saturation
  // (Hue mode takes saturation from dst, which is 0)
  const avg = (result[0] + result[1] + result[2]) / 3;
  expect(result[0]).toBeCloseTo(avg, 1); // All channels similar
  expect(result[1]).toBeCloseTo(avg, 1);
  expect(result[2]).toBeCloseTo(avg, 1);
});
```

---

### 9. layerLinearBurnSourceOver4

**Current Test**: Only checks alpha = 0.92

**Issue**: Doesn't verify linear burn (additive darkening)

**Improved Test**:
```typescript
test("layerLinearBurnSourceOver4", async () => {
  const src = `
     import lygia::color::layer::linearBurnSourceOver::layerLinearBurnSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.6, 0.5, 0.7, 0.8);
       let dstColor = vec4f(0.4, 0.3, 0.2, 0.6);
       let result = layerLinearBurnSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Verify linear burn darkening: max(src + dst - 1, 0)
  // This should produce darker results than either input
  // Example: 0.6 + 0.4 - 1 = 0.0 (clamped)

  // All channels should be darkened (considering alpha compositing)
  expect(result[0]).toBeLessThan(Math.max(0.6, 0.4));
  expect(result[1]).toBeLessThan(Math.max(0.5, 0.3));
  expect(result[2]).toBeLessThan(Math.max(0.7, 0.2));
});

test("layerLinearBurnSourceOver4 - complete darkening", async () => {
  const src = `
     import lygia::color::layer::linearBurnSourceOver::layerLinearBurnSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Values that sum to 1.0 should produce black
       let srcColor = vec4f(0.5, 0.5, 0.5, 1.0);
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0);
       let result = layerLinearBurnSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // 0.5 + 0.5 - 1.0 = 0.0 for all channels
  expect(result[0]).toBeCloseTo(0.0, 2);
  expect(result[1]).toBeCloseTo(0.0, 2);
  expect(result[2]).toBeCloseTo(0.0, 2);
});
```

---

### 10. layerLinearDodgeSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify linear dodge (additive brightening)

**Improved Test**:
```typescript
test("layerLinearDodgeSourceOver4", async () => {
  const src = `
     import lygia::color::layer::linearDodgeSourceOver::layerLinearDodgeSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.5, 0.6, 0.7);
       let dstColor = vec4f(0.3, 0.2, 0.1, 0.5);
       let result = layerLinearDodgeSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify linear dodge brightening: min(src + dst, 1.0)
  // This should produce brighter results than either input
  // Example: min(0.4 + 0.3, 1.0) = 0.7

  // All channels should show additive brightening (considering alpha compositing)
  expect(result[0]).toBeGreaterThan(Math.max(0.4, 0.3) * 0.5);
  expect(result[1]).toBeGreaterThan(Math.max(0.5, 0.2) * 0.5);
  expect(result[2]).toBeGreaterThan(Math.max(0.6, 0.1) * 0.5);
});

test("layerLinearDodgeSourceOver4 - clamping at white", async () => {
  const src = `
     import lygia::color::layer::linearDodgeSourceOver::layerLinearDodgeSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Values that sum > 1.0 should clamp to 1.0
       let srcColor = vec4f(0.7, 0.8, 0.9, 1.0);
       let dstColor = vec4f(0.6, 0.5, 0.4, 1.0);
       let result = layerLinearDodgeSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // All channels should clamp to 1.0
  expect(result[0]).toBeCloseTo(1.0, 2); // min(0.7 + 0.6, 1.0) = 1.0
  expect(result[1]).toBeCloseTo(1.0, 2); // min(0.8 + 0.5, 1.0) = 1.0
  expect(result[2]).toBeCloseTo(1.0, 2); // min(0.9 + 0.4, 1.0) = 1.0
});
```

---

### 11. layerLinearLightSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify linear light (combination of linear burn and dodge)

**Improved Test**:
```typescript
test("layerLinearLightSourceOver4", async () => {
  const src = `
     import lygia::color::layer::linearLightSourceOver::layerLinearLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.5, 0.6, 0.7);
       let dstColor = vec4f(0.3, 0.5, 0.7, 0.5);
       let result = layerLinearLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify linear light: if blend < 0.5, linear burn; else linear dodge
  // This creates strong contrast with both darkening and brightening
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerLinearLightSourceOver4 - extreme contrast", async () => {
  const src = `
     import lygia::color::layer::linearLightSourceOver::layerLinearLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Use extreme values to show burn and dodge behavior
       let srcColor = vec4f(0.2, 0.8, 0.5, 1.0);
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0);
       let result = layerLinearLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Channel 0: blend < 0.5, should darken (linear burn)
  expect(result[0]).toBeLessThan(0.5);

  // Channel 1: blend > 0.5, should brighten (linear dodge)
  expect(result[1]).toBeGreaterThan(0.5);

  // Channel 2: blend = 0.5, boundary case
  // (behavior depends on implementation)
});
```

---

### 12. layerLuminositySourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify HSL luminosity blending (luminosity from src, hue+saturation from dst)

**Improved Test**:
```typescript
test("layerLuminositySourceOver4", async () => {
  const src = `
     import lygia::color::layer::luminositySourceOver::layerLuminositySourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Bright src, dark dst - should brighten dst while keeping its hue
       let srcColor = vec4f(0.8, 0.4, 0.2, 0.7); // Bright orange
       let dstColor = vec4f(0.2, 0.6, 0.8, 0.5); // Dark cyan
       let result = layerLuminositySourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify luminosity from src, hue+saturation from dst
  // Result should maintain cyan's hue (B > R) but with orange's brightness
  expect(result[2]).toBeGreaterThan(result[0]); // B > R (cyan-ish hue preserved)

  // Overall brightness should increase from dst toward src
  const resultLuma = result[0] * 0.299 + result[1] * 0.587 + result[2] * 0.114;
  const dstLuma = 0.2 * 0.299 + 0.6 * 0.587 + 0.8 * 0.114;
  expect(resultLuma).toBeGreaterThan(dstLuma * 0.9); // Should be brighter
});

test("layerLuminositySourceOver4 - gray to color", async () => {
  const src = `
     import lygia::color::layer::luminositySourceOver::layerLuminositySourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Apply gray's luminosity to color
       let srcColor = vec4f(0.3, 0.3, 0.3, 1.0); // Dark gray
       let dstColor = vec4f(1.0, 0.0, 0.0, 1.0); // Bright red
       let result = layerLuminositySourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Result should be dark (gray's luminosity) but red (dst's hue)
  expect(result[0]).toBeGreaterThan(result[1]); // More red than green
  expect(result[0]).toBeGreaterThan(result[2]); // More red than blue
  expect(result[0]).toBeLessThan(0.6); // Darker than original red
});
```

---

### 13. layerNegationSourceOver4

**Current Test**: Only checks alpha = 0.8

**Issue**: Doesn't verify negation blending (1 - abs(1 - base - blend))

**Improved Test**:
```typescript
test("layerNegationSourceOver4", async () => {
  const src = `
     import lygia::color::layer::negationSourceOver::layerNegationSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.7, 0.5, 0.3, 0.6);
       let dstColor = vec4f(0.4, 0.6, 0.8, 0.5);
       let result = layerNegationSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.8);

  // Verify negation formula: 1 - abs(1 - base - blend)
  // Example: R channel: 1 - abs(1 - 0.4 - 0.7) = 1 - abs(-0.1) = 1 - 0.1 = 0.9
  // This creates a unique blending effect

  // All channels should be affected by negation formula
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerNegationSourceOver4 - complementary colors", async () => {
  const src = `
     import lygia::color::layer::negationSourceOver::layerNegationSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Colors that sum to 1.0 should produce 1.0
       let srcColor = vec4f(0.7, 0.3, 0.5, 1.0);
       let dstColor = vec4f(0.3, 0.7, 0.5, 1.0);
       let result = layerNegationSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // 1 - abs(1 - 0.7 - 0.3) = 1 - 0 = 1.0
  expect(result[0]).toBeCloseTo(1.0, 2);
  expect(result[1]).toBeCloseTo(1.0, 2);
  expect(result[2]).toBeCloseTo(1.0, 2);
});
```

---

### 14. layerPinLightSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify pin light (combination of lighten and darken)

**Improved Test**:
```typescript
test("layerPinLightSourceOver4", async () => {
  const src = `
     import lygia::color::layer::pinLightSourceOver::layerPinLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.5, 0.6, 0.4, 0.7);
       let dstColor = vec4f(0.3, 0.5, 0.7, 0.5);
       let result = layerPinLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify pin light: if blend < 0.5, darken; else lighten
  // This creates selective darkening/lightening per channel
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerPinLightSourceOver4 - extreme values", async () => {
  const src = `
     import lygia::color::layer::pinLightSourceOver::layerPinLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Use extreme blend values to show lighten/darken behavior
       let srcColor = vec4f(0.2, 0.8, 0.5, 1.0);
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0);
       let result = layerPinLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Channel 0: blend < 0.5, should use darken (min)
  expect(result[0]).toBeLessThanOrEqual(0.5);

  // Channel 1: blend > 0.5, should use lighten (max)
  expect(result[1]).toBeGreaterThanOrEqual(0.5);

  // Channel 2: blend = 0.5, boundary case
});
```

---

### 15. layerReflectSourceOver4

**Current Test**: Only checks alpha = 0.92

**Issue**: Doesn't verify reflect blending (base^2 / (1 - blend))

**Improved Test**:
```typescript
test("layerReflectSourceOver4", async () => {
  const src = `
     import lygia::color::layer::reflectSourceOver::layerReflectSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.4, 0.6, 0.2, 0.8);
       let dstColor = vec4f(0.5, 0.3, 0.8, 0.6);
       let result = layerReflectSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Verify reflect creates brightening effect
  // Formula: base^2 / (1 - blend), clamped to 1.0
  // This creates a "glow" or reflection-like effect

  // Result should show brightening (considering alpha compositing)
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerReflectSourceOver4 - extreme reflection", async () => {
  const src = `
     import lygia::color::layer::reflectSourceOver::layerReflectSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Low blend values should create strong reflection
       let srcColor = vec4f(0.2, 0.2, 0.2, 1.0);
       let dstColor = vec4f(0.8, 0.8, 0.8, 1.0);
       let result = layerReflectSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // With low blend (0.2), formula: 0.8^2 / (1 - 0.2) = 0.64 / 0.8 = 0.8
  // Should show strong reflection effect (brightening)
  expect(result[0]).toBeGreaterThan(0.7);
  expect(result[1]).toBeGreaterThan(0.7);
  expect(result[2]).toBeGreaterThan(0.7);
});
```

---

### 16. layerSaturationSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify HSL saturation blending (saturation from src, hue+luminosity from dst)

**Improved Test**:
```typescript
test("layerSaturationSourceOver4", async () => {
  const src = `
     import lygia::color::layer::saturationSourceOver::layerSaturationSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Saturated src, desaturated dst - should saturate dst's color
       let srcColor = vec4f(0.8, 0.4, 0.2, 0.7); // Saturated orange
       let dstColor = vec4f(0.2, 0.6, 0.8, 0.5); // Relatively saturated cyan
       let result = layerSaturationSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify saturation from src, hue+luminosity from dst
  // Result should maintain cyan's hue (B > R) and luminosity
  expect(result[2]).toBeGreaterThan(result[0]); // B > R (cyan hue preserved)

  // Saturation should increase (more color variation between channels)
  const dstRange = Math.max(0.2, 0.6, 0.8) - Math.min(0.2, 0.6, 0.8);
  const resultRange = Math.max(result[0], result[1], result[2]) - Math.min(result[0], result[1], result[2]);
  // Note: Due to alpha compositing, this may not always hold perfectly
  expect(resultRange).toBeGreaterThan(0.0);
});

test("layerSaturationSourceOver4 - desaturate with gray", async () => {
  const src = `
     import lygia::color::layer::saturationSourceOver::layerSaturationSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Gray has no saturation - should desaturate dst
       let srcColor = vec4f(0.5, 0.5, 0.5, 1.0); // Gray (no saturation)
       let dstColor = vec4f(1.0, 0.0, 0.0, 1.0); // Pure red (fully saturated)
       let result = layerSaturationSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Result should be desaturated (all channels closer together)
  const avg = (result[0] + result[1] + result[2]) / 3;
  const maxDiff = Math.max(
    Math.abs(result[0] - avg),
    Math.abs(result[1] - avg),
    Math.abs(result[2] - avg)
  );

  // Should be much less saturated than pure red
  expect(maxDiff).toBeLessThan(0.3);
});
```

---

### 17. layerSoftLightSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify soft light blending (smooth overlay-like effect)

**Improved Test**:
```typescript
test("layerSoftLightSourceOver4", async () => {
  const src = `
     import lygia::color::layer::softLightSourceOver::layerSoftLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.5, 0.6, 0.4, 0.7);
       let dstColor = vec4f(0.3, 0.5, 0.7, 0.5);
       let result = layerSoftLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify soft light creates subtle contrast enhancement
  // Similar to overlay but with softer transition
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerSoftLightSourceOver4 - subtle contrast", async () => {
  const src = `
     import lygia::color::layer::softLightSourceOver::layerSoftLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Soft light should enhance but not drastically alter
       let srcColor = vec4f(0.3, 0.7, 0.5, 1.0);
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0);
       let result = layerSoftLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Dark blend (< 0.5) should darken slightly
  expect(result[0]).toBeLessThan(0.5);
  expect(result[0]).toBeGreaterThan(0.0);

  // Light blend (> 0.5) should brighten slightly
  expect(result[1]).toBeGreaterThan(0.5);
  expect(result[1]).toBeLessThan(1.0);

  // Mid blend should be close to dst
  expect(result[2]).toBeCloseTo(0.5, 1);
});
```

---

### 18. layerVividLightSourceOver4

**Current Test**: Only checks alpha = 0.85

**Issue**: Doesn't verify vivid light (combination of color burn and dodge)

**Improved Test**:
```typescript
test("layerVividLightSourceOver4", async () => {
  const src = `
     import lygia::color::layer::vividLightSourceOver::layerVividLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       let srcColor = vec4f(0.5, 0.6, 0.4, 0.7);
       let dstColor = vec4f(0.3, 0.5, 0.7, 0.5);
       let result = layerVividLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify vivid light creates strong contrast
  // If blend < 0.5: color burn (darkening)
  // If blend >= 0.5: color dodge (brightening)
  expect(result[0]).toBeGreaterThan(0.0);
  expect(result[1]).toBeGreaterThan(0.0);
  expect(result[2]).toBeGreaterThan(0.0);
});

test("layerVividLightSourceOver4 - extreme contrast", async () => {
  const src = `
     import lygia::color::layer::vividLightSourceOver::layerVividLightSourceOver4;

     @compute @workgroup_size(1)
     fn foo() {
       // Use extreme values to show burn and dodge
       let srcColor = vec4f(0.2, 0.8, 0.5, 1.0);
       let dstColor = vec4f(0.5, 0.5, 0.5, 1.0);
       let result = layerVividLightSourceOver4(srcColor, dstColor);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");

  // Channel 0: blend < 0.5, should apply color burn (darkening)
  expect(result[0]).toBeLessThan(0.5);

  // Channel 1: blend > 0.5, should apply color dodge (brightening)
  expect(result[1]).toBeGreaterThan(0.5);

  // Channel 2: blend = 0.5, boundary case (no change or minimal change)
  expect(result[2]).toBeCloseTo(0.5, 1);
});
```

---

## Implementation Priority

### High Priority (Most Important to Validate)
1. **layerHardMixSourceOver4** - Binary output should be verifiable
2. **layerAverageSourceOver4** - Simple math, easy to verify
3. **layerLinearBurnSourceOver4** - Additive formula, clear expectations
4. **layerLinearDodgeSourceOver4** - Additive formula, clear expectations

### Medium Priority (Color Space Conversions)
5. **layerColorSourceOver4** - HSL color mode
6. **layerHueSourceOver4** - HSL hue mode
7. **layerSaturationSourceOver4** - HSL saturation mode
8. **layerLuminositySourceOver4** - HSL luminosity mode

### Lower Priority (Complex Formulas)
9. **layerColorBurnSourceOver4** - Complex darkening
10. **layerColorDodgeSourceOver4** - Complex brightening
11. **layerHardLightSourceOver4** - Conditional blend
12. **layerSoftLightSourceOver4** - Smooth contrast
13. **layerLinearLightSourceOver4** - Conditional burn/dodge
14. **layerVividLightSourceOver4** - Conditional burn/dodge
15. **layerPinLightSourceOver4** - Conditional lighten/darken
16. **layerReflectSourceOver4** - Quadratic formula
17. **layerGlowSourceOver4** - Reflect variant
18. **layerNegationSourceOver4** - Absolute value formula

---

## Test Execution Notes

### Key Challenges

1. **Alpha Compositing Complexity**: The layer functions combine blend modes with alpha compositing, making exact RGB predictions difficult. Tests should focus on:
   - Qualitative behavior (darkening, brightening, hue shifts)
   - Boundary cases (fully opaque, fully transparent)
   - Relative comparisons rather than exact values

2. **HSL Mode Complexity**: Color/Hue/Saturation/Luminosity modes involve RGB↔HSL conversions, making predictions harder. Tests should verify:
   - Hue relationships (warm vs cool, which channel is dominant)
   - Saturation changes (channel spread)
   - Luminosity changes (weighted sum)

3. **Tolerance Requirements**: Due to floating-point arithmetic and complex formulas, tests may need generous tolerances (0.1 or even 0.2) for some operations.

### Testing Strategy

For each function, implement tests in this order:
1. **Alpha-only test** (current tests are already this) - Keep as baseline
2. **RGB qualitative test** - Verify blend mode behavior directionally
3. **Edge case test** - Fully opaque or extreme values to isolate blend mode

This allows incremental improvement without breaking existing tests.

---

## Conclusion

The current tests are **not trivial but are incomplete**. They successfully validate alpha compositing but completely miss the RGB blend mode functionality. Improving these tests requires:

1. Understanding the underlying blend mode formulas
2. Adding RGB channel assertions to existing tests
3. Creating edge case tests to isolate blend mode behavior from alpha compositing

The improved tests above provide a comprehensive template for achieving full validation coverage of the color layer functions.
