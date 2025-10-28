# Test Plan: lighting-misc.test.ts

## Summary

**Total tests:** 6 (2 skipped)
**Good tests:** 3
**Trivial/Range-only tests:** 3
**Approval rate:** 50% (3/6)
**Status:** ⚠️ **NEEDS IMPROVEMENT**

**Updated per test-review.md guidance:** Improved tests now include specific expected values, test lighting behavior with meaningful inputs, and avoid trivial range-only validation.

## Test Status by Function

### ✅ Good Tests (Keep As-Is)

1. **fresnel vec3f** - Tests Fresnel at normal incidence (NoV=1.0), verifies result ≈ f0 (0.04), physically correct behavior
2. **fresnelF32** - Tests Fresnel f32 overload at normal incidence (NoV=1.0), verifies result ≈ f0 (0.04)
3. **fresnelFromVectors** - Tests Fresnel from explicit vectors (perpendicular view), verifies result ≈ f0, tests convenience overload

### ⚠️ Tests Needing Improvement (Updated with Specific Values)

4. **fresnelRoughness** - IMPROVED: Now tests specific mathematical formula, verifies roughness modulation at grazing angles (smooth≈0.9, rough≈0.1)
5. **specularCookTorrance** - IMPROVED: Now tests Cook-Torrance BRDF components (D*V*F), verifies roughness effect on specular lobe width
6. **toShininess** - IMPROVED: Now tests conversion formula with specific expected values (smooth≈194.4, rough≈9.8)

---

## Detailed Improvement Plans

### 1. fresnelRoughness - Test roughness effect on Fresnel contrast

**Current test (lines 63-94):**
```typescript
// Only checks that higher roughness reduces Fresnel at grazing angles
expect(result[2]).toBeLessThan(result[1]);
// Low roughness should be similar to standard Fresnel
expect(Math.abs(result[0] - result[1])).toBeLessThan(0.2);
// At normal incidence, should be close to f0
expectCloseTo([0.04], [result[3]], 0.05);
```

**Problem:** The test has good structure but weak assertions. It doesn't verify specific mathematical behavior of the roughness-adjusted Fresnel formula.

**Improved test:**
```typescript
test("fresnelRoughness", async () => {
  const src = `
     import lygia::lighting::fresnel::{fresnel, fresnelRoughness};

     @compute @workgroup_size(1)
     fn foo() {
       // fresnelRoughness attenuates high speculars at glancing angles
       // Formula: f0 + (max(1-roughness, f0) - f0) * pow5(1-NoV)
       let f0 = vec3f(0.04, 0.04, 0.04);

       // Test 1: At normal incidence (NoV=1.0), roughness should have minimal effect
       // pow5(1-1.0) = 0, so result should equal f0 regardless of roughness
       let normalSmooth = fresnelRoughness(f0, 1.0, 0.1);
       let normalRough = fresnelRoughness(f0, 1.0, 0.9);

       // Test 2: At grazing angle (NoV=0.1), roughness should modulate the Fresnel peak
       // For smooth surfaces (low roughness), Fresnel approaches 1.0 at grazing angles
       // For rough surfaces (high roughness), Fresnel peak is attenuated
       let grazingSmooth = fresnelRoughness(f0, 0.1, 0.1);  // max(0.9, 0.04) = 0.9 at grazing
       let grazingRough = fresnelRoughness(f0, 0.1, 0.9);   // max(0.1, 0.04) = 0.1 at grazing

       // Test 3: Mid-angle (NoV=0.5) should show intermediate behavior
       let midSmooth = fresnelRoughness(f0, 0.5, 0.1);
       let midRough = fresnelRoughness(f0, 0.5, 0.9);

       test::results[0] = vec4f(normalSmooth.x, normalRough.x, grazingSmooth.x, grazingRough.x);
       test::results[1] = vec4f(midSmooth.x, midRough.x, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f", 2);

  // Test 1: At normal incidence, both should equal f0 (0.04)
  expectCloseTo([0.04, 0.04], [result[0], result[1]], 0.01);

  // Test 2: At grazing angle, smooth surface should have much higher Fresnel than rough
  expect(result[2]).toBeGreaterThan(0.7);  // Smooth approaches ~0.9
  expect(result[3]).toBeLessThan(0.3);     // Rough is attenuated to ~0.1
  expect(result[2]).toBeGreaterThan(result[3] * 2);  // At least 2x difference

  // Test 3: Mid-angle should show intermediate values between normal and grazing
  expect(result[4]).toBeGreaterThan(result[0]);  // Mid > normal incidence
  expect(result[4]).toBeLessThan(result[2]);     // Mid < grazing (smooth)
  expect(result[5]).toBeGreaterThan(result[1]);  // Mid > normal incidence
  expect(result[5]).toBeLessThan(result[3]);     // Mid < grazing (rough)

  // Test 4: Roughness effect should be stronger at grazing angles
  let normalDiff = Math.abs(result[0] - result[1]);
  let grazingDiff = Math.abs(result[2] - result[3]);
  expect(grazingDiff).toBeGreaterThan(normalDiff * 10);  // Much larger effect at grazing
});
```

**Key improvements:**
- Tests the mathematical formula directly: `f0 + (max(1-roughness, f0) - f0) * pow5(1-NoV)`
- Verifies that at normal incidence (NoV=1.0), roughness has no effect (pow5(0) = 0)
- Verifies that at grazing angles (NoV=0.1), roughness significantly modulates the Fresnel peak
- Tests intermediate angles to show continuous behavior
- Validates physical property: roughness effect is strongest at grazing angles
- Uses specific expected values: smooth≈0.9, rough≈0.1 at grazing angles

**Aligns with test-review.md guidance:**
- ✅ Avoids range-only validation
- ✅ Tests specific mathematical behavior
- ✅ Uses meaningful inputs with expected outputs
- ✅ Verifies physical properties quantitatively

---

### 2. specularCookTorrance - Test Cook-Torrance BRDF components

**Current test (lines 146-183):**
```typescript
// Only checks that outputs are positive and different
expect(result[0]).toBeGreaterThan(0.0);
expect(result[1]).toBeGreaterThan(0.0);
// Rough should be greater than smooth at this particular angle
expect(result[1]).toBeGreaterThan(result[0]);
// Perfect alignment produces strong specular
expect(result[2]).toBeGreaterThan(0.01);
```

**Problem:** Test doesn't verify the actual Cook-Torrance formula or the relationship between roughness and specular behavior across different viewing angles.

**Improved test:**
```typescript
test("specularCookTorrance", async () => {
  const src = `
     import lygia::lighting::specular::cookTorrance::specularCookTorrance;

     @compute @workgroup_size(1)
     fn foo() {
       // Cook-Torrance BRDF: (D * V * F) where:
       // D = GGX distribution (normal distribution function)
       // V = Smith visibility term (geometric shadowing/masking)
       // F = Fresnel (view-dependent reflectance)

       let specularColor = vec3f(0.04, 0.04, 0.04);
       let N = vec3f(0.0, 0.0, 1.0);

       // Test 1: Perfect specular reflection (H = N)
       // Light and view aligned with normal
       let L1 = N;
       let V1 = N;
       let H1 = N;
       let NoV1 = 1.0;
       let NoL1 = 1.0;
       let NoH1 = 1.0;
       let perfectSmooth = specularCookTorrance(L1, N, H1, NoV1, NoL1, NoH1, 0.1, specularColor);
       let perfectRough = specularCookTorrance(L1, N, H1, NoV1, NoL1, NoH1, 0.9, specularColor);

       // Test 2: Grazing angle (light from side)
       // Smooth surfaces show strong specular at grazing angles
       // Rough surfaces scatter light more evenly
       let L2 = normalize(vec3f(1.0, 0.0, 0.1));  // Near-horizontal light
       let V2 = vec3f(0.0, 0.0, 1.0);             // View from above
       let H2 = normalize(L2 + V2);
       let NoV2 = dot(N, V2);
       let NoL2 = dot(N, L2);
       let NoH2 = dot(N, H2);
       let grazingSmooth = specularCookTorrance(L2, N, H2, NoV2, NoL2, NoH2, 0.1, specularColor);
       let grazingRough = specularCookTorrance(L2, N, H2, NoV2, NoL2, NoH2, 0.9, specularColor);

       // Test 3: Off-specular (H != N)
       // When H deviates from N, specular should decrease
       // Effect is stronger for smooth surfaces (narrow lobe)
       let L3 = normalize(vec3f(0.5, 0.0, 1.0));  // 45° from normal
       let V3 = vec3f(0.0, 0.0, 1.0);
       let H3 = normalize(L3 + V3);
       let NoV3 = dot(N, V3);
       let NoL3 = dot(N, L3);
       let NoH3 = dot(N, H3);
       let offSpecSmooth = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.1, specularColor);
       let offSpecRough = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.9, specularColor);

       test::results[0] = vec4f(perfectSmooth.x, perfectRough.x, grazingSmooth.x, grazingRough.x);
       test::results[1] = vec4f(offSpecSmooth.x, offSpecRough.x, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f", 2);

  // Test 1: Perfect alignment produces strongest specular
  // Smooth surfaces have sharper, taller peaks
  expect(result[0]).toBeGreaterThan(0.1);  // Strong specular for smooth
  expect(result[0]).toBeGreaterThan(result[1]);  // Smooth > rough at peak

  // Test 2: At grazing angles, smooth surfaces still show specular
  // but rough surfaces scatter more
  expect(result[2]).toBeGreaterThan(0.0);
  expect(result[3]).toBeGreaterThan(0.0);

  // Test 3: Off-specular should be dimmer than perfect specular
  expect(result[4]).toBeLessThan(result[0]);  // Off-spec < perfect (smooth)
  expect(result[5]).toBeLessThan(result[1]);  // Off-spec < perfect (rough)

  // Test 4: Roughness effect on specular falloff
  // Smooth surfaces have sharper falloff (larger ratio)
  let smoothRatio = result[0] / (result[4] + 0.001);  // Peak / off-spec
  let roughRatio = result[1] / (result[5] + 0.001);
  expect(smoothRatio).toBeGreaterThan(roughRatio);  // Smooth falls off faster

  // Test 5: Perfect specular should be strongest among all cases
  expect(result[0]).toBeGreaterThan(result[2]);  // Perfect > grazing (smooth)
  expect(result[0]).toBeGreaterThan(result[4]);  // Perfect > off-spec (smooth)
});
```

**Key improvements:**
- Tests the Cook-Torrance BRDF formula: (D * V * F)
- Validates behavior at perfect specular reflection (H = N)
- Tests grazing angle behavior (Fresnel effect)
- Tests off-specular falloff (GGX distribution narrow vs. wide lobes)
- Verifies physical property: smooth surfaces have sharper, taller specular highlights
- Tests roughness effect on specular lobe width
- Uses specific geometric configurations with expected relative behaviors

**Aligns with test-review.md guidance:**
- ✅ Avoids range-only validation
- ✅ Tests specific mathematical behavior (BRDF components)
- ✅ Uses meaningful inputs (perfect specular, grazing angle, off-specular)
- ✅ Verifies physical properties through relational comparisons

---

### 3. toShininess - Test PBR to Blinn-Phong conversion formula

**Current test (lines 185-219):**
```typescript
// Only checks output is in reasonable range
expect(result[0]).toBeGreaterThan(result[1]);
expect(result[1]).toBeGreaterThan(0.0);
expect(result[2]).toBeGreaterThan(0.0);
expect(result[2]).toBeLessThan(300.0);
expect(Math.abs(result[2] - result[3])).toBeGreaterThan(1.0);
```

**Problem:** Test doesn't verify the actual conversion formula or validate the inverse relationship between roughness and shininess.

**Improved test:**
```typescript
test("toShininess", async () => {
  const src = `
     import lygia::lighting::toShininess::toShininess;

     @compute @workgroup_size(1)
     fn foo() {
       // toShininess converts PBR roughness to Blinn-Phong shininess
       // Formula: s = (0.95 - roughness*0.5)^4 * (80 + 160*(1-metallic))
       // Inverse relationship: high roughness -> low shininess

       // Test 1: Extremes of roughness (dielectric)
       let verySmooth = toShininess(0.0, 0.0);   // s = 0.95^4 * 240 ≈ 194.4
       let veryRough = toShininess(1.0, 0.0);    // s = 0.45^4 * 240 ≈ 9.8

       // Test 2: Mid-roughness (dielectric)
       let midRough = toShininess(0.5, 0.0);     // s = 0.7^4 * 240 ≈ 57.6

       // Test 3: Metallic vs dielectric at same roughness
       let dielectric = toShininess(0.3, 0.0);   // Uses 240 multiplier (80+160*1)
       let metallic = toShininess(0.3, 1.0);     // Uses 80 multiplier (80+160*0)

       // Test 4: Intermediate metallic values
       let halfMetal = toShininess(0.3, 0.5);    // Uses 160 multiplier (80+160*0.5)

       // Test 5: Verify formula at specific point
       // roughness=0.2, metallic=0.0:
       // s = 0.85^4 * 240 = 0.5220 * 240 ≈ 125.3
       let formula_test = toShininess(0.2, 0.0);

       test::results[0] = vec4f(verySmooth, veryRough, midRough, dielectric);
       test::results[1] = vec4f(metallic, halfMetal, formula_test, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f", 2);

  // Test 1: Very smooth has highest shininess
  expect(result[0]).toBeGreaterThan(150.0);  // Should be ~194
  expectCloseTo([194.4], [result[0]], 5.0);

  // Very rough has low shininess
  expect(result[1]).toBeLessThan(15.0);  // Should be ~9.8
  expectCloseTo([9.8], [result[1]], 2.0);

  // Test 2: Inverse relationship - smooth >> rough
  expect(result[0]).toBeGreaterThan(result[1] * 10);  // At least 10x difference

  // Test 3: Mid-roughness is between extremes
  expect(result[2]).toBeGreaterThan(result[1]);  // Mid > rough
  expect(result[2]).toBeLessThan(result[0]);     // Mid < smooth
  expectCloseTo([57.6], [result[2]], 5.0);

  // Test 4: Metallic reduces shininess (smaller multiplier)
  expect(result[4]).toBeLessThan(result[3]);  // Metallic < dielectric
  // Dielectric uses 240, metallic uses 80, ratio should be 3:1
  expect(result[3] / result[4]).toBeCloseTo(3.0, 1);

  // Test 5: Half-metallic is between dielectric and full metallic
  expect(result[5]).toBeGreaterThan(result[4]);  // Half > full metal
  expect(result[5]).toBeLessThan(result[3]);     // Half < dielectric
  // Half-metallic uses 160, ratio to dielectric (240) should be 2:3
  expect(result[5] / result[3]).toBeCloseTo(2.0/3.0, 1);

  // Test 6: Verify specific formula calculation
  // roughness=0.2: s = 0.85^4 * 240 = 125.3
  expectCloseTo([125.3], [result[6]], 3.0);

  // Test 7: All values should be in valid shininess range
  expect(result[0]).toBeLessThan(250.0);   // Max is 240 * 0.95^4
  expect(result[1]).toBeGreaterThan(0.0);  // Min is positive
});
```

**Key improvements:**
- Tests the actual formula: `s = (0.95 - roughness*0.5)^4 * (80 + 160*(1-metallic))`
- Verifies inverse relationship between roughness and shininess
- Tests extreme values (0.0 and 1.0 roughness)
- Validates metallic parameter effect on multiplier (80 vs 240)
- Tests specific numerical values against expected calculations (194.4, 9.8, 57.6, 125.3)
- Verifies all outputs are in valid range for Blinn-Phong exponents

**Aligns with test-review.md guidance:**
- ✅ Avoids range-only validation
- ✅ Tests specific mathematical behavior (conversion formula)
- ✅ Uses meaningful inputs with specific expected outputs
- ✅ Verifies mathematical properties (inverse relationship, metallic multiplier)

---

## Implementation Priority

1. **toShininess** (HIGHEST) - Simple formula verification, good learning example
2. **fresnelRoughness** (MEDIUM) - Important for PBR, tests physical properties
3. **specularCookTorrance** (LOWER) - Most complex, but comprehensive test shows BRDF understanding

---

## Notes

### Skipped Tests

- **fresnelReflection** - Requires `envMap` function not yet converted to WESL
- **raymarchCast** - Requires user-defined `map()` function before import, WESL module system limitation

These skipped tests are appropriately marked and documented with clear reasons for skipping.

---

## Testing Philosophy for Lighting Functions

Lighting functions in computer graphics are based on physical laws and empirical models. Good tests should:

1. **Verify mathematical formulas** - Check that the implementation matches the documented formula
2. **Test physical properties** - Validate expected behavior (e.g., Fresnel increases at grazing angles)
3. **Use extreme values** - Test boundary conditions (roughness=0, roughness=1, normal incidence, grazing)
4. **Compare relative behaviors** - Smooth vs. rough, metallic vs. dielectric, different angles
5. **Include specific numerical checks** - Don't just check ranges, verify actual computed values
6. **Use meaningful inputs** - Avoid trivial cases like zero inputs or pass-through values

Avoid:
- Range-only checks that just verify output is positive or in [0,1]
- Tests that only check "doesn't crash"
- Single-case tests that don't explore the function's parameter space
- Tests without physical/mathematical justification for expected values
- Trivial zero cases that don't exercise the function's logic
- Pass-through tests where input equals output

**Reference:** See `/Users/lee/wesl/lygia/notes/test-review.md` for comprehensive guidelines on what makes a test trivial vs. non-trivial.
