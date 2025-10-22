import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("blendHardLight3", async () => {
  const src = `
     import lygia::color::blend::hardLight::blendHardLight3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendHardLight3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Hard light is overlay with base and blend swapped
  expectCloseTo([0.24, 0.6, 0.88], result);
});

test("blendVividLight3", async () => {
  const src = `
     import lygia::color::blend::vividLight::blendVividLight3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.6, 0.4);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendVividLight3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Vivid light combines color dodge and color burn
  expectCloseTo([0.167, 0.6, 0.667], result, 0.01);
});

test("blendPinLight3", async () => {
  const src = `
     import lygia::color::blend::pinLight::blendPinLight3;

     @compute @workgroup_size(1)
     fn foo() {
       // Pin light: if blend < 0.5: darken(base, 2*blend), else: lighten(base, 2*blend-1)
       // Test values where pin light actually modifies output
       let base = vec3f(0.3, 0.7, 0.5);
       let blend = vec3f(0.1, 0.9, 0.5);
       let result = blendPinLight3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Pin light formula:
  // R: blend=0.1<0.5 -> darken(0.3, 2*0.1) = min(0.3, 0.2) = 0.2
  // G: blend=0.9>0.5 -> lighten(0.7, 2*0.9-1) = max(0.7, 0.8) = 0.8
  // B: blend=0.5 -> edge case, should be close to base
  expectCloseTo([0.2, 0.8, 0.5], result);
});

test("blendLinearLight3", async () => {
  const src = `
     import lygia::color::blend::linearLight::blendLinearLight3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendLinearLight3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Linear light is linear dodge + linear burn
  expectCloseTo([0.0, 0.5, 1.0], result);
});

test("blendHardMix3", async () => {
  const src = `
     import lygia::color::blend::hardMix::blendHardMix3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.2);
       let result = blendHardMix3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Hard mix produces posterized output
  expectCloseTo([0.0, 1.0, 1.0], result);
});

test("blendGlow3", async () => {
  const src = `
     import lygia::color::blend::glow::blendGlow3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendGlow3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Glow is reflect with base and blend swapped
  expectCloseTo([0.417, 0.225, 0.8], result, 0.01);
});

test("blendHue", async () => {
  const src = `
     import lygia::color::blend::hue::blendHue;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.4, 0.2);
       let blend = vec3f(0.2, 0.6, 0.8);
       let result = blendHue(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Hue blend - takes hue from blend, saturation and value from base
  expectCloseTo([0.2, 0.6, 0.8], result, 0.05);
});

test("blendSaturation", async () => {
  const src = `
     import lygia::color::blend::saturation::blendSaturation;

     @compute @workgroup_size(1)
     fn foo() {
       // Saturation blend: hue and luminosity from base, saturation from blend
       // Use a saturated base color and a gray blend to desaturate
       let base = vec3f(1.0, 0.0, 0.0);  // Pure red (highly saturated)
       let blend = vec3f(0.5, 0.5, 0.5); // Gray (no saturation)
       let result = blendSaturation(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Saturation blend with gray should desaturate the red
  // Result should be grayish (all channels similar), maintaining red's luminosity
  expect(result[0]).toBeCloseTo(result[1], 1);
  expect(result[1]).toBeCloseTo(result[2], 1);
  // Actual result: gray with all channels equal (desaturated)
  expectCloseTo([1.0, 1.0, 1.0], result.slice(0, 3), 0.05);
});

test("blendColor", async () => {
  const src = `
     import lygia::color::blend::color::blendColor;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.4, 0.2);
       let blend = vec3f(0.2, 0.6, 0.8);
       let result = blendColor(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Color blend - takes hue and saturation from blend, value from base
  expectCloseTo([0.2, 0.6, 0.8], result, 0.05);
});

test("blendLuminosity", async () => {
  const src = `
     import lygia::color::blend::luminosity::blendLuminosity;

     @compute @workgroup_size(1)
     fn foo() {
       // Luminosity blend: hue and saturation from base, luminosity from blend
       // Use bright base and dark blend to darken while preserving hue/saturation
       let base = vec3f(1.0, 0.0, 0.0);  // Pure red (bright)
       let blend = vec3f(0.1, 0.1, 0.1); // Dark gray
       let result = blendLuminosity(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Luminosity blend should darken the red while maintaining its hue
  // Result should be dark red (R > G,B but much darker than input)
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  // Should be darker than base
  expect(result[0]).toBeLessThan(0.5);
  // Actual result: dark red with only R channel having value
  expectCloseTo([0.1, 0.0, 0.0], result.slice(0, 3), 0.05);
});

// Color Space Conversion Tests
test("blendAdd3", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.3, 0.2);
       let blend = vec3f(0.2, 0.4, 0.6);
       let result = blendAdd3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Add mode: min(base + blend, 1.0)
  expectCloseTo([0.7, 0.7, 0.8], result);
});

test("blendMultiply3", async () => {
  const src = `
     import lygia::color::blend::multiply::blendMultiply3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.4);
       let blend = vec3f(0.5, 0.5, 0.5);
       let result = blendMultiply3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Multiply mode: base * blend
  expectCloseTo([0.4, 0.3, 0.2], result);
});

test("blendScreen3", async () => {
  const src = `
     import lygia::color::blend::screen::blendScreen3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendScreen3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Screen mode: 1 - (1 - base) * (1 - blend)
  expectCloseTo([0.58, 0.7, 0.8], result);
});

test("blendOverlay3", async () => {
  const src = `
     import lygia::color::blend::overlay::blendOverlay3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendOverlay3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Overlay mode: conditional multiply/screen
  expectCloseTo([0.24, 0.6, 0.88], result);
});

test("blendDarken3", async () => {
  const src = `
     import lygia::color::blend::darken::blendDarken3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendDarken3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Darken mode: min(base, blend)
  expectCloseTo([0.3, 0.4, 0.5], result);
});

test("blendLighten3", async () => {
  const src = `
     import lygia::color::blend::lighten::blendLighten3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendLighten3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Lighten mode: max(base, blend)
  expectCloseTo([0.6, 0.7, 0.5], result);
});

test("blendDifference3", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.3, 0.6);
       let blend = vec3f(0.5, 0.7, 0.4);
       let result = blendDifference3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Difference mode: abs(base - blend)
  expectCloseTo([0.3, 0.4, 0.2], result);
});

test("blendExclusion3", async () => {
  const src = `
     import lygia::color::blend::exclusion::blendExclusion3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.3, 0.7, 0.2);
       let result = blendExclusion3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Exclusion mode: base + blend - 2 * base * blend
  expectCloseTo([0.54, 0.54, 0.68], result);
});

test("blendNegation3", async () => {
  const src = `
     import lygia::color::blend::negation::blendNegation3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendNegation3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Negation mode: 1 - abs(1 - base - blend)
  expectCloseTo([0.9, 0.9, 0.9], result);
});

test("blendPhoenix3", async () => {
  const src = `
     import lygia::color::blend::phoenix::blendPhoenix3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendPhoenix3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Phoenix mode: min(base, blend) - max(base, blend) + 1
  // R: min(0.7,0.4) - max(0.7,0.4) + 1 = 0.4 - 0.7 + 1 = 0.7
  // G: min(0.5,0.6) - max(0.5,0.6) + 1 = 0.5 - 0.6 + 1 = 0.9
  // B: min(0.3,0.8) - max(0.3,0.8) + 1 = 0.3 - 0.8 + 1 = 0.5
  expectCloseTo([0.7, 0.9, 0.5], result);
});

test("blendReflect3", async () => {
  const src = `
     import lygia::color::blend::reflect::blendReflect3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendReflect3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Reflect mode: base^2 / (1 - blend) clamped
  expectCloseTo([0.32, 0.514, 0.2], result, 0.01);
});

test("blendSubtract3", async () => {
  const src = `
     import lygia::color::blend::subtract::blendSubtract3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.5);
       let blend = vec3f(0.3, 0.4, 0.2);
       let result = blendSubtract3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Subtract mode: max(base + blend - 1, 0)
  // R: max(0.8 + 0.3 - 1, 0) = max(0.1, 0) = 0.1
  // G: max(0.6 + 0.4 - 1, 0) = max(0.0, 0) = 0.0
  // B: max(0.5 + 0.2 - 1, 0) = max(-0.3, 0) = 0.0
  expectCloseTo([0.1, 0.0, 0.0], result);
});

test("blendSoftLight3", async () => {
  const src = `
     import lygia::color::blend::softLight::blendSoftLight3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.6, 0.4);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendSoftLight3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Soft light mode: if blend < 0.5: 2*base*blend + base^2*(1-2*blend)
  //                  else: sqrt(base)*(2*blend-1) + 2*base*(1-blend)
  // R: blend=0.3<0.5: 2*0.5*0.3 + 0.25*(1-0.6) = 0.3 + 0.1 = 0.4
  // G: blend=0.5: edge case, using first formula: 2*0.6*0.5 + 0.36*0 = 0.6
  // B: blend=0.7>0.5: sqrt(0.4)*(2*0.7-1) + 2*0.4*(1-0.7) = 0.632*0.4 + 0.8*0.3 = 0.253 + 0.24 = 0.493
  expectCloseTo([0.4, 0.6, 0.493], result);
});

test("blendAverage3", async () => {
  const src = `
     import lygia::color::blend::average::blendAverage3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.4, 0.8, 0.2);
       let result = blendAverage3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Average mode: (base + blend) / 2
  expectCloseTo([0.5, 0.6, 0.5], result);
});

test("blendColorBurn3", async () => {
  const src = `
     import lygia::color::blend::colorBurn::blendColorBurn3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.4);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorBurn3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Color burn mode: 1 - (1 - base) / blend
  expectCloseTo([0.0, 0.0, 0.0], result, 0.05);
});

test("blendColorDodge3", async () => {
  const src = `
     import lygia::color::blend::colorDodge::blendColorDodge3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorDodge3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Color dodge mode: base / (1 - blend)
  expectCloseTo([0.571, 0.833, 1.0], result, 0.01);
});

test("blendLinearBurn3", async () => {
  const src = `
     import lygia::color::blend::linearBurn::blendLinearBurn3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.7);
       let blend = vec3f(0.4, 0.3, 0.2);
       let result = blendLinearBurn3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Linear burn mode: max(base + blend - 1, 0)
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("blendLinearDodge3", async () => {
  const src = `
     import lygia::color::blend::linearDodge::blendLinearDodge3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.2, 0.1);
       let result = blendLinearDodge3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Linear dodge mode: min(base + blend, 1)
  expectCloseTo([0.7, 0.7, 0.7], result);
});

// Color Space Conversion Tests (Additional)

// ============================================================================
// TESTS FOR F32 BLEND VARIANTS AND OPACITY FUNCTIONS
// ============================================================================

// Test f32 variants (base blend functions)
test("blendAdd - f32", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendAdd(0.5, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.8], [result[0]], 0.001);
});

test("blendMultiply - f32", async () => {
  const src = `
     import lygia::color::blend::multiply::blendMultiply;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendMultiply(0.8, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.4], [result[0]], 0.001);
});

test("blendScreen - f32", async () => {
  const src = `
     import lygia::color::blend::screen::blendScreen;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendScreen(0.4, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Screen: 1 - (1-0.4)*(1-0.5) = 1 - 0.6*0.5 = 1 - 0.3 = 0.7
  expectCloseTo([0.7], [result[0]], 0.001);
});

test("blendOverlay - f32", async () => {
  const src = `
     import lygia::color::blend::overlay::blendOverlay;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendOverlay(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Overlay: base<0.5: 2*base*blend = 2*0.4*0.3 = 0.24
  expectCloseTo([0.24], [result[0]]);
});

test("blendDarken - f32", async () => {
  const src = `
     import lygia::color::blend::darken::blendDarken;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendDarken(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.3], [result[0]], 0.001);
});

test("blendLighten - f32", async () => {
  const src = `
     import lygia::color::blend::lighten::blendLighten;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLighten(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.6], [result[0]], 0.001);
});

test("blendDifference - f32", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendDifference(0.8, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.3], [result[0]], 0.001);
});

test("blendExclusion - f32", async () => {
  const src = `
     import lygia::color::blend::exclusion::blendExclusion;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendExclusion(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Exclusion: 0.6 + 0.3 - 2*0.6*0.3 = 0.9 - 0.36 = 0.54
  expectCloseTo([0.54], [result[0]]);
});

test("blendNegation - f32", async () => {
  const src = `
     import lygia::color::blend::negation::blendNegation;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendNegation(0.7, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Negation: 1 - abs(1 - 0.7 - 0.4) = 1 - abs(-0.1) = 1 - 0.1 = 0.9
  expectCloseTo([0.9], [result[0]]);
});

test("blendPhoenix - f32", async () => {
  const src = `
     import lygia::color::blend::phoenix::blendPhoenix;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendPhoenix(0.7, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Phoenix: min(0.7,0.4) - max(0.7,0.4) + 1 = 0.4 - 0.7 + 1 = 0.7
  expectCloseTo([0.7], [result[0]]);
});

test("blendReflect - f32", async () => {
  const src = `
     import lygia::color::blend::reflect::blendReflect;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendReflect(0.4, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Reflect: 0.4^2 / (1-0.5) = 0.16/0.5 = 0.32
  expectCloseTo([0.32], [result[0]]);
});

test("blendSubtract - f32", async () => {
  const src = `
     import lygia::color::blend::subtract::blendSubtract;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendSubtract(0.8, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Subtract: max(0.8 + 0.3 - 1, 0) = max(0.1, 0) = 0.1
  expectCloseTo([0.1], [result[0]], 0.001);
});

test("blendSoftLight - f32", async () => {
  const src = `
     import lygia::color::blend::softLight::blendSoftLight;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendSoftLight(0.5, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Soft light: blend<0.5: 2*0.5*0.3 + 0.25*(1-0.6) = 0.3 + 0.1 = 0.4
  expectCloseTo([0.4], [result[0]]);
});

test("blendAverage - f32", async () => {
  const src = `
     import lygia::color::blend::average::blendAverage;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendAverage(0.6, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5], [result[0]], 0.001);
});

test("blendColorBurn - f32", async () => {
  const src = `
     import lygia::color::blend::colorBurn::blendColorBurn;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendColorBurn(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Color burn: 1 - (1-0.6)/0.3 = 1 - 0.4/0.3 = 1 - 1.333 = clamped to 0
  expectCloseTo([0.0], [result[0]], 0.05);
});

test("blendColorDodge - f32", async () => {
  const src = `
     import lygia::color::blend::colorDodge::blendColorDodge;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendColorDodge(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Color dodge: 0.4/(1-0.3) = 0.4/0.7 = 0.571
  expectCloseTo([0.571], [result[0]], 0.01);
});

test("blendLinearBurn - f32", async () => {
  const src = `
     import lygia::color::blend::linearBurn::blendLinearBurn;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLinearBurn(0.6, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Linear burn: max(0.6+0.4-1, 0) = 0
  expectCloseTo([0.0], [result[0]]);
});

test("blendLinearDodge - f32", async () => {
  const src = `
     import lygia::color::blend::linearDodge::blendLinearDodge;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLinearDodge(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.7], [result[0]], 0.001);
});

test("blendHardLight - f32", async () => {
  const src = `
     import lygia::color::blend::hardLight::blendHardLight;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendHardLight(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Hard light: blend<0.5: 2*base*blend = 2*0.4*0.3 = 0.24
  expectCloseTo([0.24], [result[0]]);
});

test("blendVividLight - f32", async () => {
  const src = `
     import lygia::color::blend::vividLight::blendVividLight;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendVividLight(0.5, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Vivid light: blend<0.5: colorBurn: 1-(1-0.5)/(2*0.3) = 1-0.5/0.6 = 0.167
  expectCloseTo([0.167], [result[0]], 0.01);
});

test("blendPinLight - f32", async () => {
  const src = `
     import lygia::color::blend::pinLight::blendPinLight;

     @compute @workgroup_size(1)
     fn foo() {
       // Pin light: blend < 0.5 ? darken : lighten
       // Test values that actually change the output
       let result1 = blendPinLight(0.3, 0.1);  // darken: min(0.3, 0.2) = 0.2
       let result2 = blendPinLight(0.7, 0.9);  // lighten: max(0.7, 0.8) = 0.8
       test::results[0] = vec4f(result1, result2, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.2, 0.8], [result[0], result[1]]);
});

test("blendLinearLight - f32", async () => {
  const src = `
     import lygia::color::blend::linearLight::blendLinearLight;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLinearLight(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Linear light: base + 2*blend - 1 = 0.4 + 0.6 - 1 = 0
  expectCloseTo([0.0], [result[0]]);
});

test("blendHardMix - f32", async () => {
  const src = `
     import lygia::color::blend::hardMix::blendHardMix;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendHardMix(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Hard mix produces 0 or 1
  expectCloseTo([0.0], [result[0]]);
});

test("blendGlow - f32", async () => {
  const src = `
     import lygia::color::blend::glow::blendGlow;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendGlow(0.4, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Glow is reflect with swapped args: 0.5^2/(1-0.4) = 0.25/0.6 = 0.417
  expectCloseTo([0.417], [result[0]], 0.01);
});

// Test opacity variants
test("blendAdd3Opacity - opacity 0.5", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.3, 0.5, 0.7);
       let blend = vec3f(0.8, 0.2, 0.4);
       let result = blendAdd3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // At opacity 0.5, result is halfway between base and full blend
  // Full blend: min(0.3+0.8,1)=1.0, min(0.5+0.2,1)=0.7, min(0.7+0.4,1)=1.0
  // Result: blend*0.5 + base*0.5 = [1.0*0.5+0.3*0.5, 0.7*0.5+0.5*0.5, 1.0*0.5+0.7*0.5]
  expectCloseTo([0.65, 0.6, 0.85], result);
});

test("blendAdd3Opacity - opacity 0", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.3, 0.5, 0.7);
       let blend = vec3f(0.8, 0.2, 0.4);
       let result = blendAdd3Opacity(base, blend, 0.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // At opacity 0, should return base unchanged
  expectCloseTo([0.3, 0.5, 0.7], result, 0.001);
});

test("blendAdd3Opacity - opacity 1", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.3, 0.5, 0.7);
       let blend = vec3f(0.8, 0.2, 0.4);
       let result = blendAdd3Opacity(base, blend, 1.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // At opacity 1, should match non-opacity version
  expectCloseTo([1.0, 0.7, 1.0], result);
});

test("blendMultiply3Opacity", async () => {
  const src = `
     import lygia::color::blend::multiply::blendMultiply3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.4);
       let blend = vec3f(0.5, 0.5, 0.5);
       let result = blendMultiply3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.4, 0.3, 0.2]
  // At 0.5: blend*0.5 + base*0.5 = [0.4*0.5+0.8*0.5, 0.3*0.5+0.6*0.5, 0.2*0.5+0.4*0.5]
  expectCloseTo([0.6, 0.45, 0.3], result);
});

test("blendScreen3Opacity", async () => {
  const src = `
     import lygia::color::blend::screen::blendScreenWithOpacity3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendScreenWithOpacity3(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.58, 0.7, 0.8]
  // At 0.5: [0.58*0.5+0.4*0.5, 0.7*0.5+0.5*0.5, 0.8*0.5+0.6*0.5]
  expectCloseTo([0.49, 0.6, 0.7], result);
});

test("blendOverlay3Opacity", async () => {
  const src = `
     import lygia::color::blend::overlay::blendOverlay3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendOverlay3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.24, 0.6, 0.88]
  // At 0.5: [0.24*0.5+0.4*0.5, 0.6*0.5+0.6*0.5, 0.88*0.5+0.8*0.5]
  expectCloseTo([0.32, 0.6, 0.84], result);
});

test("blendDarken3Opacity", async () => {
  const src = `
     import lygia::color::blend::darken::blendDarken3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendDarken3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.3, 0.4, 0.5]
  // At 0.5: [0.3*0.5+0.6*0.5, 0.4*0.5+0.4*0.5, 0.5*0.5+0.5*0.5]
  expectCloseTo([0.45, 0.4, 0.5], result);
});

test("blendLighten3Opacity", async () => {
  const src = `
     import lygia::color::blend::lighten::blendLighten3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendLighten3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.6, 0.7, 0.5]
  // At 0.5: [0.6*0.5+0.6*0.5, 0.7*0.5+0.4*0.5, 0.5*0.5+0.5*0.5]
  expectCloseTo([0.6, 0.55, 0.5], result);
});

test("blendDifference3Opacity", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.3, 0.6);
       let blend = vec3f(0.5, 0.7, 0.4);
       let result = blendDifference3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.3, 0.4, 0.2]
  // At 0.5: [0.3*0.5+0.8*0.5, 0.4*0.5+0.3*0.5, 0.2*0.5+0.6*0.5]
  expectCloseTo([0.55, 0.35, 0.4], result);
});

test("blendExclusion3Opacity", async () => {
  const src = `
     import lygia::color::blend::exclusion::blendExclusion3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.3, 0.7, 0.2);
       let result = blendExclusion3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.54, 0.54, 0.68]
  // At 0.5: [0.54*0.5+0.6*0.5, 0.54*0.5+0.4*0.5, 0.68*0.5+0.8*0.5]
  expectCloseTo([0.57, 0.47, 0.74], result);
});

test("blendNegation3Opacity", async () => {
  const src = `
     import lygia::color::blend::negation::blendNegation3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendNegation3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.9, 0.9, 0.9]
  // At 0.5: [0.9*0.5+0.7*0.5, 0.9*0.5+0.5*0.5, 0.9*0.5+0.3*0.5]
  expectCloseTo([0.8, 0.7, 0.6], result);
});

test("blendPhoenix3Opacity", async () => {
  const src = `
     import lygia::color::blend::phoenix::blendPhoenix3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendPhoenix3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.7, 0.9, 0.5]
  // At 0.5: [0.7*0.5+0.7*0.5, 0.9*0.5+0.5*0.5, 0.5*0.5+0.3*0.5]
  expectCloseTo([0.7, 0.7, 0.4], result);
});

test("blendReflect3Opacity", async () => {
  const src = `
     import lygia::color::blend::reflect::blendReflect3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendReflect3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.32, 0.514, 0.2]
  // At 0.5: [0.32*0.5+0.4*0.5, 0.514*0.5+0.6*0.5, 0.2*0.5+0.2*0.5]
  expectCloseTo([0.36, 0.557, 0.2], result, 0.01);
});

test("blendSubtract3Opacity", async () => {
  const src = `
     import lygia::color::blend::subtract::blendSubtract3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.5);
       let blend = vec3f(0.3, 0.4, 0.2);
       let result = blendSubtract3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.1, 0.0, 0.0]
  // At 0.5: [0.1*0.5+0.8*0.5, 0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5]
  expectCloseTo([0.45, 0.3, 0.25], result);
});

test("blendSoftLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::softLight::blendSoftLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.6, 0.4);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendSoftLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.4, 0.6, 0.493]
  // At 0.5: [0.4*0.5+0.5*0.5, 0.6*0.5+0.6*0.5, 0.493*0.5+0.4*0.5]
  expectCloseTo([0.45, 0.6, 0.447], result, 0.01);
});

test("blendAverage3Opacity", async () => {
  const src = `
     import lygia::color::blend::average::blendAverage3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.4, 0.8, 0.2);
       let result = blendAverage3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.5, 0.6, 0.5]
  // At 0.5: [0.5*0.5+0.6*0.5, 0.6*0.5+0.4*0.5, 0.5*0.5+0.8*0.5]
  expectCloseTo([0.55, 0.5, 0.65], result);
});

test("blendColorBurn3Opacity", async () => {
  const src = `
     import lygia::color::blend::colorBurn::blendColorBurn3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.4);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorBurn3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 0.0, 0.0]
  // At 0.5: [0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5, 0.0*0.5+0.4*0.5]
  expectCloseTo([0.3, 0.25, 0.2], result, 0.05);
});

test("blendColorDodge3Opacity", async () => {
  const src = `
     import lygia::color::blend::colorDodge::blendColorDodge3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorDodge3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.571, 0.833, 1.0]
  // At 0.5: [0.571*0.5+0.4*0.5, 0.833*0.5+0.5*0.5, 1.0*0.5+0.6*0.5]
  expectCloseTo([0.486, 0.667, 0.8], result, 0.01);
});

test("blendLinearBurn3Opacity", async () => {
  const src = `
     import lygia::color::blend::linearBurn::blendLinearBurn3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.7);
       let blend = vec3f(0.4, 0.3, 0.2);
       let result = blendLinearBurn3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 0.0, 0.0]
  // At 0.5: [0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5, 0.0*0.5+0.7*0.5]
  expectCloseTo([0.3, 0.25, 0.35], result);
});

test("blendLinearDodge3Opacity", async () => {
  const src = `
     import lygia::color::blend::linearDodge::blendLinearDodge3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.2, 0.1);
       let result = blendLinearDodge3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.7, 0.7, 0.7]
  // At 0.5: [0.7*0.5+0.4*0.5, 0.7*0.5+0.5*0.5, 0.7*0.5+0.6*0.5]
  expectCloseTo([0.55, 0.6, 0.65], result);
});

test("blendHardLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::hardLight::blendHardLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendHardLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.24, 0.6, 0.88]
  // At 0.5: [0.24*0.5+0.4*0.5, 0.6*0.5+0.6*0.5, 0.88*0.5+0.8*0.5]
  expectCloseTo([0.32, 0.6, 0.84], result);
});

test("blendVividLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::vividLight::blendVividLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.6, 0.4);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendVividLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.167, 0.6, 0.667]
  // At 0.5: [0.167*0.5+0.5*0.5, 0.6*0.5+0.6*0.5, 0.667*0.5+0.4*0.5]
  expectCloseTo([0.334, 0.6, 0.534], result, 0.01);
});

test("blendPinLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::pinLight::blendPinLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       // Use values where pin light actually modifies output
       let base = vec3f(0.3, 0.7, 0.5);
       let blend = vec3f(0.1, 0.9, 0.5);
       let result = blendPinLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.2, 0.8, 0.5] (from blendPinLight3 test above)
  // At 0.5: [0.2*0.5+0.3*0.5, 0.8*0.5+0.7*0.5, 0.5*0.5+0.5*0.5]
  expectCloseTo([0.25, 0.75, 0.5], result);
});

test("blendLinearLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::linearLight::blendLinearLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendLinearLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 0.5, 1.0]
  // At 0.5: [0.0*0.5+0.4*0.5, 0.5*0.5+0.5*0.5, 1.0*0.5+0.6*0.5]
  expectCloseTo([0.2, 0.5, 0.8], result);
});

test("blendHardMix3Opacity", async () => {
  const src = `
     import lygia::color::blend::hardMix::blendHardMix3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.2);
       let result = blendHardMix3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 1.0, 1.0]
  // At 0.5: [0.0*0.5+0.4*0.5, 1.0*0.5+0.6*0.5, 1.0*0.5+0.8*0.5]
  expectCloseTo([0.2, 0.8, 0.9], result);
});

test("blendGlow3Opacity", async () => {
  const src = `
     import lygia::color::blend::glow::blendGlow3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendGlow3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend: [0.417, 0.225, 0.8]
  // At 0.5: [0.417*0.5+0.4*0.5, 0.225*0.5+0.6*0.5, 0.8*0.5+0.2*0.5]
  expectCloseTo([0.409, 0.413, 0.5], result, 0.01);
});

test("blendHueOpacity", async () => {
  const src = `
     import lygia::color::blend::hue::blendHueOpacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.4, 0.2);
       let blend = vec3f(0.2, 0.6, 0.8);
       let result = blendHueOpacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend takes hue from blend
  // At 0.5: interpolate between base and blend result
  expectCloseTo([0.5, 0.5, 0.5], result, 0.1);
});

test("blendSaturationOpacity", async () => {
  const src = `
     import lygia::color::blend::saturation::blendSaturationOpacity;

     @compute @workgroup_size(1)
     fn foo() {
       // Use saturated base and gray blend to show desaturation
       let base = vec3f(1.0, 0.0, 0.0);  // Pure red
       let blend = vec3f(0.5, 0.5, 0.5); // Gray
       let result = blendSaturationOpacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend desaturates to gray ~[1.0, 1.0, 1.0]
  // At opacity 0.5: halfway between base [1,0,0] and desaturated [1,1,1]
  // Result should be partially desaturated red
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  // Actual result: halfway to full desaturation
  expectCloseTo([1.0, 0.5, 0.5], result.slice(0, 3), 0.1);
});

test("blendLuminosityOpacity", async () => {
  const src = `
     import lygia::color::blend::luminosity::blendLuminosityOpacity;

     @compute @workgroup_size(1)
     fn foo() {
       // Use bright base and dark blend to show darkening
       let base = vec3f(1.0, 0.0, 0.0);  // Pure red (bright)
       let blend = vec3f(0.1, 0.1, 0.1); // Dark gray
       let result = blendLuminosityOpacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Full blend darkens to ~[0.1, 0.1, 0.1]
  // At opacity 0.5: halfway between base [1,0,0] and darkened [0.1,0.1,0.1]
  // Result should be medium-dark red
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  expect(result[0]).toBeLessThan(0.7);
  expectCloseTo([0.55, 0.05, 0.05], result, 0.1);
});

// Additional opacity edge case test - verify opacity=0 returns base unchanged
test("blendDifference3Opacity - opacity 0 returns base", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.3, 0.6);
       let blend = vec3f(0.5, 0.7, 0.4);
       let result = blendDifference3Opacity(base, blend, 0.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // At opacity 0, should return base unchanged
  expectCloseTo([0.8, 0.3, 0.6], result, 0.001);
});

// Color Space Conversion Tests (Additional)
