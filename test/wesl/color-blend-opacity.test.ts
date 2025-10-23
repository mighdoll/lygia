import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

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
  expectCloseTo([0.3, 0.5, 0.7], result);
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
  // Relaxed precision for division-based blend mode with opacity
  expectCloseTo([0.3, 0.25, 0.2], result);
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
  expectCloseTo([0.5, 0.5, 0.5], result);
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
  expectCloseTo([0.8, 0.3, 0.6], result);
});

// Color Space Conversion Tests (Additional)
