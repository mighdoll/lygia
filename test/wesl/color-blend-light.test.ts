import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

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
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Glow is reflect with base and blend swapped
  expectCloseTo([0.417, 0.225, 0.8], result, 0.01);
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Glow is reflect with swapped args: 0.5^2/(1-0.4) = 0.25/0.6 = 0.417
  expectCloseTo([0.417], [result[0]], 0.01);
});

// Test opacity variants
