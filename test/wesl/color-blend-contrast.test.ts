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
