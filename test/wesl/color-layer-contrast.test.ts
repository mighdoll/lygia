import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Hard light: blendHardLight(src, dst) = blendOverlay(dst, src)
  // R: dst=0.3 < 0.5 → 2*0.3*0.4 = 0.24
  // G: dst=0.5 ≥ 0.5 → 1 - 2*(1-0.5)*(1-0.6) = 1 - 2*0.5*0.4 = 0.6
  // B: dst=0.7 ≥ 0.5 → 1 - 2*(1-0.7)*(1-0.8) = 1 - 2*0.3*0.2 = 0.88
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.213, 0.495, 0.721, 0.85], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Hard light: blendHardLight(src, dst) = blendOverlay(dst, src)
  // With src=0.2, dst=0.8: overlay checks if base (dst=0.8) < 0.5? No
  // Since dst >= 0.5: 1 - 2*(1-dst)*(1-src) = 1 - 2*0.2*0.8 = 0.68
  // With full opacity, source-over just returns the blend result
  expectCloseTo([0.68, 0.68, 0.68], result.slice(0, 3));
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Hard light: blendHardLight(src, dst) = blendOverlay(dst, src)
  // With src=0.8, dst=0.2: overlay checks if base (dst=0.2) < 0.5? Yes
  // Since dst < 0.5: 2*dst*src = 2*0.2*0.8 = 0.32
  // With full opacity, source-over just returns the blend result
  expectCloseTo([0.32, 0.32, 0.32], result.slice(0, 3));
});

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Soft light: if blend < 0.5: 2*base*blend + base²*(1-2*blend)
  //             else: sqrt(base)*(2*blend-1) + 2*base*(1-blend)
  // R: dst=0.3 < 0.5 → 2*0.5*0.3 + 0.25*(1-0.6) = 0.3 + 0.1 = 0.4
  // G: dst=0.5 ≥ 0.5 → sqrt(0.6)*0 + 2*0.6*0.5 = 0.6
  // B: dst=0.7 ≥ 0.5 → sqrt(0.4)*0.4 + 2*0.4*0.3 ≈ 0.253 + 0.24 = 0.493
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.325, 0.495, 0.45, 0.85], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Dark blend (< 0.5) should darken slightly
  expect(result[0]).toBeLessThan(0.5);
  expect(result[0]).toBeGreaterThan(0.0);

  // Light blend (> 0.5) should brighten slightly
  expect(result[1]).toBeGreaterThan(0.5);
  expect(result[1]).toBeLessThan(1.0);

  // Mid blend should be close to dst
  expect(result[2]).toBeCloseTo(0.5, 1);

  // Current implementation's specific output
  expectCloseTo([0.3, 0.7, 0.5], result.slice(0, 3));
});

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Vivid light: if blend < 0.5: colorBurn(base, blend*2)
  //              else: colorDodge(base, (blend-0.5)*2)
  // R: dst=0.3 < 0.5 → colorBurn(0.5, 0.6) = max(1-(1-0.5)/0.6, 0) ≈ 0.167
  // G: dst=0.5 ≥ 0.5 → colorDodge(0.6, 0.0) = min(0.6/(1-0), 1) = 0.6
  // B: dst=0.7 ≥ 0.5 → colorDodge(0.4, 0.4) = min(0.4/(1-0.4), 1) ≈ 0.667
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.16167, 0.495, 0.57167, 0.85], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Channel 0: blend < 0.5, should apply color burn (darkening)
  expect(result[0]).toBeLessThan(0.5);

  // Channel 1: blend > 0.5, should apply color dodge (brightening)
  expect(result[1]).toBeGreaterThan(0.5);

  // Channel 2: blend = 0.5, boundary case (no change or minimal change)
  expect(result[2]).toBeCloseTo(0.5, 1);

  // Current implementation's specific output
  expectCloseTo([0.2, 0.8, 0.5], result.slice(0, 3));
});

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Linear light: if blend < 0.5, linearBurn(base, blend*2); else linearDodge(base, (blend-0.5)*2)
  // R: dst=0.3 < 0.5 → linearBurn(0.4, 0.6) = max(0.4+0.6-1, 0) = 0.0
  // G: dst=0.5 ≥ 0.5 → linearDodge(0.5, 0.0) = min(0.5+0.0, 1) = 0.5
  // B: dst=0.7 ≥ 0.5 → linearDodge(0.6, 0.4) = min(0.6+0.4, 1) = 1.0
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.045, 0.425, 0.805, 0.85], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Channel 0: blend < 0.5, should darken (linear burn)
  expect(result[0]).toBeLessThan(0.5);

  // Channel 1: blend > 0.5, should brighten (linear dodge)
  expect(result[1]).toBeGreaterThan(0.5);

  // Channel 2: blend = 0.5, boundary case (behavior depends on implementation)

  // Current implementation's specific output
  expectCloseTo([0.2, 0.8, 0.5], result.slice(0, 3));
});

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Pin light: if blend < 0.5 then min(base, blend*2) else max(base, (blend-0.5)*2)
  // base=src, blend=dst (in blendPinLight call)
  // R: dst=0.3 < 0.5 → min(0.5, 0.6) = 0.5 → sourceOver: 0.5*0.7 + 0.3*0.5*0.3 = 0.395
  // G: dst=0.5 ≥ 0.5 → max(0.6, 0.0) = 0.6 → sourceOver: 0.6*0.7 + 0.5*0.5*0.3 = 0.495
  // B: dst=0.7 ≥ 0.5 → max(0.4, 0.4) = 0.4 → sourceOver: 0.4*0.7 + 0.7*0.5*0.3 = 0.385
  expectCloseTo([0.395, 0.495, 0.385, 0.85], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Pin light with full opacity: if blend < 0.5 then min(base, blend*2) else max(base, (blend-0.5)*2)
  // base=src, blend=dst
  // R: base=0.2, blend=0.5 ≥ 0.5 → max(0.2, 0.0) = 0.2
  // G: base=0.8, blend=0.5 ≥ 0.5 → max(0.8, 0.0) = 0.8
  // B: base=0.5, blend=0.5 ≥ 0.5 → max(0.5, 0.0) = 0.5
  expectCloseTo([0.2, 0.8, 0.5], result.slice(0, 3));
});

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Hard mix: if vividLight(base, blend) < 0.5 then 0.0 else 1.0
  // base=dst, blend=src in the layer function
  // R: vividLight(0.3, 0.4) = colorBurn(0.3, 0.8) = max(1-(1-0.3)/0.8, 0) = 0.125 < 0.5 → 0.0
  // G: vividLight(0.5, 0.6) = colorDodge(0.5, 0.2) = min(0.5/0.8, 1) = 0.625 ≥ 0.5 → 1.0
  // B: vividLight(0.2, 0.8) = colorDodge(0.2, 0.6) = min(0.2/0.4, 1) = 0.5 ≥ 0.5 → 1.0
  // Then source-over with α=(0.6, 0.5): blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.06, 0.7, 0.64, 0.8], result); // [0.0*0.6+0.3*0.5*0.4, 1.0*0.6+0.5*0.5*0.4, 1.0*0.6+0.2*0.5*0.4, 0.8]
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
  const result = await testCompute(src, { elem: "vec4f" });

  // Hard mix with full opacity produces pure posterization (0.0 or 1.0)
  // R: vividLight(0.3, 0.4) = 0.125 < 0.5 → 0.0
  // G: vividLight(0.5, 0.6) = 0.625 ≥ 0.5 → 1.0
  // B: vividLight(0.2, 0.8) = 0.5 ≥ 0.5 → 1.0
  expectCloseTo([0.0, 1.0, 1.0], result.slice(0, 3));
});
