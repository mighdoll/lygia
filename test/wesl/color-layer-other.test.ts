import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify alpha compositing (source-over)
  expect(result[3]).toBeCloseTo(0.92); // 0.8 + 0.6 * (1 - 0.8)

  // Verify average blending is applied to RGB
  // Average blend: (src + dst) * 0.5 = (0.8+0.4, 0.6+0.2, 0.4+0.6) * 0.5 = (0.6, 0.4, 0.5)
  // Then source-over compositing: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  // R: 0.6*0.8 + 0.4*0.6*0.2 = 0.48 + 0.048 = 0.528
  // G: 0.4*0.8 + 0.2*0.6*0.2 = 0.32 + 0.024 = 0.344
  // B: 0.5*0.8 + 0.6*0.6*0.2 = 0.4 + 0.072 = 0.472
  expectCloseTo([0.528, 0.344, 0.472, 0.92], result);
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Alpha should be 1.0 (fully opaque)
  // RGB should be the pure average blend: (src + dst) * 0.5
  expectCloseTo([0.5, 0.5, 0.5, 1.0], result);
});

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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.8);

  // Verify negation formula: 1 - abs(1 - base - blend)
  // R: 1 - abs(1 - 0.4 - 0.7) = 1 - abs(-0.1) = 0.9
  // G: 1 - abs(1 - 0.6 - 0.5) = 1 - abs(-0.1) = 0.9
  // B: 1 - abs(1 - 0.8 - 0.3) = 1 - abs(-0.1) = 0.9
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.62, 0.66, 0.7, 0.8], result);
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // 1 - abs(1 - 0.7 - 0.3) = 1 - 0 = 1.0
  expectCloseTo([1.0, 1.0, 1.0], result.slice(0, 3));
});

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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Reflect formula: min(base² / (1-blend), 1.0) where base=src, blend=dst
  // R: min(0.4² / (1-0.5), 1) = min(0.16 / 0.5, 1) = 0.32
  // G: min(0.6² / (1-0.3), 1) = min(0.36 / 0.7, 1) ≈ 0.514
  // B: min(0.2² / (1-0.8), 1) = min(0.04 / 0.2, 1) = 0.2
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.316, 0.44743, 0.256, 0.92], result);
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Reflect formula: min(base*base / (1 - blend), 1.0)
  // With src=0.2 (blend), dst=0.8 (base): min(0.2*0.2 / (1 - 0.8), 1.0) = min(0.04 / 0.2, 1.0) = 0.2
  // With full opacity, source-over just returns the blend result
  expectCloseTo([0.2, 0.2, 0.2], result.slice(0, 3));
});

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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Glow = reflect(dst, src) where reflect(blend, base) = min(blend² / (1-base), 1)
  // R: reflect(0.5, 0.4) = min(0.25 / 0.6, 1) ≈ 0.417
  // G: reflect(0.3, 0.6) = min(0.09 / 0.4, 1) = 0.225
  // B: reflect(0.8, 0.2) = min(0.64 / 0.8, 1) = 0.8
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.39333, 0.216, 0.736, 0.92], result);
});
