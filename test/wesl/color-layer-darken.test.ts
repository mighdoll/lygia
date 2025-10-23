import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Verify color burn darkens the image
  // Color burn formula: max((1 - (1 - base) / blend), 0)
  // For these values, all channels produce 0 or near-0 after blending:
  // R: max(1 - (1 - 0.4) / 0.6, 0) = max(1 - 1.0, 0) = 0.0
  // G: max(1 - (1 - 0.3) / 0.5, 0) = max(1 - 1.4, 0) = 0.0
  // B: max(1 - (1 - 0.5) / 0.4, 0) = max(1 - 1.25, 0) = 0.0
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.048, 0.036, 0.06, 0.92], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // When blend is black (0.0), color burn returns 0.0
  // Then source-over: 0.0 * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.096, 0.072, 0.048], result.slice(0, 3));
});

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
  const result = await testCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.92);

  // Verify linear burn darkening: max(base + blend - 1, 0)
  // R: max(0.4 + 0.6 - 1, 0) = 0.0
  // G: max(0.3 + 0.5 - 1, 0) = 0.0
  // B: max(0.2 + 0.7 - 1, 0) = 0.0
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.048, 0.036, 0.024, 0.92], result);
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
  const result = await testCompute(src, { elem: "vec4f" });

  // 0.5 + 0.5 - 1.0 = 0.0 for all channels
  expectCloseTo([0.0, 0.0, 0.0], result.slice(0, 3));
});
