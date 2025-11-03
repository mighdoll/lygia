import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify color dodge brightens the image
  // Color dodge formula: min(base / (1 - blend), 1.0) where base=src, blend=dst
  // R: min(0.4 / (1 - 0.3), 1.0) = min(0.4 / 0.7, 1.0) ≈ 0.571
  // G: min(0.5 / (1 - 0.4), 1.0) = min(0.5 / 0.6, 1.0) ≈ 0.833
  // B: min(0.6 / (1 - 0.5), 1.0) = min(0.6 / 0.5, 1.0) = 1.0
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.445, 0.64333, 0.775, 0.85], result);
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // When blend is white (1.0), color dodge returns 1.0
  // Then source-over: 1.0 * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.83, 0.84, 0.85], result.slice(0, 3));
});

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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify alpha compositing
  expect(result[3]).toBeCloseTo(0.85);

  // Verify linear dodge brightening: min(base + blend, 1.0)
  // R: min(0.3 + 0.4, 1.0) = 0.7
  // G: min(0.2 + 0.5, 1.0) = 0.7
  // B: min(0.1 + 0.6, 1.0) = 0.7
  // Then source-over: blend * srcAlpha + dst * dstAlpha * (1 - srcAlpha)
  expectCloseTo([0.535, 0.52, 0.505, 0.85], result);
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
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // All channels should clamp to 1.0
  expectCloseTo([1.0, 1.0, 1.0], result.slice(0, 3));
});
