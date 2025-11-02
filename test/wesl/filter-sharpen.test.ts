import { expect, test } from "vitest";
import {
  checkerboardTexture,
  createSampler,
  getGPUDevice,
  solidTexture,
} from "wesl-test";
import {
  expectCloseTo,
  lygiaTestCompute,
  lygiaTestFragment,
} from "./testUtil.ts";

// Sharpen filter tests - require texture/sampler, use fragment shaders

test("sharpenAdaptive", async () => {
  const device = await getGPUDevice();
  const checkerTex = checkerboardTexture(device, 256, 256, 8); // Larger cells for more edge visibility
  const solidTex = solidTexture(device, [0.5, 0.5, 0.5, 1.0], 256, 256);
  const sampler = createSampler(device);

  const srcSharpen = `
    import lygia::filter::sharpen::adaptive::sharpenAdaptive;

    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0, 1.0 / 256.0);
      return sharpenAdaptive(input_tex, input_samp, uv, pixel_size);
    }`;

  // Test sharpening on solid color (no edges to sharpen)
  const sharpenedSolid = await lygiaTestFragment(srcSharpen, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: solidTex, sampler }],
  });

  // Test sharpening on checkerboard (has edges to sharpen)
  const sharpenedChecker = await lygiaTestFragment(srcSharpen, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: checkerTex, sampler }],
  });

  // Sharpening a solid color should produce minimal change (no edges to enhance)
  expect(Math.abs(sharpenedSolid[0] - 0.5)).toBeLessThan(0.1);

  // The function should execute without errors (non-trivial execution test)
  // Both results should be different (one has edges, one doesn't)
  expect(sharpenedChecker[0]).not.toBe(sharpenedSolid[0]);

  // Exact value regression tests
  expectCloseTo([0, 0, 0, 1], sharpenedChecker.slice(0, 4));
  expectCloseTo([0.502, 0.502, 0.502, 1], sharpenedSolid.slice(0, 4));
});

test("sharpenAdaptive4", async () => {
  const device = await getGPUDevice();
  const checkerTex = checkerboardTexture(device, 256, 256, 8); // Larger cells
  const solidTex = solidTexture(device, [0.6, 0.6, 0.6, 1.0], 256, 256);
  const sampler = createSampler(device);

  const srcSharpen = `
    import lygia::filter::sharpen::adaptive::sharpenAdaptive4;

    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0, 1.0 / 256.0);
      return sharpenAdaptive4(input_tex, input_samp, uv, pixel_size, 1.0);
    }`;

  const sharpenedChecker = await lygiaTestFragment(srcSharpen, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: checkerTex, sampler }],
  });

  const sharpenedSolid = await lygiaTestFragment(srcSharpen, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: solidTex, sampler }],
  });

  // Alpha should be preserved
  expect(sharpenedChecker[3]).toBeCloseTo(1.0);
  expect(sharpenedSolid[3]).toBeCloseTo(1.0);

  // Solid color should remain close to original
  expect(Math.abs(sharpenedSolid[0] - 0.6)).toBeLessThan(0.1);

  // Checker and solid should produce different results (edges vs no edges)
  expect(sharpenedChecker[0]).not.toBe(sharpenedSolid[0]);

  // Exact value regression tests
  expectCloseTo([0, 0, 0, 1], sharpenedChecker.slice(0, 4));
  expectCloseTo([0.6, 0.6, 0.6, 1], sharpenedSolid.slice(0, 4));
});

test("sharpenContrastAdaptive", async () => {
  const device = await getGPUDevice();
  const checkerTex = checkerboardTexture(device, 256, 256, 32);
  const solidTex = solidTexture(device, [0.5, 0.5, 0.5, 1.0], 256, 256);
  const sampler = createSampler(device);

  const src = `
    import lygia::filter::sharpen::adaptive::sharpenContrastAdaptive;

    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0, 1.0 / 256.0);
      let sharpened = sharpenContrastAdaptive(input_tex, input_samp, uv, pixel_size, 1.0);
      return vec4f(sharpened, 1.0);
    }`;

  const sharpenedChecker = await lygiaTestFragment(src, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: checkerTex, sampler }],
  });

  const sharpenedSolid = await lygiaTestFragment(src, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: solidTex, sampler }],
  });

  // Checkerboard (with edges) and solid (without edges) should produce different results
  // Sharpening should enhance edges in checker but leave solid mostly unchanged
  const diff = Math.abs(sharpenedChecker[0] - sharpenedSolid[0]);
  expect(diff).toBeGreaterThan(0.1);

  // Solid color sharpening should stay near original value
  expect(Math.abs(sharpenedSolid[0] - 0.5)).toBeLessThan(0.1);

  // Exact value regression tests
  expectCloseTo([0, 0, 0, 1], sharpenedChecker.slice(0, 4));
  expectCloseTo([0.502, 0.502, 0.502, 1], sharpenedSolid.slice(0, 4));
});

test("sharpenFast", async () => {
  const device = await getGPUDevice();
  const checkerTex = checkerboardTexture(device, 256, 256, 32);
  const solidTex = solidTexture(device, [0.6, 0.6, 0.6, 1.0], 256, 256);
  const sampler = createSampler(device);

  const src = `
    import lygia::filter::sharpen::fast::sharpenFast;

    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0, 1.0 / 256.0);
      return sharpenFast(input_tex, input_samp, uv, pixel_size);
    }`;

  const sharpenedChecker = await lygiaTestFragment(src, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: checkerTex, sampler }],
  });

  const sharpenedSolid = await lygiaTestFragment(src, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: solidTex, sampler }],
  });

  // Fast sharpening should behave differently on textured vs solid input
  expect(sharpenedChecker[0]).not.toBeCloseTo(sharpenedSolid[0], 1);

  // Solid should remain close to original (0.6)
  expect(Math.abs(sharpenedSolid[0] - 0.6)).toBeLessThan(0.1);

  // Exact value regression tests
  expectCloseTo([0, 0, 0, 1], sharpenedChecker.slice(0, 4));
  expectCloseTo([0.6, 0.6, 0.6, 1], sharpenedSolid.slice(0, 4));
});

test("sharpenFast4", async () => {
  const device = await getGPUDevice();
  const checkerTex = checkerboardTexture(device, 256, 256, 32);
  const solidTex = solidTexture(device, [0.7, 0.7, 0.7, 1.0], 256, 256);
  const sampler = createSampler(device);

  const src = `
    import lygia::filter::sharpen::fast::sharpenFast4;

    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0, 1.0 / 256.0);
      return sharpenFast4(input_tex, input_samp, uv, pixel_size, 1.0);
    }`;

  const sharpenedChecker = await lygiaTestFragment(src, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: checkerTex, sampler }],
  });

  const sharpenedSolid = await lygiaTestFragment(src, {
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: solidTex, sampler }],
  });

  // Checkerboard and solid should produce different results
  expect(sharpenedChecker[0]).not.toBeCloseTo(sharpenedSolid[0], 1);

  // Solid should remain close to original (0.7)
  expect(Math.abs(sharpenedSolid[0] - 0.7)).toBeLessThan(0.1);

  // Alpha should be preserved
  expect(sharpenedChecker[3]).toBeCloseTo(1.0);
  expect(sharpenedSolid[3]).toBeCloseTo(1.0);

  // Exact value regression tests
  expectCloseTo([0, 0, 0, 1], sharpenedChecker.slice(0, 4));
  expectCloseTo([0.702, 0.702, 0.702, 1], sharpenedSolid.slice(0, 4));
});

test("sharpendAdaptiveControl4", async () => {
  const src = `
     import lygia::filter::sharpen::adaptive::sharpendAdaptiveControl4;

     @compute @workgroup_size(1)
     fn foo() {
       // sharpendAdaptiveControl4 computes perceptual luma: dot(rgba*rgba, vec4(0.21266, 0.71516, 0.07219, 0.0))
       // Test with gray color
       let gray = vec4f(0.5, 0.5, 0.5, 1.0);
       let result1 = sharpendAdaptiveControl4(gray);

       // Test with colored input
       let orange = vec4f(0.8, 0.5, 0.2, 1.0);
       let result2 = sharpendAdaptiveControl4(orange);

       // Test with black (should be 0)
       let black = vec4f(0.0, 0.0, 0.0, 1.0);
       let result3 = sharpendAdaptiveControl4(black);

       test::results[0] = vec3f(result1, result2, result3);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Gray: (0.5^2) * (0.21266 + 0.71516 + 0.07219) = 0.25 * 1.0 = 0.25
  expect(result[0]).toBeCloseTo(0.25, 2);

  // Orange: (0.8^2)*0.21266 + (0.5^2)*0.71516 + (0.2^2)*0.07219
  //       = 0.64*0.21266 + 0.25*0.71516 + 0.04*0.07219
  //       = 0.13610 + 0.17879 + 0.00289 = 0.31778
  expect(result[1]).toBeCloseTo(0.318, 2);

  // Black should be 0
  expect(result[2]).toBeCloseTo(0.0, 3);
});
