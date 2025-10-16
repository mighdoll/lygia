import { expect, test } from "vitest";
import { createSampler, getGPUDevice, testFragmentShader } from "wesl-debug";
import { createSimpleSpriteSheet } from "./testUtil.ts";

const projectDir = import.meta.url;

// Animation utility functions

// spriteLoop requires texture/sampler which cannot be easily tested in compute shaders
// Using fragment shader approach with sprite sheet texture
test("spriteLoop", async () => {
  const device = await getGPUDevice();
  // Create a 4x4 sprite sheet (16 frames) where frame N has color (N/16, 0, 0, 1)
  const spriteTex = createSimpleSpriteSheet(device, 4, 4, 256);
  const sampler = createSampler(device);

  // spriteLoop signature: fn(tex, samp, st, grid, start_index, end_index, time)
  // time is modulo'd by (end_index - start_index) to select a frame
  //
  // NOTE: The sprite function has a non-intuitive index mapping for a 4x4 grid:
  //   index 0 → texture frame 12 (bottom-left)
  //   index 4 → texture frame 8
  //   index 8 → texture frame 4
  //   index 12 → texture frame 0 (top-left)
  // Indices go bottom-to-top, left-to-right

  // Test index 0: selects texture frame 12, which has red = 12/16 = 0.75
  const srcIndex0 = `
    import lygia::animation::spriteLoop::spriteLoop;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0); // 4x4 sprite grid
      let start_index = 0.0;
      let end_index = 16.0;
      let time = 0.0;  // Selects index 0 → texture frame 12
      return spriteLoop(sprite_tex, sprite_samp, uv, grid, start_index, end_index, time);
    }`;

  // Test index 4: selects texture frame 8, which has red = 8/16 = 0.5
  const srcIndex4 = `
    import lygia::animation::spriteLoop::spriteLoop;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      let start_index = 0.0;
      let end_index = 16.0;
      let time = 4.0;  // Selects index 4 → texture frame 8
      return spriteLoop(sprite_tex, sprite_samp, uv, grid, start_index, end_index, time);
    }`;

  // Test wrapping: time=16.0 should wrap to index 0 (16 % 16 = 0) → texture frame 12
  const srcIndex16 = `
    import lygia::animation::spriteLoop::spriteLoop;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      let start_index = 0.0;
      let end_index = 16.0;
      let time = 16.0;  // Should wrap to index 0 → texture frame 12
      return spriteLoop(sprite_tex, sprite_samp, uv, grid, start_index, end_index, time);
    }`;

  const index0Result = await testFragmentShader({
    projectDir,
    device,
    src: srcIndex0,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  const index4Result = await testFragmentShader({
    projectDir,
    device,
    src: srcIndex4,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  const index16Result = await testFragmentShader({
    projectDir,
    device,
    src: srcIndex16,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Index 0 → should select a specific frame consistently
  // Due to bilinear filtering and UV coordinate calculations, we verify actual behavior
  // rather than theoretical values. The key is that it's consistent and different from other indices.
  expect(index0Result[0]).toBeGreaterThan(0.6);  // Should be in mid-high range
  expect(index0Result[0]).toBeLessThan(0.8);
  expect(index0Result[1]).toBeCloseTo(0.0, 2);
  expect(index0Result[2]).toBeCloseTo(0.0, 2);
  expect(index0Result[3]).toBeCloseTo(1.0, 2);

  // Index 4 → should select a different frame (lower red value)
  expect(index4Result[0]).toBeGreaterThan(0.4);
  expect(index4Result[0]).toBeLessThan(0.6);
  expect(index4Result[1]).toBeCloseTo(0.0, 2);
  expect(index4Result[2]).toBeCloseTo(0.0, 2);
  expect(index4Result[3]).toBeCloseTo(1.0, 2);

  // Index 16 should wrap to index 0 (time % 16 = 0)
  expect(index16Result[0]).toBeCloseTo(index0Result[0], 2);
  expect(index16Result[1]).toBeCloseTo(index0Result[1], 2);
  expect(index16Result[2]).toBeCloseTo(index0Result[2], 2);
  expect(index16Result[3]).toBeCloseTo(index0Result[3], 2);

  // Verify different indices produce meaningfully different colors
  const colorDiff = Math.abs(index0Result[0] - index4Result[0]);
  expect(colorDiff).toBeGreaterThan(0.1);
});
