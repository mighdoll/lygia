import { expect, test } from "vitest";
import { createSampler, getGPUDevice, testFragmentShader } from "wesl-debug";
import { createSpriteSheetTexture } from "./spriteTestUtil.ts";

const projectDir = import.meta.url;

// Sample utility functions - require texture/sampler

test("sampleSprite", async () => {
  const device = await getGPUDevice();
  const spriteTex = createSpriteSheetTexture(device, 4, 4, 256);
  const sampler = createSampler(device);

  // Sample frame 0
  const srcFrame0 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0); // 4x4 sprite grid
      let frame = 0.0;
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, frame);
    }`;

  // Sample frame 5 (different cell)
  const srcFrame5 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      let frame = 5.0;
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, frame);
    }`;

  const frame0Result = await testFragmentShader({
    projectDir,
    device,
    src: srcFrame0,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  const frame5Result = await testFragmentShader({
    projectDir,
    device,
    src: srcFrame5,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Different frames should sample different colors from the sprite sheet
  const colorDiff =
    Math.abs(frame0Result[0] - frame5Result[0]) +
    Math.abs(frame0Result[1] - frame5Result[1]) +
    Math.abs(frame0Result[2] - frame5Result[2]);

  expect(colorDiff).toBeGreaterThan(0.1); // Frames should have visibly different colors

  // Both should have valid alpha
  expect(frame0Result[3]).toBeCloseTo(1.0);
  expect(frame5Result[3]).toBeCloseTo(1.0);

  // All values should be in valid range
  expect(frame0Result[0]).toBeGreaterThanOrEqual(0.0);
  expect(frame0Result[0]).toBeLessThanOrEqual(1.0);
  expect(frame5Result[0]).toBeGreaterThanOrEqual(0.0);
  expect(frame5Result[0]).toBeLessThanOrEqual(1.0);
});
