import { expect, test } from "vitest";
import { getGPUDevice } from "wesl-debug";
import { createSimpleSpriteSheet } from "./spriteTestUtil.ts";
import { testFragment } from "./testUtil.ts";

// Sample utility functions - require texture/sampler

test("sampleSprite", async () => {
  const device = await getGPUDevice();
  const cols = 2;
  const rows = 2;
  const totalFrames = cols * rows;

  const spriteTex = createSimpleSpriteSheet(device, cols, rows, 128);
  const sampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
  });

  // Calculate expected texture frame indices
  const calcTextureFrameIndex = (inputFrame: number) => {
    const i = inputFrame + cols;
    const cellX = Math.floor(i) % cols;
    const cellY = rows - Math.floor(i / cols);
    return cellY * cols + cellX;
  };

  const expectedFrame0 = calcTextureFrameIndex(0) / totalFrames;
  const expectedFrame2 = calcTextureFrameIndex(2) / totalFrames;

  // Test frame 0
  const srcFrame0 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 128.0;
      let grid = vec2f(2.0, 2.0);
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, 0.0);
    }`;

  const frame0Result = await testFragment(srcFrame0, {
    textureFormat: "rgba32float",
    size: [128, 128],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Test frame 2
  const srcFrame2 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 128.0;
      let grid = vec2f(2.0, 2.0);
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, 2.0);
    }`;

  const frame2Result = await testFragment(srcFrame2, {
    textureFormat: "rgba32float",
    size: [128, 128],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Verify frame 0 (red channel = frameIndex/totalFrames, others = 0 or 1)
  expect(frame0Result[0]).toBeCloseTo(expectedFrame0, 2);
  expect(frame0Result[1]).toBeCloseTo(0.0, 2);
  expect(frame0Result[2]).toBeCloseTo(0.0, 2);
  expect(frame0Result[3]).toBeCloseTo(1.0, 2);

  // Verify frame 2
  expect(frame2Result[0]).toBeCloseTo(expectedFrame2, 2);
  expect(frame2Result[1]).toBeCloseTo(0.0, 2);
  expect(frame2Result[2]).toBeCloseTo(0.0, 2);
  expect(frame2Result[3]).toBeCloseTo(1.0, 2);

  // Verify frames are different (sanity check)
  expect(Math.abs(frame0Result[0] - frame2Result[0])).toBeGreaterThan(0.1);
});
