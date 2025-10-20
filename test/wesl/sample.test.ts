import { expect, test } from "vitest";
import { getGPUDevice, testFragmentShader } from "wesl-debug";
import { createSimpleSpriteSheet } from "./spriteTestUtil.ts";

const projectDir = import.meta.url;

// Sample utility functions - require texture/sampler

test("sampleSprite", async () => {
  const device = await getGPUDevice();
  const cols = 4;
  const rows = 4;
  const totalFrames = cols * rows;

  // Use simple sprite sheet where frame N has color (N/totalFrames, 0, 0, 1)
  const spriteTex = createSimpleSpriteSheet(device, cols, rows, 256);

  // Create nearest-neighbor sampler to avoid filtering issues
  const sampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
  });

  // The sprite() function computes:
  //   i = index + grid.x
  //   cell = (floor(i), grid.y - floor(i / grid.x))
  // For a 4x4 grid, this maps input frame to texture cell
  const calcTextureFrameIndex = (inputFrame: number) => {
    const i = inputFrame + cols;
    const cellX = Math.floor(i) % cols;
    const cellY = rows - Math.floor(i / cols);
    return cellY * cols + cellX;
  };

  const expectedFrame0 = calcTextureFrameIndex(0) / totalFrames;
  const expectedFrame5 = calcTextureFrameIndex(5) / totalFrames;
  const expectedFrame10 = calcTextureFrameIndex(10) / totalFrames;

  // Test frame 0
  const srcFrame0 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, 0.0);
    }`;

  const frame0Result = await testFragmentShader({
    projectDir,
    device,
    src: srcFrame0,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Test frame 5
  const srcFrame5 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, 5.0);
    }`;

  const frame5Result = await testFragmentShader({
    projectDir,
    device,
    src: srcFrame5,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Test frame 10
  const srcFrame10 = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, 10.0);
    }`;

  const frame10Result = await testFragmentShader({
    projectDir,
    device,
    src: srcFrame10,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: spriteTex, sampler }],
  });

  // Verify each frame matches expected color (red channel = frameIndex/totalFrames, green=0, blue=0, alpha=1)
  // Note: Small differences (~0.004) are due to 8-bit texture quantization (Math.floor(N/16*255)/255)
  expect(frame0Result[0]).toBeCloseTo(expectedFrame0, 2); // Red channel
  expect(frame0Result[1]).toBeCloseTo(0.0, 2); // Green = 0
  expect(frame0Result[2]).toBeCloseTo(0.0, 2); // Blue = 0
  expect(frame0Result[3]).toBeCloseTo(1.0, 2); // Alpha = 1

  expect(frame5Result[0]).toBeCloseTo(expectedFrame5, 2);
  expect(frame5Result[1]).toBeCloseTo(0.0, 2);
  expect(frame5Result[2]).toBeCloseTo(0.0, 2);
  expect(frame5Result[3]).toBeCloseTo(1.0, 2);

  expect(frame10Result[0]).toBeCloseTo(expectedFrame10, 2);
  expect(frame10Result[1]).toBeCloseTo(0.0, 2);
  expect(frame10Result[2]).toBeCloseTo(0.0, 2);
  expect(frame10Result[3]).toBeCloseTo(1.0, 2);

  // Verify frames are actually different (sanity check)
  const colorDiff0to5 =
    Math.abs(frame0Result[0] - frame5Result[0]) +
    Math.abs(frame0Result[1] - frame5Result[1]) +
    Math.abs(frame0Result[2] - frame5Result[2]);

  expect(colorDiff0to5).toBeGreaterThan(0.1); // Frames must have distinct colors
});
