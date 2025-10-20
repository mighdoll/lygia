import { expect } from "vitest";
import {
  createCheckerboardTexture,
  createGradientTexture,
  createSampler,
  createSolidTexture,
  getGPUDevice,
  testComputeShader,
  testFragmentShader,
  type WgslElementType,
} from "wesl-debug";

const projectDir = import.meta.url;

/** compare two arrays for approximate equality */
export function expectCloseTo(
  a: number[],
  b: number[],
  epsilon = 0.0001,
): void {
  const match = a.every((val, index) => Math.abs(val - b[index]) < epsilon);
  if (match) return;
  expect.fail(`arrays don't match:\n  ${a}\n  ${b}`);
}

/** test WGSL compute shader with typical defaults */
export async function testCompute(
  src: string,
  elem: WgslElementType = "f32",
  conditions?: Record<string, boolean>,
  constants?: Record<string, string | number>,
) {
  const device = await getGPUDevice();
  return testComputeShader({
    projectDir,
    device,
    src,
    resultFormat: elem,
    conditions,
    constants,
  });
}

/** test WGSL fragment shader with typical defaults */
export async function testFragment(
  src: string,
  size?: [number, number],
  textureFormat: GPUTextureFormat = "rgba32float",
  conditions?: Record<string, boolean>,
  constants?: Record<string, string | number>,
) {
  const device = await getGPUDevice();
  return await testFragmentShader({
    projectDir,
    device,
    src,
    textureFormat,
    size,
    conditions,
    constants,
  });
}

// Re-export sprite sheet utilities from spriteTestUtil
export {
  createSpriteSheetTexture,
  createSimpleSpriteSheet,
} from "./spriteTestUtil.ts";

// Re-export texture helpers for convenience
export {
  createSolidTexture,
  createGradientTexture,
  createCheckerboardTexture,
  createSampler,
};
