import { expect } from "vitest";
import type { WgslElementType } from "wesl-debug";
import {
  createCheckerboardTexture,
  createGradientTexture,
  createSampler,
  createSolidTexture,
  getGPUDevice,
  testComputeShader,
  testFragmentShader,
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

/**
 * Get the byte size for a WGSL element type
 */
function getElementStride(elem: WgslElementType): number {
  switch (elem) {
    case "f32":
    case "i32":
    case "u32":
      return 4;
    case "vec2f":
    case "vec2i":
    case "vec2u":
      return 8;
    case "vec3f":
    case "vec3i":
    case "vec3u":
      return 12;
    case "vec4f":
    case "vec4i":
    case "vec4u":
      return 16;
    default:
      return 4; // default to f32 size
  }
}

/**
 * Test distribution properties of a random function.
 * Collects multiple samples and returns them for statistical analysis.
 *
 * @param src - WESL shader source that writes samples to test::results
 * @param sampleCount - Number of samples to collect
 * @param elem - Element type (default "f32")
 * @param constants - Constants to pass via constants:: namespace (e.g., SAMPLE_COUNT)
 * @returns Array of sample values
 */
export async function testDistribution(
  src: string,
  sampleCount: number,
  elem: WgslElementType = "f32",
  constants?: Record<string, string | number>,
): Promise<number[]> {
  const device = await getGPUDevice();
  const bufferSize = sampleCount * getElementStride(elem);

  return testComputeShader({
    projectDir,
    device,
    src,
    resultFormat: elem,
    size: bufferSize,
    constants,
  });
}

/**
 * Validate that samples follow a uniform distribution.
 *
 * @param samples - Array of sample values
 * @param range - Expected [min, max] range
 * @param options - Test options (meanTolerance, bucketTolerance)
 */
export function expectDistribution(
  samples: number[],
  range: [number, number],
  options: {
    meanTolerance?: number;
    bucketTolerance?: number;
  } = {},
): void {
  const { meanTolerance = 0.05, bucketTolerance = 0.03 } = options;
  const [min, max] = range;
  const expectedMean = (min + max) / 2;

  // Test 1: Mean
  const actualMean = samples.reduce((sum, v) => sum + v, 0) / samples.length;
  const meanDiff = Math.abs(actualMean - expectedMean);
  if (meanDiff > meanTolerance) {
    expect.fail(
      `Mean ${actualMean.toFixed(4)} differs from expected ${expectedMean} by ${meanDiff.toFixed(4)} (threshold: ${meanTolerance})`,
    );
  }

  // Test 2: Bucket distribution
  const bucketCount = 10;
  const buckets = new Array(bucketCount).fill(0);
  const bucketWidth = (max - min) / bucketCount;

  for (const value of samples) {
    const bucketIndex = Math.min(
      Math.floor((value - min) / bucketWidth),
      bucketCount - 1,
    );
    buckets[bucketIndex]++;
  }

  const expectedRatio = 1.0 / bucketCount; // 0.1 for 10 buckets

  for (let i = 0; i < bucketCount; i++) {
    const ratio = buckets[i] / samples.length;
    const diff = Math.abs(ratio - expectedRatio);

    if (diff > bucketTolerance) {
      const bucketRange = [
        (min + i * bucketWidth).toFixed(2),
        (min + (i + 1) * bucketWidth).toFixed(2),
      ];
      expect.fail(
        `Bucket ${i} [${bucketRange[0]}, ${bucketRange[1]}) has ${(ratio * 100).toFixed(1)}% of samples (expected ${expectedRatio * 100}% ± ${bucketTolerance * 100}%)\n` +
          `Distribution: ${buckets.map((b) => ((b / samples.length) * 100).toFixed(1) + "%").join(", ")}`,
      );
    }
  }
}

// Re-export texture helpers for convenience
export {
  createCheckerboardTexture,
  createGradientTexture,
  createSampler,
  createSolidTexture,
};
