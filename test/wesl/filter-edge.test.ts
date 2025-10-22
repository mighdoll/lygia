import { expect, test } from "vitest";
import {
  createGradientTexture,
  createSampler,
  createSolidTexture,
  getGPUDevice,
  testFragmentShader,
} from "wesl-debug";
import { expectCloseTo } from "./testUtil.js";

const projectDir = import.meta.url;

// Edge detection filter tests - require texture/sampler, use fragment shaders

test("edgePrewitt", async () => {
  const device = await getGPUDevice();
  const gradientTex = createGradientTexture(device, 256, 256, "horizontal");
  const solidTex = createSolidTexture(device, [0.5, 0.5, 0.5, 1.0], 256, 256);
  const sampler = createSampler(device);

  // Test 1: Horizontal gradient should produce strong horizontal edge response
  const src1 = `
    import lygia::filter::edge::prewitt::edgePrewitt;

    @group(0) @binding(0) var input_tex: texture_2d<f32>;
    @group(0) @binding(1) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0, 1.0 / 256.0);
      let edge = edgePrewitt(input_tex, input_samp, uv, pixel_size);
      return vec4f(edge, 1.0);
    }`;

  const gradientResult = await testFragmentShader({
    projectDir,
    device,
    src: src1,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: gradientTex, sampler }],
  });

  // Test 2: Solid color should produce near-zero edge response
  const solidResult = await testFragmentShader({
    projectDir,
    device,
    src: src1,
    textureFormat: "rgba32float",
    size: [256, 256],
    inputTextures: [{ texture: solidTex, sampler }],
  });

  // Gradient should produce significant edge magnitude (Prewitt detects horizontal edges)
  // For a uniform gradient from 0 to 1 over 256 pixels, the Prewitt operator
  // computes the gradient magnitude which should be approximately 1/256 * kernel_sum
  expect(gradientResult[0]).toBeGreaterThan(0.01);
  expect(gradientResult[0]).toBeLessThan(1.0);

  // Solid color should produce very small edge magnitude (near zero)
  expect(solidResult[0]).toBeLessThan(0.01);

  // Edge detection on gradient should be significantly stronger than on solid
  expect(gradientResult[0]).toBeGreaterThan(solidResult[0] * 10);

  // Regression test - exact values to catch implementation changes
  expectCloseTo([0.011764707043766975], [gradientResult[0]]);
  expectCloseTo([1.1920928955078125e-7], [solidResult[0]]);
});
