import { afterAll, beforeAll, expect, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import {
  destroySharedDevice,
  getGPUDevice,
  testFragmentShaderImage,
} from "wesl-debug";
import perlinNoiseFbm from "./shaders/perlin-noise-fbm.wesl?raw";

// Setup image snapshot matcher
imageMatcher();

const projectDir = new URL("../../", import.meta.url).href;
let device: GPUDevice;

beforeAll(async () => {
  device = await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("Perlin noise FBM pattern", async () => {
  const result = await testFragmentShaderImage({
    projectDir,
    device,
    src: perlinNoiseFbm,
    size: [512, 512],
  });

  // Visual regression test - creates reference on first run
  await expect(result).toMatchImage("perlin-noise-fbm");
});
