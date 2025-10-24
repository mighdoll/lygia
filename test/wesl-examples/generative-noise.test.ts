import { afterAll, beforeAll, expect, test } from "vitest";
import {
  getGPUDevice,
  destroySharedDevice,
  testFragmentShaderImage,
} from "wesl-debug";
import { imageMatcher } from "vitest-image-snapshot";

// Setup image snapshot matcher
imageMatcher();

const projectDir = import.meta.url;
let device: GPUDevice;

beforeAll(async () => {
  device = await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("Perlin noise FBM pattern", async () => {
  const src = `
    import lygia::generative::cnoise::cnoise2;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let st = pos.xy / 512.0;  // Normalize coordinates

      // Multi-octave FBM (Fractional Brownian Motion)
      var n = 0.0;
      var amplitude = 0.5;
      var frequency = 3.0;

      for (var i = 0; i < 5; i++) {
        n += cnoise2(st * frequency) * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
      }

      // Map [-1, 1] noise to [0, 1] color range
      let color = vec3f(n * 0.5 + 0.5);
      return vec4f(color, 1.0);
    }
  `;

  const result = await testFragmentShaderImage({
    projectDir,
    device,
    src,
    size: [512, 512],
  });

  // Visual regression test - creates reference on first run
  await expect(result).toMatchImage("perlin-noise-fbm");
});
