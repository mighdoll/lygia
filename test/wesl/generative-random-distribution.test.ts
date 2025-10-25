import { test } from "vitest";
import { expectDistribution, testDistribution } from "./testUtil.ts";

// These tests validate that random functions produce uniform distributions
// with correct statistical properties (mean, uniformity across range).

test("random - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        test::results[i] = random(f32(i));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = random2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random3 - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random3;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 16u);
        let y = f32((i / 16u) % 16u);
        let z = f32(i / 256u);
        test::results[i] = random3(vec3f(x, y, z));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("srandom - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        // Vary inputs more to avoid patterns
        test::results[i] = srandom(f32(i) * 1.234 + 0.567);
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("srandom2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = srandom2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("random22 - distribution (x component)", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random22;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        let sample = random22(vec2f(x, y));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random33 - distribution (x component)", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random33;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 16u);
        let y = f32((i / 16u) % 16u);
        let z = f32(i / 256u);
        let sample = random33(vec3f(x, y, z));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("srandom22 - distribution (x component)", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom22;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        // Vary inputs more to avoid patterns
        let x = f32(i % 32u) * 1.1 + 0.3;
        let y = f32(i / 32u) * 1.3 + 0.7;
        let sample = srandom22(vec2f(x, y));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});
