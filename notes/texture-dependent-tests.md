# Texture-Dependent Skipped Tests

**Status**: Awaiting texture test harness implementation in wesl-debug
**Tracking**: See [texture-test-harness-plan.md](./texture-test-harness-plan.md)

## Overview

8 tests are currently skipped because they require texture/sampler bindings, which are not yet supported by the test infrastructure.

All these tests use **fragment shaders** since they need to:
- Sample from textures using `textureSample()`
- Use UV coordinates
- Process 2D image data

---

## Edge Detection Filters (1 test)

### 1. edgePrewitt
**File**: `test/wesl/filter-edge.test.ts:6`
**WESL function**: `lygia::filter::edge::prewitt::edgePrewitt`

**What it does**: Applies Prewitt edge detection operator (3x3 convolution kernel)

**Test requirements**:
- Input texture: Gradient or checkerboard pattern (to detect edges)
- Recommended: 256x256 horizontal gradient (black to white)
- Expected output: Edge strength values (higher at gradient center)

**Fix once harness exists**:
```typescript
test("edgePrewitt", async () => {
  const device = await getGPUDevice();
  const texture = createTestTexture(device, {
    type: 'gradient',
    width: 256,
    height: 256
  });
  const sampler = createTestSampler(device);

  const src = `
    import lygia::filter::edge::prewitt::edgePrewitt;

    @group(0) @binding(0) var input_tex: texture_2d<f32>;
    @group(0) @binding(1) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0);
      let edge = edgePrewitt(input_tex, input_samp, uv, pixel_size);
      return vec4f(edge, 1.0);
    }`;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    inputTextures: [{ texture, sampler, binding: 0 }]
  });

  // Edge detection should produce values in [0, 1] range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // At horizontal gradient, expect moderate edge response
  expect(result[0]).toBeGreaterThan(0.1);
});
```

---

## Sharpen Filters (5 tests)

All sharpen filters use similar patterns - they sample neighboring pixels to enhance edges.

### 2. sharpenAdaptive
**File**: `test/wesl/filter-sharpen.test.ts:6`
**WESL function**: `lygia::filter::sharpen::adaptive::sharpenAdaptive`

**Test requirements**:
- Input: Checkerboard or gradient pattern
- Expected: Sharpened version (enhanced edges)

### 3. sharpenAdaptive4
**File**: `test/wesl/filter-sharpen.test.ts:25`
**WESL function**: `lygia::filter::sharpen::adaptive::sharpenAdaptive4`

**Same as above, but returns vec4 with alpha channel preserved**

### 4. sharpenContrastAdaptive
**File**: `test/wesl/filter-sharpen.test.ts:44`
**WESL function**: `lygia::filter::sharpen::adaptive::sharpenContrastAdaptive`

**Test requirements**:
- Input: Same as sharpenAdaptive
- Takes additional `strength` parameter (test with 1.0)

### 5. sharpenFast
**File**: `test/wesl/filter-sharpen.test.ts:64`
**WESL function**: `lygia::filter::sharpen::fast::sharpenFast`

**Test requirements**:
- Input: Same as sharpenAdaptive
- Optimized version, should produce similar results

### 6. sharpenFast4
**File**: `test/wesl/filter-sharpen.test.ts:83`
**WESL function**: `lygia::filter::sharpen::fast::sharpenFast4`

**Same as sharpenFast, but returns vec4**

**Unified fix template for all sharpen tests**:
```typescript
test("sharpen*", async () => {
  const device = await getGPUDevice();
  const texture = createTestTexture(device, {
    type: 'checkerboard',  // or 'gradient'
    width: 256,
    height: 256
  });
  const sampler = createTestSampler(device, {
    filterMode: 'linear',
    addressMode: 'clamp-to-edge'
  });

  const src = `
    import lygia::filter::sharpen::*::sharpen*;

    @group(0) @binding(0) var input_tex: texture_2d<f32>;
    @group(0) @binding(1) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let pixel_size = vec2f(1.0 / 256.0);
      return sharpen*(input_tex, input_samp, uv, pixel_size);
      // For *4 variants: already returns vec4
      // For *Contrast: add strength parameter: ..., pixel_size, 1.0);
    }`;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    inputTextures: [{ texture, sampler, binding: 0 }]
  });

  // Sharpened output should be in valid range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
});
```

---

## Animation Utilities (1 test)

### 7. spriteLoop
**File**: `test/wesl/animation-misc.test.ts:8`
**WESL function**: `lygia::animation::spriteLoop::spriteLoop`

**What it does**: Samples from a sprite sheet (grid of animation frames) and loops through them

**Test requirements**:
- Input texture: 4x4 sprite sheet (16 frames)
- Each cell should have distinct color/pattern
- Test parameters: grid size (4x4), time values for frame selection

**Recommended test texture**: 4x4 grid where each cell is a different solid color

**Fix once harness exists**:
```typescript
test("spriteLoop", async () => {
  const device = await getGPUDevice();

  // Create 4x4 sprite sheet (256x256 texture, each sprite is 64x64)
  // Frame 0: red, Frame 1: green, Frame 2: blue, etc.
  const spriteData = createSpriteSheetTexture(4, 4);
  const texture = createTestTexture(device, {
    type: 'custom',
    width: 256,
    height: 256,
    data: spriteData
  });
  const sampler = createTestSampler(device);

  const src = `
    import lygia::animation::spriteLoop::spriteLoop;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);  // 4x4 sprite grid
      let time = 0.0;
      let frames = 8.0;  // Use first 8 frames
      let fps = 2.0;
      return spriteLoop(sprite_tex, sprite_samp, uv, grid, time, frames, fps);
    }`;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    inputTextures: [{ texture, sampler, binding: 0 }]
  });

  // At time=0, should sample frame 0 (red in this example)
  expect(result[0]).toBeGreaterThan(0.9);  // Red channel
  expect(result[1]).toBeLessThan(0.1);     // Green channel
});
```

**Helper needed**: `createSpriteSheetTexture(cols, rows)` that generates a test sprite sheet with distinct colors per frame.

---

## Sample Utilities (1 test)

### 8. sampleSprite
**File**: `test/wesl/sample.test.ts:6`
**WESL function**: `lygia::sample::sprite::sampleSprite`

**What it does**: Samples a specific frame from a sprite sheet (no animation, just static frame selection)

**Test requirements**:
- Input texture: Same 4x4 sprite sheet as spriteLoop
- Test parameters: grid size, frame index

**Fix once harness exists**:
```typescript
test("sampleSprite", async () => {
  const device = await getGPUDevice();
  const texture = createSpriteSheetTexture(4, 4);  // Reuse helper
  const sampler = createTestSampler(device);

  const src = `
    import lygia::sample::sprite::sampleSprite;

    @group(0) @binding(0) var sprite_tex: texture_2d<f32>;
    @group(0) @binding(1) var sprite_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 256.0;
      let grid = vec2f(4.0, 4.0);
      let frame = 2.0;  // Sample frame 2 (3rd frame, 0-indexed)
      return sampleSprite(sprite_tex, sprite_samp, uv, grid, frame);
    }`;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    inputTextures: [{ texture, sampler, binding: 0 }]
  });

  // Should sample frame 2 (blue in our example scheme)
  expect(result[2]).toBeGreaterThan(0.9);  // Blue channel
});
```

---

## Summary Table

| Test | File | Category | Input Pattern | Difficulty |
|------|------|----------|---------------|------------|
| edgePrewitt | filter-edge.test.ts:6 | Edge Detection | Gradient | Easy |
| sharpenAdaptive | filter-sharpen.test.ts:6 | Sharpen | Checkerboard | Easy |
| sharpenAdaptive4 | filter-sharpen.test.ts:25 | Sharpen | Checkerboard | Easy |
| sharpenContrastAdaptive | filter-sharpen.test.ts:44 | Sharpen | Checkerboard | Easy |
| sharpenFast | filter-sharpen.test.ts:64 | Sharpen | Checkerboard | Easy |
| sharpenFast4 | filter-sharpen.test.ts:83 | Sharpen | Checkerboard | Easy |
| spriteLoop | animation-misc.test.ts:8 | Animation | Sprite Sheet | Medium |
| sampleSprite | sample.test.ts:6 | Sampling | Sprite Sheet | Medium |

**Total time estimate after harness**: 2-3 hours to unskip all 8 tests

---

## Implementation Checklist

- [ ] Texture harness implemented in wesl-debug
- [ ] Test patterns implemented: solid, gradient, checkerboard
- [ ] Sprite sheet helper implemented (for animation tests)
- [ ] edgePrewitt test fixed
- [ ] sharpenAdaptive test fixed
- [ ] sharpenAdaptive4 test fixed
- [ ] sharpenContrastAdaptive test fixed
- [ ] sharpenFast test fixed
- [ ] sharpenFast4 test fixed
- [ ] spriteLoop test fixed
- [ ] sampleSprite test fixed
- [ ] All tests passing
- [ ] Documentation updated
