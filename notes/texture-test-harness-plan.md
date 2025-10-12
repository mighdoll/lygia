# Texture Test Harness Implementation Plan

**Target audience**: wesl-debug developers
**Goal**: Add texture sampling support to fragment shader tests

## Current State

The existing `testFragmentShader` function in wesl-debug can:
- Create a render pipeline with a fragment shader
- Render to a texture
- Read back pixel values for validation

**Missing capability**: Cannot bind input textures and samplers for the shader to sample from.

## Required Features

### 1. Texture Creation Helper

Add a utility to create test textures with known patterns:

```typescript
interface TexturePattern {
  type: 'solid' | 'gradient' | 'checkerboard' | 'noise';
  width: number;
  height: number;
  format?: GPUTextureFormat;
}

/**
 * Create a test texture with a specific pattern
 * - solid: uniform color (e.g., [0.5, 0.5, 0.5, 1.0])
 * - gradient: horizontal gradient from black to white
 * - checkerboard: 2x2 black/white pattern
 * - noise: random values (seeded for reproducibility)
 */
function createTestTexture(
  device: GPUDevice,
  pattern: TexturePattern,
  data?: Float32Array | Uint8Array
): GPUTexture;
```

**Example patterns needed**:
- **Solid gray** (0.5, 0.5, 0.5, 1.0) - for testing basic sampling
- **Horizontal gradient** - for edge detection filters
- **Checkerboard** - for sharpen/blur filters
- **4x4 sprite sheet** - for sprite sampling tests

### 2. Sampler Creation Helper

Add a utility for creating samplers with common configurations:

```typescript
interface SamplerConfig {
  addressMode?: 'clamp-to-edge' | 'repeat' | 'mirror-repeat';
  filterMode?: 'nearest' | 'linear';
}

function createTestSampler(
  device: GPUDevice,
  config?: SamplerConfig
): GPUSampler;
```

**Common configs**:
- Linear filtering with clamp (default for most tests)
- Nearest filtering with repeat (for sprite sampling)

### 3. Enhanced Fragment Shader Test Function

Extend `testFragmentShader` to accept input textures and samplers:

```typescript
interface FragmentShaderTestOptions {
  projectDir: string;
  device: GPUDevice;
  src: string;
  textureFormat?: GPUTextureFormat;
  size?: [number, number];
  conditions?: Record<string, boolean>;

  // NEW: Input texture bindings
  inputTextures?: Array<{
    texture: GPUTexture;
    sampler: GPUSampler;
    binding: number;  // @binding(0), @binding(1), etc.
  }>;
}

async function testFragmentShader(
  options: FragmentShaderTestOptions
): Promise<Float32Array>;
```

### 4. Bind Group Setup

The implementation needs to:

1. Create a bind group layout matching the shader's `@group(0) @binding(N)` declarations
2. Create a bind group with the provided textures and samplers
3. Set the bind group before rendering

**Example shader binding pattern**:
```wgsl
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;
```

The implementation should automatically detect the bindings from the shader or accept explicit configuration.

## Implementation Strategy

### Phase 1: Basic Texture Support
1. Implement `createTestTexture` with 'solid' pattern only
2. Implement `createTestSampler` with default config
3. Extend `testFragmentShader` to accept a single input texture
4. Validate with a simple test (reading a solid color texture)

### Phase 2: Pattern Support
5. Add 'gradient', 'checkerboard', 'noise' patterns
6. Add ability to pass custom texture data

### Phase 3: Multiple Textures
7. Support multiple input textures (for advanced filters)
8. Auto-detect bindings from shader reflection

## Test Validation Strategy

After implementation, validate with these simple tests:

### Test 1: Solid Color Sampling
```wgsl
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_samp: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / 256.0;
  return textureSample(input_tex, input_samp, uv);
}
```
**Expected**: Output matches input texture color

### Test 2: Gradient Sampling
```wgsl
// Same bindings as above
@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = vec2f(0.5, 0.5);  // Center sample
  return textureSample(input_tex, input_samp, uv);
}
```
**Expected**: Output is middle gray (0.5) when sampling gradient center

### Test 3: UV Coordinate Verification
```wgsl
// Same bindings as above
@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / 256.0;
  let color = textureSample(input_tex, input_samp, uv);
  return vec4f(uv.x, uv.y, 0.0, 1.0);  // Return UV as color
}
```
**Expected**: Output matches expected UV gradient

## API Design Considerations

### Convenience Wrapper
Consider adding a high-level helper that combines texture creation and testing:

```typescript
async function testFragmentShaderWithTexture(
  device: GPUDevice,
  projectDir: string,
  src: string,
  pattern: TexturePattern,
  options?: {
    size?: [number, number];
    samplerConfig?: SamplerConfig;
  }
): Promise<Float32Array>;
```

This would internally:
1. Create the texture with the pattern
2. Create the sampler
3. Call `testFragmentShader` with bindings
4. Clean up resources

### Resource Management
- Textures and samplers should be created by the caller and cleaned up after test
- Consider adding a `cleanup()` helper or using try-finally blocks in examples

## Integration with LYGIA Tests

Once implemented, LYGIA tests will use it like this:

```typescript
import { getGPUDevice, createTestTexture, testFragmentShaderWithTexture } from "wesl-debug";

test("edgePrewitt", async () => {
  const device = await getGPUDevice();

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

  const result = await testFragmentShaderWithTexture(
    device,
    import.meta.url,
    src,
    { type: 'gradient', width: 256, height: 256 }
  );

  expect(result[0]).toBeGreaterThanOrEqual(0.0);
});
```

## Performance Considerations

- Texture creation should be cached when possible (e.g., same pattern used across tests)
- Consider lazy initialization for GPU device
- Small textures (64x64 or 128x128) are sufficient for most tests

## Documentation Needs

After implementation, document:
1. API reference for new functions
2. Common patterns (solid, gradient, checkerboard) with visual examples
3. Example tests showing typical usage
4. Troubleshooting guide (common binding errors, format mismatches)

## Non-Goals (Out of Scope)

- 3D texture support (LYGIA tests only need 2D)
- Cube map support
- Mipmap generation (use mip level 0 only)
- Texture arrays
- Storage textures (tests only need sampled textures)
- Custom texture upload formats (stick to f32 RGBA)

## Questions for wesl-debug Maintainers

1. Should texture creation be part of wesl-debug core or a separate test utilities package?
2. Preferred texture format for test patterns (rgba32float vs rgba8unorm)?
3. Should we use automatic bind group layout detection or require explicit configuration?
4. Do you want a resource pool/cache for commonly used test textures?
