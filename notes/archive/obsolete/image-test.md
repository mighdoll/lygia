# Image Test Infrastructure for wesl-debug

**Target Audience:** Developers working on `wesl-debug` package
**Purpose:** Add infrastructure for testing fragment shaders with uniforms, textures, and animation
**Status:** Design document for implementation

---

## Executive Summary

This document outlines extensions to `wesl-debug`'s `testFragmentShader()` to support:
1. **Uniform values** (resolution, time, custom parameters)
2. **Texture loading** from files (PNG/JPG)
3. **Animation testing** (single-frame and multi-frame)

These additions enable testing real-world shader use cases, particularly for libraries like LYGIA that have 112+ shader examples requiring these features.

**Key principle:** Keep `wesl-debug` LYGIA-agnostic. The infrastructure should work with any WGSL/WESL shaders.

---

## Context: Real-World Shader Requirements

Analysis of 112 shader examples from LYGIA (shader library used with glslViewer) reveals common patterns:

### Feature Usage Statistics
- **Resolution uniform**: ~99% of examples (512x512 typical)
- **Time parameter**: ~81% of examples (for animation/variation)
- **Input textures**: ~36% of examples (image processing, filters)
- **Mouse input**: ~30% of examples ✅ **Include** (cheap to add, decent coverage)
- **Pure computational**: ~11% of examples ✅ **Already work today**

### How Shaders Use Time

**Pattern 1: Declared but unused** (~30% of "animated" shaders)
```wgsl
// Import standard uniforms (time available but unused)
import test::Uniforms;

@group(0) @binding(0) var<uniform> u: test::Uniforms;
// time declared but never actually used in shader code
```
**Test strategy:** Single frame at time=0.0

**Pattern 2: Variation seed** (~50% of "animated" shaders)
```wgsl
// Animate noise/random patterns for visual interest
let noise = fbm(vec3f(position * 5.0, u.time));
let cells = voronoi(vec2f(st * 5.0 + u.time));
```
**Test strategy:** Single frame at fixed time (e.g., time=5.0)

**Pattern 3: True animation** (~20% of "animated" shaders)
```wgsl
// Cyclic parameter animation
let angle = u.time * 0.4;  // Continuous rotation
let phase = fract(u.time * 0.25);  // 4-second cycle
let size = 0.1 + sin(u.time) * 0.05;  // Oscillating size
```
**Test strategy:** Multi-frame validation (verify frames differ over time)

### How Shaders Use Mouse

**Pattern 1: Interactive positioning** (~60% of mouse examples)
```wgsl
// Position elements based on mouse
let dist = length(st - u.mouse);
let highlight = step(dist, 0.1);
```
**Test strategy:** Single frame with fixed mouse position (e.g., mouse=[0.5, 0.5])

**Pattern 2: Mouse as variation** (~40% of mouse examples)
```wgsl
// Use mouse to control parameters
let zoom = u.mouse.x * 5.0;
let rotation = u.mouse.y * 3.14159;
```
**Test strategy:** Test at 2-3 different mouse positions to verify behavior

### Key Insight: Resolution Uniform vs textureDimensions()

**`resolution` uniform = output viewport size**, not input texture size:
- Many shaders have no input textures (generative/math shaders)
- WGSL has no built-in to query output attachment size
- Used to normalize coordinates: `let st = position.xy / uniforms.resolution;`

**The `resolution` uniform is automatically populated from the `size` parameter** - users never need to specify it separately.

**Standard Uniforms module:** Tests use the standard `Uniforms` struct from the `test` package (provided by test framework):
```wgsl
@group(0) @binding(0) var<uniform> u: test::Uniforms;
// Provides: u.resolution, u.time, u.mouse
```

**Input texture sizes** can use `textureDimensions()`:
```wgsl
// Old GLSL way: uniform vec2 u_tex0Resolution;
// WGSL way: let tex_size = vec2<f32>(textureDimensions(input_tex)); ✅
```

---

## Requirements

### 1. Uniform Buffer Support
Create and bind a standard uniform buffer with common shader parameters.

**Uniform struct (provided by test framework as `test::Uniforms`):**
```wgsl
struct Uniforms {
  resolution: vec2f,  // Output viewport dimensions (auto from size param)
  time: f32,          // Elapsed time in seconds
  mouse: vec2f,       // Mouse position in [0,1] normalized coords
}
// Note: WGSL automatically adds implicit padding for alignment
// No need to declare _pad0 or _pad1 fields - WGSL handles this
```

**Binding convention:**
- Uniforms: `@group(0) @binding(0)`
- First texture: `@group(0) @binding(1)` (texture)
- First sampler: `@group(0) @binding(2)` (sampler)
- Additional textures: bindings 3,4,5,6... (texture, sampler pairs)

### 2. Auto-Population of Resolution
The `resolution` uniform is **automatically set** to match the `size` parameter:
```typescript
testFragmentShader({
  size: [512, 512],  // Creates 512x512 output texture
  uniforms: {
    time: 5.0,
    mouse: [0.5, 0.5],
    // resolution: [512, 512] auto-populated! ✨
  }
})
```

Users **never specify resolution manually** - it's always derived from `size`.

### 3. Texture Loading from Memory
For now, accept pre-created textures (file loading can be added later):
```typescript
testFragmentShader({
  inputTextures: [
    { texture: myGPUTexture, sampler: mySampler },
  ]
})
```

---

## API Design

### Extended `testFragmentShader()` Interface

```typescript
interface FragmentTestParams {
  /** WESL/WGSL fragment shader source */
  src: string;

  /** Project directory for import resolution */
  projectDir: string;

  /** GPU device */
  device: GPUDevice;

  /** Output texture format (default: "rgba32float") */
  outputFormat?: GPUTextureFormat;

  /** Output texture size (default: [1, 1])
   * Also auto-populates uniforms.resolution */
  size?: [width: number, height: number];

  /** Conditional compilation flags */
  conditions?: Record<string, boolean>;

  /** ✨ NEW: Uniform values */
  uniforms?: {
    /** Elapsed time in seconds (default: 0.0) */
    time?: number;

    /** Mouse position in [0,1] normalized coords (default: [0.0, 0.0]) */
    mouse?: [number, number];

    // Note: resolution is auto-populated from size parameter, not specified here
  };

  /** Input textures + samplers (existing parameter) */
  inputTextures?: Array<{
    texture: GPUTexture;
    sampler: GPUSampler;
  }>;
}

/** Returns pixel (0,0) color values */
async function testFragmentShader(
  params: FragmentTestParams
): Promise<number[]>
```

### Multi-Frame Testing Helper

```typescript
interface AnimatedTestParams extends FragmentTestParams {
  /** Time points to render (e.g., [0.0, 2.0, 4.0]) */
  timePoints: number[];
}

/** Returns array of frames, one per time point */
async function testAnimatedShader(
  params: AnimatedTestParams
): Promise<number[][]> {
  return await Promise.all(
    params.timePoints.map(time =>
      testFragmentShader({ ...params, uniforms: { ...params.uniforms, time } })
    )
  );
}
```

---

## Implementation Guide

### Phase 1: Uniform Buffer Infrastructure

**Step 0: Create standard Uniforms WESL module**

The test framework provides `test::Uniforms` - create `src/test/Uniforms.wesl`:
```wgsl
// Standard uniforms for fragment shader testing
// Available as test::Uniforms in all test shaders
export struct Uniforms {
  resolution: vec2f,  // Output viewport dimensions
  time: f32,          // Elapsed time in seconds
  mouse: vec2f,       // Mouse position [0,1] normalized coords
}
// Note: No explicit padding needed - WGSL handles alignment automatically
```

Tests use this:
```wgsl
@group(0) @binding(0) var<uniform> u: test::Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / u.resolution;
  // ... use u.time, u.mouse as needed
}
```

**Step 1: Define uniform buffer layout**

Create `src/UniformHelpers.ts`:
```typescript
/** Standard uniform buffer structure */
export interface StandardUniforms {
  time?: number;
  mouse?: [number, number];
  // Note: resolution is derived from outputSize, not passed in uniforms
}

/** Create uniform buffer with given values
 * @param outputSize - Output texture dimensions (becomes uniforms.resolution)
 * @param uniforms - User-provided uniform values (time, mouse)
 */
export function createUniformBuffer(
  device: GPUDevice,
  outputSize: [number, number],
  uniforms: StandardUniforms = {}
): GPUBuffer {
  const resolution = outputSize;
  const time = uniforms.time ?? 0.0;
  const mouse = uniforms.mouse ?? [0.0, 0.0];

  const buffer = device.createBuffer({
    label: "standard-uniforms",
    size: 32, // vec2f (8) + f32 (4) + implicit pad (4) + vec2f (8) + struct pad (8)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Layout matches WGSL struct with implicit padding
  // [resolution.xy, time, <implicit pad>, mouse.xy, <struct pad>]
  const data = new Float32Array([
    resolution[0],
    resolution[1],
    time,
    0.0, // implicit padding for vec2f alignment
    mouse[0],
    mouse[1],
    0.0, // struct padding to 32 bytes
    0.0, // struct padding to 32 bytes
  ]);

  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/** Create bind group layout for uniforms + textures */
export function createBindGroupLayout(
  device: GPUDevice,
  textureCount: number = 0
): GPUBindGroupLayout {
  const entries: GPUBindGroupLayoutEntry[] = [
    {
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform" },
    },
  ];

  // Add texture+sampler pairs at bindings 1,2,3,4...
  for (let i = 0; i < textureCount; i++) {
    entries.push(
      {
        binding: i * 2 + 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
      {
        binding: i * 2 + 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      }
    );
  }

  return device.createBindGroupLayout({ entries });
}

/** Create bind group with uniforms + textures */
export function createBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  textures?: Array<{ texture: GPUTexture; sampler: GPUSampler }>
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
  ];

  if (textures) {
    textures.forEach((tex, i) => {
      entries.push(
        { binding: i * 2 + 1, resource: tex.texture.createView() },
        { binding: i * 2 + 2, resource: tex.sampler }
      );
    });
  }

  return device.createBindGroup({ layout, entries });
}
```

**Step 2: Modify `TestFragmentShader.ts`**

Update `testFragmentShader()` to use uniforms:
```typescript
export async function testFragmentShader(
  params: FragmentTestParams,
): Promise<number[]> {
  const { projectDir, device, src, conditions = {} } = params;
  const { outputFormat = "rgba32float", size = [1, 1] } = params;
  const { inputTextures = [], uniforms = {} } = params;

  // Create uniform buffer (resolution auto-populated from size)
  const uniformBuffer = createUniformBuffer(device, size, uniforms);

  // Create bind group layout and bind group
  const bindGroupLayout = createBindGroupLayout(device, inputTextures.length);
  const bindGroup = createBindGroup(
    device,
    bindGroupLayout,
    uniformBuffer,
    inputTextures
  );

  // Compile shader
  const completeSrc = src + "\n\n" + fullscreenTriangleVertex;
  const shaderParams = { projectDir, device, src: completeSrc, conditions };
  const module = await compileShader(shaderParams);

  // Create pipeline with explicit layout
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module },
    fragment: {
      module,
      targets: [{ format: outputFormat }],
    },
    primitive: { topology: "triangle-list" },
  });

  // Create output texture and render
  const texture = device.createTexture({
    label: "fragment-test-output",
    size: { width: size[0], height: size[1], depthOrArrayLayers: 1 },
    format: outputFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  executeRenderPass(device, pipeline, texture, bindGroup);

  // Read back pixel (0,0)
  const extractCount = numComponents(outputFormat);
  const data = await withTextureCopy(device, texture, texData =>
    Array.from(texData.slice(0, extractCount)),
  );

  // Cleanup
  texture.destroy();
  uniformBuffer.destroy();

  return data;
}
```

**Step 3: Update `executeRenderPass()`**

Already accepts optional bindGroup - no changes needed if signature is:
```typescript
function executeRenderPass(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  texture: GPUTexture,
  bindGroup?: GPUBindGroup, // ✅ Already supports this
): void
```

### Phase 2: Add Tests

**Test uniform support** in `src/test/TestFragmentShader.test.ts`:

```typescript
test("shader with resolution uniform (auto-populated)", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let st = pos.xy / u.resolution;
      return vec4f(st.x, st.y, 0.0, 1.0);
    }
  `;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    // resolution auto-populated as [256, 256]
  });

  // Pixel (0,0) normalized: (0.5/256, 0.5/256, 0, 1)
  expect(result[0]).toBeCloseTo(0.5 / 256, 4);
  expect(result[1]).toBeCloseTo(0.5 / 256, 4);
  expect(result[2]).toBe(0.0);
  expect(result[3]).toBe(1.0);
});

test("shader with time uniform", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main() -> @location(0) vec4f {
      return vec4f(u.time, u.time * 2.0, u.time * 3.0, 1.0);
    }
  `;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    uniforms: { time: 5.0 },
  });

  expect(result[0]).toBeCloseTo(5.0);
  expect(result[1]).toBeCloseTo(10.0);
  expect(result[2]).toBeCloseTo(15.0);
});

test("shader with mouse uniform", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let st = pos.xy / u.resolution;
      let dist = length(st - u.mouse);
      return vec4f(dist, 0.0, 0.0, 1.0);
    }
  `;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    uniforms: { mouse: [0.5, 0.5] },
  });

  // Distance from (0.5/256, 0.5/256) to (0.5, 0.5)
  const st = [0.5 / 256, 0.5 / 256];
  const expectedDist = Math.sqrt(
    Math.pow(st[0] - 0.5, 2) + Math.pow(st[1] - 0.5, 2)
  );
  expect(result[0]).toBeCloseTo(expectedDist, 4);
});

test("multi-frame animation", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main() -> @location(0) vec4f {
      let phase = sin(u.time);
      return vec4f(phase, 0.0, 0.0, 1.0);
    }
  `;

  const frames = await testAnimatedShader({
    projectDir: import.meta.url,
    device,
    src,
    timePoints: [0.0, 1.57, 3.14], // 0, π/2, π
  });

  expect(frames[0][0]).toBeCloseTo(0.0, 2);    // sin(0) = 0
  expect(frames[1][0]).toBeCloseTo(1.0, 2);    // sin(π/2) = 1
  expect(frames[2][0]).toBeCloseTo(0.0, 2);    // sin(π) ≈ 0
});
```

**Test uniforms + textures together:**

```typescript
test("shader with uniforms and texture", async () => {
  const inputTex = createSolidTexture(device, [0.5, 0.5, 0.5, 1.0], 64, 64);
  const sampler = createSampler(device);

  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;
    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / u.resolution;
      let tex_color = textureSample(input_tex, input_samp, uv);
      let time_mod = vec4f(u.time * 0.1);
      return tex_color + time_mod;
    }
  `;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [64, 64],
    uniforms: { time: 10.0 },
    inputTextures: [{ texture: inputTex, sampler }],
  });

  // 0.5 (texture) + 1.0 (time * 0.1 where time=10) = 1.5 clamped to 1.0
  expect(result[0]).toBeCloseTo(1.5, 2);
});
```

### Phase 3: Export Helper Function

Add to `src/index.ts`:
```typescript
export * from "./UniformHelpers.ts";

// Convenience export for multi-frame testing
export async function testAnimatedShader(
  params: FragmentTestParams & { timePoints: number[] }
): Promise<number[][]> {
  const { timePoints, ...baseParams } = params;
  return await Promise.all(
    timePoints.map(time =>
      testFragmentShader({
        ...baseParams,
        uniforms: { ...baseParams.uniforms, time },
      })
    )
  );
}
```

---

## Testing Strategy

### For wesl-debug Test Suite

1. **Basic uniform support**
   - Test resolution uniform (auto-injection)
   - Test time uniform (explicit value)
   - Test both together

2. **Uniforms + textures**
   - Test uniform buffer coexists with texture bindings
   - Verify binding order (0=uniform, 1,2,3...=textures)

3. **Multi-frame animation**
   - Test `testAnimatedShader()` helper
   - Verify frames differ over time

4. **Edge cases**
   - Empty uniforms object (should still work)
   - size=[1,1] default (resolution auto-injected)
   - No textures (uniform-only shaders)

### For LYGIA Usage (Future)

Once infrastructure is in place, LYGIA can use it:

```typescript
// LYGIA test example (not part of wesl-debug)
test("lygia color/mix functions", async () => {
  const src = `
    import lygia::color::mixOklab;
    import lygia::color::mixSpectral;
    import lygia::color::mixRYB;

    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let st = pos.xy / u.resolution;
      let a = vec3f(0.0, 0.07, 0.16);
      let b = vec3f(0.843, 0.6, 0.0);

      var color: vec3f;
      if (st.y < 0.33) {
        color = mixOklab(a, b, st.x);
      } else if (st.y < 0.66) {
        color = mixSpectral(a, b, st.x);
      } else {
        color = mixRYB(a, b, st.x);
      }

      return vec4f(color, 1.0);
    }
  `;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [512, 512],
  });

  // Validate: produces reasonable color values
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
});
```

---

## Usage Examples (Independent of LYGIA)

### Example 1: Animated Pattern

```typescript
test("animated noise pattern", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    fn hash(p: vec2f) -> f32 {
      let p3 = fract(vec3f(p.x, p.y, p.x) * 0.13);
      return fract((p3.x + p3.y) * p3.z);
    }

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let st = pos.xy / u.resolution;
      let noise = hash(st * 10.0 + u.time);
      return vec4f(vec3f(noise), 1.0);
    }
  `;

  // Test that pattern changes over time
  const frame1 = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    uniforms: { time: 0.0 },
  });

  const frame2 = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    uniforms: { time: 5.0 },
  });

  // Verify frames are different (animation works)
  expect(frame1).not.toEqual(frame2);
});
```

### Example 2: Image Processing with Resolution

```typescript
test("image scaling aware shader", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;
    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let pixel_size = 1.0 / u.resolution;
      let st = pos.xy / u.resolution;

      // Simple 2x2 box blur
      var color = vec4f(0.0);
      color += textureSample(input_tex, input_samp, st + vec2f(-pixel_size.x, -pixel_size.y));
      color += textureSample(input_tex, input_samp, st + vec2f( pixel_size.x, -pixel_size.y));
      color += textureSample(input_tex, input_samp, st + vec2f(-pixel_size.x,  pixel_size.y));
      color += textureSample(input_tex, input_samp, st + vec2f( pixel_size.x,  pixel_size.y));

      return color * 0.25;
    }
  `;

  const inputTex = createGradientTexture(device, 128, 128, "horizontal");
  const sampler = createSampler(device);

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [128, 128],
    inputTextures: [{ texture: inputTex, sampler }],
  });

  // Verify blurred output
  expect(result[0]).toBeGreaterThan(0.0);
});
```

### Example 3: Coordinate Normalization

```typescript
test("coordinate systems with resolution", async () => {
  const src = `
    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Method 1: Manual normalization
      let st = pos.xy / u.resolution;

      // Method 2: Center-based coordinates
      let centered = (pos.xy - u.resolution * 0.5) / u.resolution.y;

      return vec4f(st, 0.0, 1.0);
    }
  `;

  const result = await testFragmentShader({
    projectDir: import.meta.url,
    device,
    src,
    size: [512, 512],
  });

  // At pixel (0,0), normalized coords are (0.5/512, 0.5/512)
  expect(result[0]).toBeCloseTo(0.5 / 512, 5);
  expect(result[1]).toBeCloseTo(0.5 / 512, 5);
});
```

---

## Future Enhancements (Out of Scope)

These can be added later as needs arise:

1. **Texture loading from files**
   ```typescript
   inputTextureFiles: ["path/to/texture.png"]
   ```

2. **Frame sequence export**
   ```typescript
   exportFrameSequence({
     outputDir: "./frames",
     frameCount: 120,
     fps: 30,
   })
   ```

3. **Reference image comparison**
   ```typescript
   testFragmentShader({
     expectedImage: "./reference/output.png",
     threshold: 0.01,
   })
   ```

---

## Summary

**Minimal implementation** (Phase 1-2):
- Add `UniformHelpers.ts` with uniform buffer utilities
- Modify `testFragmentShader()` to accept `uniforms` parameter
- Auto-inject `resolution` from output size
- Update bind group creation to include uniforms at binding 0
- Add tests for uniform support

**Result:** Enables testing ~95% of real-world shader use cases with single-frame static rendering at fixed time values.

**Multi-frame testing** (Phase 3):
- Add `testAnimatedShader()` helper for convenience
- Enables validation of true animation shaders (~20% of examples)

This infrastructure keeps `wesl-debug` generic and reusable while enabling LYGIA (and other shader libraries) to create comprehensive test suites.
