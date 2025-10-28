# Test Infrastructure Gaps Analysis

**Date:** 2025-10-28
**Purpose:** Document what test infrastructure exists, what's missing, and prioritize additions by impact

---

## Executive Summary

**Key Finding:** Current test infrastructure is more capable than documented! 17 deferred files can be converted immediately with existing tools.

**Infrastructure Status:**
- ✅ **Exists:** Fragment shader testing, visual regression, texture inputs, basic texture helpers
- ⚠️ **Missing (high impact):** Depth texture helper (unlocks 4 files, 2-3 hours)
- ⚠️ **Missing (medium impact):** Multi-pass rendering (unlocks 10 files, 2-3 days)
- ⏸️ **Missing (low priority):** G-buffer generation (unlocks 5-8 files, 1-2 weeks)

---

## Current Capabilities ✅

### Fragment Shader Testing

**Available:** `testFragment()` function
**Location:** `test/wesl/testUtil.ts`
**Capabilities:**
- Fragment shader execution with `@builtin(position)`
- Derivative support (`fwidth()`, `dpdx()`, `dpdy()`)
- Minimum size `[2, 2]` for derivative functions
- Returns pixel (0,0) RGBA values for validation

**Example:**
```typescript
const result = await testFragment(src, {
  size: [256, 256],
  inputTextures: [{ texture: gradientTex, sampler }]
});
expectCloseTo([0.5, 0.25], [result[0], result[1]]);
```

**Used by:** Anti-aliasing functions, derivative-based filters, texture sampling tests

---

### Visual Regression Testing

**Available:** `testFragmentShaderImage()` and `toMatchImage()`
**Location:** `test/wesl-examples/`, `vitest-image-snapshot` package
**Capabilities:**
- Renders fragment shader to texture
- Captures as PNG
- Compares against baseline snapshots
- Generates diff images on failure
- Supports custom threshold settings

**Example:**
```typescript
import { imageMatcher } from "vitest-image-snapshot";
imageMatcher(); // Call once at top of test file

test("Kuwahara oil painting effect", async () => {
  const device = await getGPUDevice();
  const result = await testFragmentShaderImage({
    projectDir: import.meta.url,
    device,
    src: kuwaharaShader,
    size: [512, 512],
    inputTextures: [{ texture: lemurTexture(device), sampler }]
  });
  await expect(result).toMatchImage("kuwahara-oil-painting");
});
```

**Used by:** Noise patterns, generative textures, complex filters
**Files unblocked:** 10 visual validation files (filter + distort categories)

---

### Texture Input Binding

**Available:** `inputTextures` parameter
**Location:** Both `testFragment()` and `testFragmentShaderImage()`
**Capabilities:**
- Bind multiple textures (array support)
- Automatic binding slot assignment
- Sampler configuration per texture
- Works with all texture helpers

**Example:**
```typescript
// Dual texture input (e.g., displacement mapping)
const result = await testFragment(displaceSrc, {
  inputTextures: [
    { texture: colorTexture, sampler: linearSampler },
    { texture: velocityTexture, sampler: nearestSampler }
  ],
  size: [256, 256]
});
```

**Files unblocked:** All texture sampling utilities, filters, distortions

---

### Texture Helpers

**Available:** Multiple texture generation functions
**Location:** `wesl-debug/src/texture-helpers.ts`

**Basic Helpers:**
- `solidTexture(device, r, g, b, a)` - Solid color texture
- `gradientTexture(device, w, h, direction)` - Linear gradient (horizontal, vertical, diagonal)
- `checkerboardTexture(device, w, h, squareSize)` - Checkerboard pattern
- `radialGradientTexture(device, w, h)` - Radial gradient from center

**Advanced Helpers:**
- `edgePatternTexture(device, w, h)` - Black/white edges for edge detection tests
- `colorBarsTexture(device, w, h)` - RGB color bars
- `noiseTexture(device, w, h, seed)` - Random noise pattern
- `lemurTexture(device)` - Bundled 512×512 test photo

**Custom Helpers:**
- `pngToTexture(device, pngPath)` - Load PNG file as texture

**Example:**
```typescript
// Test edge detection with gradient
const gradient = gradientTexture(device, 256, 256, "horizontal");
const result = await testFragment(sobelSrc, {
  inputTextures: [{ texture: gradient, sampler }],
  size: [256, 256]
});
// Expect strong edges at gradient boundaries
```

**Files unblocked:** All filter and sample utilities requiring test inputs

---

### Sampler Creation

**Available:** `createSampler()` function
**Location:** `wesl-debug/src/texture-helpers.ts`
**Capabilities:**
- Configurable address modes (clamp, repeat, mirror)
- Filter modes (linear, nearest)
- Mipmap support

**Example:**
```typescript
const linearSampler = createSampler(device, {
  addressMode: "clamp-to-edge",
  magFilter: "linear",
  minFilter: "linear"
});

const nearestSampler = createSampler(device, {
  addressMode: "repeat",
  magFilter: "nearest",
  minFilter: "nearest"
});
```

---

## Missing Infrastructure ❌

Prioritized by files unlocked and implementation effort.

---

### HIGH PRIORITY: Depth Texture Helper

**Status:** ❌ Missing
**Effort:** 2-3 hours
**Files unlocked:** 4 files
**Impact:** High - enables shadow/depth sampling tests

**What's needed:**
```typescript
// In wesl-debug/src/texture-helpers.ts
export function depthTexture(
  device: GPUDevice,
  width: number,
  height: number,
  pattern: "linear" | "radial" | "step" | "random" = "linear"
): GPUTexture {
  // Generate depth values (0.0 to 1.0) based on pattern
  // - linear: depth increases horizontally or vertically
  // - radial: depth increases from center
  // - step: discrete depth layers
  // - random: noisy depth for testing edge cases

  // Create texture with format: "depth24plus" or "depth32float"
  // Return GPUTexture suitable for shadow map / depth buffer testing
}
```

**Files unblocked:**
1. `sample/shadow.glsl` - Shadow map sampling
2. `sample/shadowLerp.glsl` - PCF shadow lerp
3. `sample/shadowPCF.glsl` - 25-sample soft shadows
4. `sample/viewPosition.glsl` - Depth → view space conversion

**Test pattern:**
```typescript
test("shadow PCF sampling", async () => {
  const device = await getGPUDevice();
  const depth = depthTexture(device, 512, 512, "radial");
  const sampler = createSampler(device, { addressMode: "clamp-to-edge" });

  const result = await testFragment(shadowPCFSrc, {
    inputTextures: [{ texture: depth, sampler }],
    size: [512, 512]
  });

  // Validate soft shadow penumbra
  expect(result[0]).toBeGreaterThan(0.0); // Not fully shadowed
  expect(result[0]).toBeLessThan(1.0);    // Not fully lit
});
```

**Implementation notes:**
- Support both `depth24plus` and `depth32float` formats
- Consider adding `depth24plus-stencil8` for stencil tests
- Pattern generation should create meaningful test cases (gradients, layers, etc.)
- May need comparison sampler support for shadow sampling

---

### MEDIUM PRIORITY: Multi-Pass Rendering

**Status:** ❌ Missing
**Effort:** 2-3 days
**Files unlocked:** 10 files
**Impact:** Medium - enables iterative algorithms

**What's needed:**
```typescript
// In test/wesl/testUtil.ts
export async function testFragmentShaderMultiPass(options: {
  passes: Array<{
    src: string;
    inputTextures?: Array<{ texture: GPUTexture; sampler: GPUSampler }>;
    outputSize?: [number, number];
  }>;
  iterations?: number; // For iterative algorithms
  size: [number, number];
}): Promise<Float32Array> {
  // Pass 1: Render to texture A
  // Pass 2: Use texture A as input, render to texture B
  // Pass 3: Use texture B as input, render to texture A
  // ... continue for N passes
  // Return final output as Float32Array
}
```

**Files unblocked:**

**Morphological (1 file):**
1. `morphological/jumpFlood.glsl` - Jump flood algorithm
   - Encode → Iterate N times → Decode
   - Each iteration: read previous result, update, write to next

**Simulate (3 files):**
2. `simulate/latticeBoltzmann.glsl` - Physics simulation
3. `simulate/simpleAndFastFluid.glsl` - Fluid dynamics
4. `simulate/grayscott.glsl` - Reaction-diffusion

**Lighting (6+ files):**
5. `lighting/ssr.glsl` - Screen-space reflections (ray march accumulation)
6. `lighting/ssao.glsl` - Screen-space ambient occlusion (multi-sample)
7. Others requiring G-buffer inputs

**Test pattern:**
```typescript
test("jumpFlood distance field", async () => {
  const device = await getGPUDevice();
  const input = edgePatternTexture(device, 256, 256);

  const result = await testFragmentShaderMultiPass({
    passes: [
      { src: jumpFloodEncodeSrc, inputTextures: [{ texture: input, sampler }] },
      { src: jumpFloodStepSrc, iterations: 8 }, // 8 jump iterations
      { src: jumpFloodDecodeSrc }
    ],
    size: [256, 256]
  });

  // Validate distance field properties
  expect(result[0]).toBeCloseTo(0.0, 0.1); // At edge
  expect(result[128]).toBeGreaterThan(50.0); // Far from edge
});
```

**Implementation notes:**
- Ping-pong rendering between two textures
- Support both fixed iteration count and convergence detection
- Handle feedback loops (output becomes next input)
- Consider exposing intermediate results for debugging

---

### LOW PRIORITY: G-Buffer Helpers

**Status:** ❌ Missing
**Effort:** 1-2 weeks
**Files unlocked:** 5-8 files
**Impact:** Low - only advanced lighting effects

**What's needed:**
```typescript
// In wesl-debug/src/gbuffer-helpers.ts
export interface GBuffer {
  depth: GPUTexture;
  normal: GPUTexture;
  position: GPUTexture;
  albedo: GPUTexture;
  // Optional:
  roughness?: GPUTexture;
  metallic?: GPUTexture;
}

export function generateGBuffer(
  device: GPUDevice,
  scene: "sphere" | "cube" | "plane" | "cornell-box",
  camera: { position: vec3, target: vec3, fov: number },
  size: [number, number]
): GBuffer {
  // Render simple scene to G-buffer
  // Return all buffers needed for deferred shading tests
}
```

**Files unblocked:**
1. `lighting/ssao.glsl` - Screen-space ambient occlusion
2. `lighting/volumetricLightScattering.glsl` - God rays
3. `lighting/sphereMap.glsl` - Environment mapping
4. `lighting/shadow.glsl` - Shadow mapping (scene-based)
5. `lighting/ssr.glsl` - Screen-space reflections
6. Others in lighting/ category

**Recommendation:** Defer until core library complete. These are advanced effects with complex setup requirements.

---

## Infrastructure Impact Summary

### By Files Unlocked

| Infrastructure | Effort | Files Unlocked | Priority |
|---------------|--------|----------------|----------|
| **None (exists!)** | 0 hours | **17 files** | ⭐⭐⭐⭐⭐ |
| Depth texture helper | 2-3 hours | 4 files | ⭐⭐⭐⭐ |
| Multi-pass rendering | 2-3 days | 10 files | ⭐⭐⭐ |
| G-buffer generation | 1-2 weeks | 5-8 files | ⭐ |

### By Category

**Can Convert Immediately (17 files):**
- Filter: kuwahara, smartDeNoise, radialBlur, fibonacciBokeh (4 files)
- Distort: all 6 files
- Morphological: erosion, dilation, alphaFill (3 files)
- Generative: fbm, voronoise, voronoi (3 files)
- Simulate: ripple (1 file)

**Need Depth Helper (4 files):**
- Sample: shadow, shadowLerp, shadowPCF, viewPosition

**Need Multi-Pass (10 files):**
- Morphological: jumpFlood
- Simulate: latticeBoltzmann, fluid, grayscott
- Lighting: ssr, ssao, + others

**Need G-Buffer (5-8 files):**
- Lighting: advanced effects only

**Truly Deferred (21 files):**
- GlslViewer integration (8 files) - out of scope
- Complex scene dependencies (6 files) - need full renderer
- Other specialized cases (7 files)

---

## Recommended Implementation Order

### Week 1 (Immediate)
1. ✅ **Convert 17 files** using existing infrastructure (0 hours infrastructure work!)
   - Write visual regression tests for filter/distort files
   - Write unit tests for morphological/generative/simulate files
   - Estimated: 1-2 days conversion work

### Week 2 (High Priority)
1. **Add depth texture helper** (2-3 hours)
   - Implement `depthTexture()` with multiple patterns
   - Add tests for depth texture generation
2. **Convert 4 shadow/depth files** (4-6 hours)
   - shadow, shadowLerp, shadowPCF, viewPosition
   - Write tests using depth texture helper

**Total Week 2 effort:** ~1 day

### Week 3-4 (Medium Priority)
1. **Add multi-pass rendering support** (2-3 days)
   - Implement `testFragmentShaderMultiPass()`
   - Add ping-pong texture support
   - Test with jumpFlood algorithm
2. **Convert iterative files** (2-3 days)
   - jumpFlood, simulation algorithms
   - Add visual regression tests for simulation outputs

**Total Week 3-4 effort:** ~1 week

### Later (Low Priority)
1. **G-buffer generation** (1-2 weeks)
   - Only if advanced lighting becomes priority
   - Can defer indefinitely

---

## Testing Infrastructure Checklist

### Currently Available ✅
- [x] Fragment shader testing (`testFragment`)
- [x] Visual regression testing (`toMatchImage`)
- [x] Texture input binding (`inputTextures`)
- [x] Basic texture helpers (gradient, checkerboard, solid, radial)
- [x] Advanced texture helpers (edge, noise, color bars, photo)
- [x] Sampler creation with filtering/wrapping modes
- [x] Compute shader testing (`testCompute`)
- [x] Image snapshot management (baseline, diff, actual)

### High Priority (Week 2) ⚠️
- [ ] Depth texture helper with multiple patterns
- [ ] Comparison sampler support (for shadow sampling)
- [ ] Depth format texture creation utilities

### Medium Priority (Week 3-4) ⚠️
- [ ] Multi-pass fragment shader testing
- [ ] Ping-pong texture rendering
- [ ] Iterative algorithm support
- [ ] Intermediate result extraction

### Low Priority (Later) ⏸️
- [ ] G-buffer generation from simple scenes
- [ ] Camera parameter utilities
- [ ] Scene setup helpers
- [ ] Deferred shading test framework

---

## Key Insights

1. **Infrastructure is more complete than documented**
   - Visual regression exists → 10 files unblocked
   - Fragment shader testing exists → 7 files unblocked
   - 62% of deferred files can now be addressed!

2. **Low-hanging fruit has high impact**
   - Depth texture helper: 2-3 hours → 4 files
   - Multi-pass support: 2-3 days → 10 files
   - Both are straightforward additions

3. **Truly complex infrastructure has low ROI**
   - G-buffer generation: 1-2 weeks → 5-8 files
   - Most benefit comes from simple additions
   - Defer complex work until needed

4. **Documentation gap was the real blocker**
   - Files marked "needs visual validation" but tool exists
   - Files marked "no default" but defaults exist
   - Better documentation → immediate unlocking

---

## Next Steps

1. ✅ Document current capabilities (this file)
2. ✅ Update DEFERRED-files-updated.md with new findings
3. **Convert 17 immediately-ready files** (this week)
4. **Add depth texture helper** (next week, 2-3 hours)
5. **Add multi-pass rendering** (weeks 3-4, 2-3 days)
6. Review color/ and space/ categories for more hidden convertibles

---

**See also:**
- `DEFERRED-files-updated.md` - Updated deferred file list
- `deferred-review-rubric.md` - Decision criteria for deferrals
- `deferred-review-results.md` - Detailed review findings
- `test/wesl/testUtil.ts` - Current test infrastructure implementation
- `test/wesl-examples/` - Visual regression test examples
