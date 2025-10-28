# Files Deferred from GLSL to WESL Conversion

**Last Updated:** 2025-10-27
**Status:** REVISED after texture/sampler test infrastructure review

---

## Summary

**Original Status (pre-review):**
- Total deferred: 105 files (24% of 432 total)

**Updated Status (post-review):**
- **Can convert NOW:** 48 files (moved from deferred to ready)
- **Still deferred:** 57 files (13% of total)

**Net change:** -48 deferred files (46% reduction in deferred count)

---

## Files NO LONGER DEFERRED ✅

These 48 files were previously deferred but can now be converted with the new test infrastructure.

### Filter/ - Can Convert (23 files)

#### High Priority - Convert Immediately (17 files)

**Edge Detection:**
1. `filter/edge/sobel.glsl` - 3x3 Sobel edge detection
2. `filter/edge/prewitt.glsl` - 3x3 Prewitt edge detection
3. `filter/edge/sobelDirectional.glsl` - Sobel with direction output
4. `filter/edge.glsl` - Barrel file for edge detection

**Box Blur:**
5. `filter/boxBlur/1D.glsl` - Separable 1D box blur
6. `filter/boxBlur/2D.glsl` - 2D box blur
7. `filter/boxBlur/2D_fast9.glsl` - Fixed 9-tap box blur
8. `filter/boxBlur.glsl` - Barrel file for box blur

**Laplacian:**
9. `filter/laplacian.glsl` - Laplacian edge-preserving highpass

**Median:**
10. `filter/median/2D_fast3.glsl` - Fixed 3x3 median filter
11. `filter/median/2D_fast5.glsl` - Fixed 5x5 median filter
12. `filter/median.glsl` - Barrel file for median filters

**Sharpening:**
13. `filter/sharpen/fast.glsl` - Simple 5-tap sharpen
14. `filter/sharpen/contrastAdaptive.glsl` - AMD CAS algorithm
15. `filter/sharpen.glsl` - Barrel file (excludes adaptive)

**Other:**
16. `filter/mean.glsl` - Mean/average filter
17. `filter/bilinear.glsl` - Bilinear interpolation

#### Medium Priority - Review Before Converting (6 files)

**Gaussian Blur:**
18. `filter/gaussianBlur/1D_fast5.glsl` - Fixed 5-tap gaussian
19. `filter/gaussianBlur/1D_fast9.glsl` - Fixed 9-tap gaussian
20. `filter/gaussianBlur/1D_fast13.glsl` - Fixed 13-tap gaussian
21. `filter/gaussianBlur/1D.glsl` - Separable 1D gaussian
22. `filter/gaussianBlur/2D.glsl` - 2D gaussian blur
23. `filter/gaussianBlur.glsl` - Barrel file for gaussian blur

**Note:** Gaussian blur files have PLATFORM_WEBGL conditionals - need careful loop handling

### Sample/ - Can Convert (25 files)

#### High Priority - Convert Immediately (18 files)

**Critical (convert FIRST):**
1. `sample/clamp2edge.glsl` - **BLOCKS filter/ conversions!**

**Trivial:**
2. `sample/normalMap.glsl` - Normal map sampling
3. `sample/repeat.glsl` - Repeat wrapping mode
4. `sample/mirror.glsl` - Mirror wrapping mode
5. `sample/zero.glsl` - Zero boundary handling
6. `sample/hue.glsl` - Hue from texture
7. `sample/heatmap.glsl` - Heatmap color extraction
8. `sample/nearest.glsl` - Nearest-neighbor sampling
9. `sample/yuv.glsl` - Dual-texture YUV sampling
10. `sample/sprite.glsl` - Sprite sheet utility

**Medium Complexity:**
11. `sample/derivative.glsl` - 3 derivative sampling modes
12. `sample/normalFromHeightMap.glsl` - Normal from heightmap gradient
13. `sample/smooth.glsl` - Smooth interpolation filter
14. `sample/flow.glsl` - Flow animation sampling
15. `sample/opticalFlow.glsl` - Optical flow vector decoding
16. `sample/2DCube.glsl` - 3D LUT from 2D texture
17. `sample/triplanar.glsl` - Triplanar texture mapping
18. `sample/bracketing.glsl` - Directional texture mapping

#### Medium Priority - Review Before Converting (7 files)

19. `sample/bumpMap.glsl` - Bump mapping (depends on derivative.glsl)
20. `sample/dither.glsl` - Dithering (depends on nearest.glsl)
21. `sample/3DSdf.glsl` - 3D SDF from texture (depends on 2DCube.glsl)
22. `sample/bicubic.glsl` - Bicubic interpolation (needs visual tests)
23. `sample/fxaa.glsl` - Fast anti-aliasing (needs visual tests)
24. `sample/untile.glsl` - Texture untiling (needs visual tests)
25. `sample/quilt.glsl` - Looking Glass display sampling

---

## Files STILL DEFERRED ❌

These 57 files remain deferred for various reasons.

### Filter/ - Still Deferred (6 files)

1. `filter/kuwahara.glsl` - Complex variance analysis, needs visual validation
2. `filter/smartDeNoise.glsl` - Adaptive denoising, complex algorithm
3. `filter/noiseBlur.glsl` - Depends on sample/nearest (not yet converted)
4. `filter/radialBlur.glsl` - Artistic effect, needs visual validation
5. `filter/fibonacciBokeh.glsl` - Bokeh effect, needs visual validation
6. `filter/sharpen/adaptive.glsl` - **Uses fwidth() derivatives**

**Reasons:**
- Complex iterative algorithms (kuwahara, smartDeNoise)
- Artistic effects requiring visual regression testing
- Derivative-based (adaptive sharpen)
- Dependencies (noiseBlur → sample/nearest)

### Sample/ - Still Deferred (6 files)

1. `sample/shadow.glsl` - Requires projection matrix, light coordinates
2. `sample/shadowLerp.glsl` - PCF shadow sampling, scene-dependent
3. `sample/shadowPCF.glsl` - 25-sample soft shadows, scene-dependent
4. `sample/viewPosition.glsl` - Requires camera matrices
5. `sample/equirect.glsl` - Complex equirectangular projection, platform-specific
6. `sample/dof.glsl` - Depth of field, requires depth buffer + scene setup

**Reasons:**
- Scene dependencies (shadows, viewPosition, dof)
- Camera/projection matrix requirements
- Complex multi-pass algorithms
- Platform-specific code

### Lighting/Material/ - Still Deferred (8 files)

1. `lighting/material/albedo.glsl` - GlslViewer integration
2. `lighting/material/normal.glsl` - GlslViewer integration
3. `lighting/material/roughness.glsl` - GlslViewer integration
4. `lighting/material/metallic.glsl` - GlslViewer integration
5. `lighting/material/specular.glsl` - GlslViewer integration
6. `lighting/material/emissive.glsl` - GlslViewer integration
7. `lighting/material/occlusion.glsl` - GlslViewer integration
8. `lighting/material/new.glsl` - Material system orchestrator

**Reasons:**
- **GlslViewer-specific integration code**, not general-purpose utilities
- Rely on global uniforms: `MATERIAL_BASECOLORMAP`, `v_texcoord`, etc.
- No function parameters - read from renderer state
- Cannot test without full GlslViewer material system
- Would require complete API redesign

**Recommendation:** These belong in GlslViewer's codebase, not LYGIA

### Lighting/ - Still Deferred (14 files)

**Main directory (6 files):**
1. `lighting/ssao.glsl` - Screen-space ambient occlusion
2. `lighting/volumetricLightScattering.glsl` - Volumetric scattering
3. `lighting/sphereMap.glsl` - Sphere mapping
4. `lighting/shadow.glsl` - Shadow mapping
5. `lighting/ssr.glsl` - Screen-space reflections

**Other (uncounted, review needed):**
- Various PBR and advanced lighting effects

**Reasons:**
- Complex scene dependencies
- Multi-pass rendering requirements
- Requires depth buffers, normal buffers, etc.

### Distort/ - Still Deferred (6 files)

1. `distort/displace.glsl` - Displacement mapping
2. `distort/barrel.glsl` - Barrel distortion
3. `distort/chromaAB.glsl` - Chromatic aberration
4. `distort/stretch.glsl` - Stretch distortion
5. `distort/pincushion.glsl` - Pincushion distortion
6. `distort/grain.glsl` - Film grain effect

**Reasons:**
- Artistic effects requiring visual validation
- Could convert but defer until higher priorities done

### Morphological/ - Still Deferred (8 files)

1. `morphological/erosion.glsl` - Morphological erosion
2. `morphological/alphaFill.glsl` - Alpha channel fill
3. `morphological/marchingSquares.glsl` - Marching squares
4. `morphological/jumpFlood.glsl` - Jump flood algorithm
5. `morphological/alphaHashing.glsl` - Alpha hashing
6. `morphological/dilation.glsl` - Morphological dilation
7. `morphological/pyramid.glsl` - Pyramid operations
8. `morphological/pyramid/upscale.glsl`, `downscale.glsl` - Pyramid utilities

**Reasons:**
- Iterative multi-pass algorithms
- Complex computational patterns
- Lower priority for general use

### Simulate/ - Still Deferred (4 files)

1. `simulate/latticeBoltzmann.glsl` - Physics simulation
2. `simulate/simpleAndFastFluid.glsl` - Fluid simulation
3. `simulate/ripple.glsl` - Water ripple simulation
4. `simulate/grayscott.glsl` - Reaction-diffusion simulation

**Reasons:**
- Multi-pass simulation algorithms
- Require persistent state across frames
- Complex computational patterns

### Generative/ - Still Deferred (3 files)

1. `generative/fbm.glsl` - FBM with user-defined FBM_FNC
2. `generative/voronoise.glsl` - Voronoi noise with VORONOISE_FNC
3. `generative/voronoi.glsl` - Voronoi with VORONOI_FNC

**Reasons:**
- Require user-pluggable function macros
- No sensible default implementation
- API redesign needed

### Color/ - Still Deferred (5 files)

1. `color/dither.glsl` - Uses DITHER_FNC macro
2. `color/lut.glsl` - Uses SAMPLER_FNC for LUT
3. `color/daltonize.glsl` - Uses DALTONIZE_FNC macro
4. `color/palette/lerp.glsl` - Uses PALETTE_LERP_MIX_FNC macro

**Reasons:**
- User-pluggable function macros
- Some may be convertible with default implementations

### Space/ - Still Deferred (3 files)

1. `space/displace.glsl` - Displacement with SAMPLER_FNC
2. `space/bracketing.glsl` - Bracketing with SAMPLER_FNC
3. `space/parallaxMapping.glsl` - Parallax mapping

**Reasons:**
- Could review for conversion
- Lower priority

### Draw/ - Still Deferred (2 files)

1. `draw/colorPicker.glsl` - Uses SAMPLER_FNC
2. `draw/colorChecker.glsl` - Uses COLORCHECKER_FNC

**Reasons:**
- Specialized UI utilities
- Lower priority

### SDF/ - Still Deferred (1 file)

1. `sdf/circleSDF.glsl` - Uses CIRCLESDF_FNC macro

**Reasons:**
- User-pluggable SDF function
- API redesign needed

### Root/ - Still Deferred (1 file)

1. `sampler.glsl` - Defines SAMPLER_FNC and SAMPLER_TYPE

**Reasons:**
- Core macro definition file
- Not applicable to WESL

---

## Conversion Priority Order

### Phase 1: Critical Foundation (1 file)
**MUST convert FIRST:**
1. `sample/clamp2edge.glsl` - Unblocks 10+ filter files

### Phase 2: Simple Sample Utilities (10 files)
**High value, low risk:**
- normalMap, repeat, mirror, zero, hue, heatmap, nearest, yuv, sprite

### Phase 3: Simple Filters (14 files)
**After clamp2edge converts:**
- Edge detection (4 files)
- Box blur (4 files)
- Laplacian (1 file)
- Median (3 files)
- Sharpen (2 files: fast, contrastAdaptive)
- Mean, bilinear

### Phase 4: Medium Sample Utilities (8 files)
- derivative, normalFromHeightMap, smooth, flow, opticalFlow, 2DCube, triplanar, bracketing

### Phase 5: Gaussian Blur & Advanced (13 files)
- Gaussian blur variants (6 files)
- Bilateral (1 file)
- Sample advanced (bicubic, fxaa, untile, etc.) (6 files)

**Total immediate path: 48 files in 5 phases**

---

## Testing Strategy

### Required Test Infrastructure (Already Available ✅)
- `testFragment()` - Fragment shader testing
- `inputTextures` - Texture binding for tests
- Texture helpers: `gradientTexture`, `checkerboardTexture`, `radialGradientTexture`, etc.
- `toMatchImage()` - Visual regression testing

### Test Patterns by File Type

**Edge Detection:**
```typescript
const result = await testFragment(edgeSrc, {
  inputTextures: [{ texture: gradientTexture(), sampler }],
  size: [256, 256],
});
// Expect strong edges on gradient boundaries
```

**Blur Filters:**
```typescript
const result = await testFragment(blurSrc, {
  inputTextures: [{ texture: checkerboardTexture(), sampler }],
});
// Measure variance reduction
```

**Visual Regression (for artistic effects):**
```typescript
await expect(result).toMatchImage("filter-kuwahara-oil-painting");
```

---

## Dependencies

### Critical Blockers

**sample/clamp2edge.glsl must convert FIRST**
- Blocks: edge/sobel, edge/prewitt, bilateral, gaussianBlur/*, bilinear

**sample/nearest.glsl needed for:**
- sample/dither
- filter/noiseBlur

**sample/derivative.glsl needed for:**
- sample/bumpMap

**sample/2DCube.glsl needed for:**
- sample/3DSdf

### Conversion Dependency Graph

```
sample/clamp2edge.wesl (CRITICAL - convert first)
├── filter/edge/*.wesl (4 files)
├── filter/bilateral.wesl
├── filter/gaussianBlur/*.wesl (6 files)
└── filter/bilinear.wesl

sample/nearest.wesl
├── sample/dither.wesl
└── filter/noiseBlur.wesl

sample/derivative.wesl
└── sample/bumpMap.wesl

sample/2DCube.wesl
└── sample/3DSdf.wesl
```

---

## Statistics

### Deferred Count Changes

| Category | Original Deferred | Now Ready | Still Deferred | Change |
|----------|-------------------|-----------|----------------|--------|
| Filter | 23 | 23 | 6 | -17 (74% reduction) |
| Sample | 27 | 25 | 6 | -21 (78% reduction) |
| Lighting/material | 8 | 0 | 8 | 0 (keep all deferred) |
| Lighting (other) | ~14 | 0 | ~14 | 0 (review needed) |
| Distort | 6 | 0 | 6 | 0 (could review) |
| Morphological | 8 | 0 | 8 | 0 (complex) |
| Simulate | 4 | 0 | 4 | 0 (complex) |
| Generative | 3 | 0 | 3 | 0 (API redesign) |
| Color | 5 | 0 | 5 | 0 (could review) |
| Space | 3 | 0 | 3 | 0 (could review) |
| Draw | 2 | 0 | 2 | 0 (low priority) |
| SDF | 1 | 0 | 1 | 0 (API redesign) |
| Root | 1 | 0 | 1 | 0 (not applicable) |
| **TOTAL** | **105** | **48** | **57** | **-48 (46% reduction)** |

### Impact on Overall Conversion

| Metric | Before Review | After Review | Change |
|--------|---------------|--------------|--------|
| Total GLSL files | 432 | 432 | - |
| Deferred files | 105 (24%) | 57 (13%) | -48 |
| Ready to convert | 327 (76%) | 375 (87%) | +48 |

---

## Key Findings

1. **Texture testing infrastructure enables most conversions** - 46% of deferred files can now be converted
2. **sample/clamp2edge.glsl is critical** - Must convert first to unblock filter/ category
3. **GlslViewer material files are special cases** - Not general-purpose utilities, belong in renderer codebase
4. **Platform conditionals can be removed** - WGSL supports dynamic loops
5. **Visual regression needed for artistic effects** - kuwahara, bokeh, radial blur
6. **Only one filter uses derivatives** - sharpen/adaptive (uses fwidth)
7. **Scene dependencies still block many files** - shadows, SSR, SSAO, etc.

---

## Next Actions

1. ✅ Update this file (DEFERRED-files.md)
2. Convert `sample/clamp2edge.glsl` FIRST (critical blocker)
3. Batch convert simple sample utilities (10 files)
4. Batch convert simple filters (14 files)
5. Add visual regression tests for complex filters
6. Review remaining deferred categories (distort, color, space, draw)
7. Document GlslViewer integration pattern for material files

---

**For detailed review methodology and findings, see:**
- `notes/deferred-review-rubric.md` - Decision criteria
- `notes/deferred-review-results.md` - Detailed agent findings
