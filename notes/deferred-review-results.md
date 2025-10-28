# DEFERRED Files Review Results

**Date:** 2025-10-27
**Reviewers:** 3 specialized subagents
**Context:** Review deferred files in light of new texture/sampler test infrastructure

---

## Executive Summary

**Original Status:** 105 files deferred (24% of total)

**Review Results:**
- ✅ **Can un-defer immediately:** 35 files (33% of deferred)
- ⚠️ **Can un-defer with care:** 13 files (12% of deferred)
- ❌ **Keep deferred:** 57 files (54% of deferred)

**Net impact:** ~48 files can be converted (46% of originally deferred files)

---

## Findings by Category

### Filter/ Directory (23 files originally deferred)

**Review outcome:**
- ✅ **Un-defer immediately:** 17 files
- ⚠️ **Un-defer with review:** 6 files
- ❌ **Keep deferred:** 8 files (originally missing, now added)

**Critical blocker:** `sample/clamp2edge.glsl` must be converted FIRST - it's used by 10+ filter files

#### ✅ High Priority - Convert Immediately (17 files)

**Edge Detection (4 files):**
1. `filter/edge/sobel.glsl` - Simple 3x3 kernel
2. `filter/edge/prewitt.glsl` - Simple 3x3 kernel
3. `filter/edge/sobelDirectional.glsl` - Sobel + direction
4. `filter/edge.glsl` - Barrel file

**Box Blur (4 files):**
5. `filter/boxBlur/1D.glsl` - Separable 1D blur
6. `filter/boxBlur/2D.glsl` - Simple 2D blur
7. `filter/boxBlur/2D_fast9.glsl` - Fixed 9-tap
8. `filter/boxBlur.glsl` - Barrel file

**Laplacian (1 file):**
9. `filter/laplacian.glsl` - 3 simple variants

**Median (3 files):**
10. `filter/median/2D_fast3.glsl` - Fixed 3x3 median
11. `filter/median/2D_fast5.glsl` - Fixed 5x5 median
12. `filter/median.glsl` - Barrel file

**Sharpening (3 files):**
13. `filter/sharpen/fast.glsl` - Simple 5-tap kernel
14. `filter/sharpen/contrastAdaptive.glsl` - AMD CAS (no derivatives)
15. `filter/sharpen.glsl` - Barrel file

**Other (2 files):**
16. `filter/mean.glsl` - Simple averaging
17. `filter/bilinear.glsl` - Interpolation helper

#### ⚠️ Medium Priority - Review Before Converting (6 files)

18. `filter/gaussianBlur/1D_fast5.glsl` - Fixed kernel
19. `filter/gaussianBlur/1D_fast9.glsl` - Fixed kernel
20. `filter/gaussianBlur/1D_fast13.glsl` - Fixed kernel
21. `filter/gaussianBlur/1D.glsl` - Platform-specific loops
22. `filter/gaussianBlur/2D.glsl` - Platform-specific loops
23. `filter/gaussianBlur.glsl` - Barrel file

**Reason:** Gaussian blur has PLATFORM_WEBGL conditionals for loop bounds - needs careful handling

24. `filter/bilateral.glsl` - Edge-preserving blur
25. `filter/jointBilateral.glsl` - Needs dual-texture API

#### ❌ Keep Deferred (8 files)

26. `filter/kuwahara.glsl` - Complex variance analysis, needs visual testing
27. `filter/smartDeNoise.glsl` - Adaptive denoising, complex
28. `filter/noiseBlur.glsl` - Depends on sample/nearest.wesl (not converted)
29. `filter/radialBlur.glsl` - Artistic effect, needs visual validation
30. `filter/fibonacciBokeh.glsl` - Artistic effect, needs visual validation
31. `filter/sharpen/adaptive.glsl` - **Uses fwidth() derivatives** - requires fragment shader

**Note:** 2 files not in original deferred list but should be:
- `filter/sharpen/adaptive.glsl` - Uses derivatives
- Others already converted

---

### Sample/ Directory (27 files originally deferred)

**Review outcome:**
- ✅ **Un-defer immediately:** 18 files
- ⚠️ **Un-defer with review:** 7 files
- ❌ **Keep deferred:** 6 files (4 removed from deferred, 2 stay)

#### ✅ High Priority - Convert Immediately (18 files)

**Trivial (10 files):**
1. `sample/normalMap.glsl` - One-liner: sample & remap
2. `sample/repeat.glsl` - One-liner: fract(st)
3. `sample/mirror.glsl` - Uses math/mirror
4. `sample/clamp2edge.glsl` - **CRITICAL:** Convert FIRST (blocks filter/ files)
5. `sample/zero.glsl` - Simple boundary checks
6. `sample/hue.glsl` - Sample + rgb2hue
7. `sample/heatmap.glsl` - Sample + rgb2heat
8. `sample/nearest.glsl` - Uses space/nearest
9. `sample/yuv.glsl` - Dual-texture YUV
10. `sample/sprite.glsl` - Sprite sheet utility

**Medium Complexity (8 files):**
11. `sample/derivative.glsl` - 3 sampling modes
12. `sample/normalFromHeightMap.glsl` - Gradient calc
13. `sample/smooth.glsl` - Smooth filtering
14. `sample/flow.glsl` - Flow animation
15. `sample/opticalFlow.glsl` - Flow vector decoding
16. `sample/2DCube.glsl` - 3D LUT from 2D texture
17. `sample/triplanar.glsl` - Triplanar mapping
18. `sample/bracketing.glsl` - Directional mapping

#### ⚠️ Medium Priority - Review Before Converting (7 files)

19. `sample/bumpMap.glsl` - After derivative.glsl converts
20. `sample/dither.glsl` - After nearest.glsl converts
21. `sample/3DSdf.glsl` - After 2DCube.glsl converts
22. `sample/bicubic.glsl` - Needs visual regression tests
23. `sample/fxaa.glsl` - Anti-aliasing, needs visual tests
24. `sample/untile.glsl` - Complex, needs visual tests
25. `sample/quilt.glsl` - Looking Glass display-specific

#### ❌ Keep Deferred (6 files)

26. `sample/shadow.glsl` - Requires projection matrix, light coords
27. `sample/shadowLerp.glsl` - PCF shadow sampling, scene-dependent
28. `sample/shadowPCF.glsl` - 25 samples for soft shadows
29. `sample/viewPosition.glsl` - Requires camera matrices
30. `sample/equirect.glsl` - Complex, multiple modes, platform-specific
31. `sample/dof.glsl` - Depth of field, needs depth buffer + scene

**Removed from deferred (false positives in original list):**
- `sample/smooth.glsl` - Can convert now
- `sample/quilt.glsl` - Move to medium priority
- `sample/dither.glsl` - Move to medium priority
- Several others

---

### Lighting/material/ Directory (8 files originally deferred)

**Review outcome:**
- ❌ **Keep ALL deferred:** 8 files

**Reason:** These are **GlslViewer integration code**, not standalone utilities

#### ❌ All Deferred (8 files)

1. `lighting/material/albedo.glsl` - GlslViewer uniforms/varyings
2. `lighting/material/normal.glsl` - GlslViewer uniforms/varyings
3. `lighting/material/roughness.glsl` - GlslViewer uniforms/varyings
4. `lighting/material/metallic.glsl` - GlslViewer uniforms/varyings
5. `lighting/material/specular.glsl` - GlslViewer uniforms/varyings
6. `lighting/material/emissive.glsl` - GlslViewer uniforms/varyings
7. `lighting/material/occlusion.glsl` - GlslViewer uniforms/varyings
8. `lighting/material/new.glsl` - Orchestrates all material files

**Why keep deferred:**
- No function parameters - read from global uniforms
- Expect GlslViewer-specific: `MATERIAL_BASECOLORMAP`, `v_texcoord`, etc.
- Cannot test without full GlslViewer material system
- Would require complete API redesign to convert

**Recommendation:** These belong in GlslViewer's codebase, not LYGIA's general-purpose utilities

---

### Other Categories

**Not reviewed in detail (assumed to stay deferred):**
- Distort/ (6 files) - All use SAMPLER_FNC but are distortion effects
- Morphological/ (8 files) - All use SAMPLER_FNC, mostly iterative algorithms
- Simulate/ (4 files) - Multi-pass simulations
- Generative/ (3 files) - User-pluggable FBM/Voronoi variants
- Color/ (5 files) - Mixed complexity
- Draw/ (2 files) - Special cases
- Space/ (3 files) - Special spatial transforms
- SDF/ (1 file) - Uses CIRCLESDF_FNC macro
- Root/ (1 file) - Defines SAMPLER_FNC itself

**Recommendation:** Focus on filter/ and sample/ first, then reassess these categories

---

## Conversion Dependencies

### Critical Path

**MUST convert first:**
1. `sample/clamp2edge.glsl` - Blocks 10+ filter files

**Should convert early:**
2. `sample/nearest.glsl` - Blocks sample/dither, filter/noiseBlur
3. `sample/derivative.glsl` - Blocks sample/bumpMap

### Dependency Tree

```
sample/clamp2edge.wesl
├── filter/edge/sobel.wesl
├── filter/edge/prewitt.wesl
├── filter/edge/sobelDirectional.wesl
├── filter/bilateral.wesl
├── filter/gaussianBlur/*.wesl
└── sample/bilinear.wesl

sample/nearest.wesl
├── sample/dither.wesl
└── filter/noiseBlur.wesl

sample/derivative.wesl
└── sample/bumpMap.wesl

sample/2DCube.wesl
└── sample/3DSdf.wesl
```

---

## Recommended Conversion Order

### Phase 1: Foundation (1 file) - CRITICAL
1. `sample/clamp2edge.glsl` - Unblocks filter/ category

### Phase 2: Simple Sample Utilities (10 files) - HIGH VALUE
2-11. Trivial sample/ files (normalMap, repeat, mirror, etc.)

### Phase 3: Simple Filters (14 files) - HIGH VALUE
12-25. Edge detection, box blur, laplacian, median, sharpen

### Phase 4: Medium Complexity (8 files)
26-33. Medium sample/ files (derivative, triplanar, 2DCube, etc.)

### Phase 5: Advanced (13 files)
34-46. Gaussian blur, bilateral, bicubic, fxaa, etc.

**Total immediate conversion candidates: 48 files**

---

## Testing Strategy

### Edge Detection
- Test with `gradientTexture()` - expect strong edges on boundaries
- Validate edge magnitude values
- Visual regression for directional output

### Blur Filters
- Test with `checkerboardTexture()` - measure variance reduction
- Gaussian: compare with theoretical distribution
- Bilateral: verify edge preservation

### Median Filters
- Add salt-and-pepper noise to texture
- Validate noise removal while preserving edges

### Sampling Utilities
- Numeric validation for simple functions (repeat, mirror, clamp)
- Visual regression for complex ones (bicubic, fxaa, untile)

---

## Summary Statistics

### Before Review
- Total deferred: 105 files
- Deferred by category: filter (23), sample (27), lighting/material (8), others (47)

### After Review
- **Can convert immediately:** 35 files (filter: 17, sample: 18)
- **Can convert with review:** 13 files (filter: 6, sample: 7)
- **Keep deferred:** 57 files (filter: 8, sample: 6, lighting/material: 8, others: 35)

### Impact
- **Reducti on in deferred files:** 48 files (46% of originally deferred)
- **New deferred total:** 57 files (13% of total files)
- **Conversion ready:** 75% of filter/ files, 81% of sample/ files

---

## Next Steps

1. **Update DEFERRED-files.md** with new classifications
2. **Create conversion issues** for high-priority files
3. **Convert sample/clamp2edge.glsl FIRST** - critical blocker
4. **Batch convert** simple utilities (10-15 files at a time)
5. **Add visual regression tests** for complex filters
6. **Document** GlslViewer integration pattern for material files

---

## Key Takeaways

1. **Texture testing infrastructure enables most conversions** - 48 files can now be converted
2. **sample/clamp2edge.glsl is critical** - blocks 10+ filter conversions
3. **GlslViewer material files are special** - not general-purpose utilities, keep deferred
4. **Platform conditionals can be removed** - WGSL supports dynamic loops
5. **Visual regression testing needed** - for artistic effects (kuwahara, bokeh, radial blur)
6. **Derivative-based filters still need fragment context** - only sharpen/adaptive affected
