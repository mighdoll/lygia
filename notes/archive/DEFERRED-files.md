# Files Deferred from GLSL to WESL Conversion

This document lists all GLSL files that use texture sampling (SAMPLER_FNC, SAMPLER_TYPE) or macro-based function composition and should be DEFERRED for later conversion.

**Total Deferred: 105 files** (24.3% of 432 total files)

---

## Deferred Files by Category

### Root Level (1 file)
- `/Users/lee/wesl/lygia/sampler.glsl` - DEFER (defines SAMPLER_FNC and SAMPLER_TYPE)

### SDF (1 file)
- `/Users/lee/wesl/lygia/sdf/circleSDF.glsl` - DEFER (uses CIRCLESDF_FNC macro)

### Color (5 files)

#### Color (main directory) - 4 files
- `/Users/lee/wesl/lygia/color/dither.glsl` - DEFER (uses DITHER_FNC)
- `/Users/lee/wesl/lygia/color/lut.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/color/daltonize.glsl` - DEFER (uses DALTONIZE_FNC)

#### Color/Palette - 1 file
- `/Users/lee/wesl/lygia/color/palette/lerp.glsl` - DEFER (uses PALETTE_LERP_MIX_FNC)

### Simulate (4 files) - ALL DEFERRED
- `/Users/lee/wesl/lygia/simulate/latticeBoltzmann.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/simulate/simpleAndFastFluid.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/simulate/ripple.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/simulate/grayscott.glsl` - DEFER (uses SAMPLER_FNC)

### Generative (3 files)
- `/Users/lee/wesl/lygia/generative/fbm.glsl` - DEFER (uses FBM_FNC)
- `/Users/lee/wesl/lygia/generative/voronoise.glsl` - DEFER (uses VORONOISE_FNC)
- `/Users/lee/wesl/lygia/generative/voronoi.glsl` - DEFER (uses VORONOI_FNC)

### Sample (27 files) - ALMOST ALL DEFERRED
- `/Users/lee/wesl/lygia/sample/smooth.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/quilt.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/2DCube.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/dof.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/yuv.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/bumpMap.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/clamp2edge.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/heatmap.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/untile.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/dither.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/nearest.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/shadowLerp.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/normalMap.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/bracketing.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/triplanar.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/zero.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/fxaa.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/equirect.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/flow.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/repeat.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/shadowPCF.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/bicubic.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/hue.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/3DSdf.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/shadow.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/mirror.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/normalFromHeightMap.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/derivative.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/viewPosition.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/sample/opticalFlow.glsl` - DEFER (uses SAMPLER_FNC)

### Lighting (22 files)

#### Lighting (main directory) - 6 files
- `/Users/lee/wesl/lygia/lighting/ssao.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/volumetricLightScattering.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/sphereMap.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/shadow.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/ssr.glsl` - DEFER (uses SAMPLER_FNC)

#### Lighting/Material - 8 files
- `/Users/lee/wesl/lygia/lighting/material/new.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/albedo.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/metallic.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/normal.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/roughness.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/specular.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/emissive.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/lighting/material/occlusion.glsl` - DEFER (uses SAMPLER_FNC)

### Filter (23 files) - ALL DEFERRED

#### Filter (main directory) - 12 files
- `/Users/lee/wesl/lygia/filter/edge.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/median.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/sharpen.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/bilateral.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/fibonacciBokeh.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/bilinear.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/gaussianBlur.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/radialBlur.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/kuwahara.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/smartDeNoise.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/boxBlur.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/noiseBlur.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/jointBilateral.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/mean.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/laplacian.glsl` - DEFER (uses SAMPLER_FNC)

#### Filter/BoxBlur - 3 files
- `/Users/lee/wesl/lygia/filter/boxBlur/1D.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/boxBlur/2D_fast9.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/boxBlur/2D.glsl` - DEFER (uses SAMPLER_FNC)

#### Filter/Sharpen - 1 file
- `/Users/lee/wesl/lygia/filter/sharpen/contrastAdaptive.glsl` - DEFER (uses SAMPLER_FNC)

#### Filter/Median - 2 files
- `/Users/lee/wesl/lygia/filter/median/2D_fast3.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/median/2D_fast5.glsl` - DEFER (uses SAMPLER_FNC)

#### Filter/GaussianBlur - 5 files
- `/Users/lee/wesl/lygia/filter/gaussianBlur/1D_fast13.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/gaussianBlur/1D_fast9.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/gaussianBlur/1D_fast5.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/gaussianBlur/1D.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/gaussianBlur/2D.glsl` - DEFER (uses SAMPLER_FNC)

#### Filter/Edge - 2 files
- `/Users/lee/wesl/lygia/filter/edge/sobel.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/filter/edge/sobelDirectional.glsl` - DEFER (uses SAMPLER_FNC)

### Space (3 files)
- `/Users/lee/wesl/lygia/space/displace.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/space/bracketing.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/space/parallaxMapping.glsl` - DEFER (uses SAMPLER_FNC)

### Distort (6 files) - ALL DEFERRED
- `/Users/lee/wesl/lygia/distort/displace.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/distort/barrel.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/distort/chromaAB.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/distort/stretch.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/distort/pincushion.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/distort/grain.glsl` - DEFER (uses SAMPLER_FNC)

### Draw (2 files)
- `/Users/lee/wesl/lygia/draw/colorPicker.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/draw/colorChecker.glsl` - DEFER (uses COLORCHECKER_FNC)

### Morphological (8 files) - ALL DEFERRED

#### Morphological (main directory) - 6 files
- `/Users/lee/wesl/lygia/morphological/erosion.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/morphological/alphaFill.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/morphological/marchingSquares.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/morphological/jumpFlood.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/morphological/alphaHashing.glsl` - DEFER (uses ALPHAHASHING_FNC)
- `/Users/lee/wesl/lygia/morphological/dilation.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/morphological/pyramid.glsl` - DEFER (uses SAMPLER_FNC)

#### Morphological/Pyramid - 2 files
- `/Users/lee/wesl/lygia/morphological/pyramid/upscale.glsl` - DEFER (uses SAMPLER_FNC)
- `/Users/lee/wesl/lygia/morphological/pyramid/downscale.glsl` - DEFER (uses SAMPLER_FNC)

---

## Summary Statistics

| Category | Deferred Files | Total Files | Deferred % | Status |
|----------|----------------|-------------|------------|--------|
| Root | 1 | 1 | 100% | All deferred |
| SDF | 1 | 40 | 2.5% | 1 deferred, 39 convertible |
| Color | 5 | 110 | 4.5% | 5 deferred, 105 convertible |
| Simulate | 4 | 4 | 100% | All deferred |
| Generative | 3 | 7 | 42.9% | 3 deferred, 4 convertible |
| Sample | 27 | 30 | 90% | 27 deferred, 3 convertible |
| Lighting | 22 | 74 | 29.7% | 22 deferred, 52 convertible |
| Filter | 23 | 23 | 100% | All deferred |
| Space | 3 | 31 | 9.7% | 3 deferred, 28 convertible |
| Distort | 6 | 6 | 100% | All deferred |
| Draw | 2 | 16 | 12.5% | 2 deferred, 14 convertible |
| Morphological | 8 | 8 | 100% | All deferred |
| **TOTAL** | **105** | **432** | **24.3%** | **327 files ready to convert** |

---

## Files Ready for Conversion

**327 files** (75.7% of total) are ready for conversion to WESL. These include:
- Most SDF functions (39 of 40)
- Most color functions (105 of 110)
- Most generative functions (4 of 7)
- Most lighting functions (52 of 74)
- Most space functions (28 of 31)
- Most draw functions (14 of 16)
- All math functions (23 files)
- All animation/easing functions (12 files)
- All geometry functions (19 files)

These files can be converted following the patterns described in [GLSLtoWESL.md](GLSLtoWESL.md).
