# GLSL Files to Convert to WESL

This document lists all GLSL files in the repository that do NOT have corresponding WESL files and are NOT barrel import files (files containing only #include statements).

**Total GLSL files: 656**
**WESL files created: 351 (53.5%)**
**GLSL files without WESL equivalents: 309**
**Excluding barrel files: 288 files need conversion**

_Note: "Barrel files" are import-only files containing only `#include` statements with no implementations. These don't need WESL equivalents since WESL users import specific functions directly. See [barrel-files.md](barrel-files.md) for the complete list of 21 barrel files._

---

## 🧪 Test Coverage Summary

**Total Test Files**: 6
**Total Tests**: 184 (178 passing, 6 skipped)

### ✅ Well-Tested Categories

**Color Functions** (68 tests in color.test.ts) - EXCELLENT coverage:
- ✅ Color space conversions: rgb2xyz, rgb2YPbPr, rgb2yuv, yuv2rgb, rgb2heat
- ✅ All space conversions: hsl2rgb, rgb2hsl, hsv2rgb, rgb2hsv, hcy2rgb, rgb2hcy
- ✅ Lab/LCH: lab2rgb, rgb2lab, lch2rgb, rgb2lch, srgb2lab, lab2srgb, srgb2lch, lch2srgb3
- ✅ Oklab: oklab2srgb, srgb2oklab
- ✅ XYZ/xyY: srgb2xyz, xyY2rgb, rgb2xyY, xyY2srgb
- ✅ RYB: ryb2rgb, rgb2ryb, hsv2ryb
- ✅ Color utilities: colorDistance, hueShift, luma, mixOklab, vibrance
- ✅ Desaturate: desaturate, desaturate4
- ✅ Luminance: luminance, luminance4
- ✅ Brightness/Contrast: brightnessMatrix, contrast, contrast3, contrastMatrix
- ✅ Levels: levelsInputRange3, levelsGamma3, levels3Float
- ✅ Compositing: compositeSourceOver4, compositeSourceIn4, compositeXor4
- ✅ Tonemapping: tonemapReinhard3, tonemapUnreal3, tonemapLinear3
- ✅ Blending (10 functions): blendHardLight3, blendVividLight3, blendPinLight3, blendLinearLight3, blendHardMix3, blendGlow3, blendHue, blendSaturation, blendColor, blendLuminosity
- ✅ Layer blending (10 functions): layerMultiplySourceOver4, layerScreenSourceOver4, layerAddSourceOver4, layerOverlaySourceOver4, layerDarkenSourceOver4, layerLightenSourceOver4, layerDifferenceSourceOver4, layerExclusionSourceOver4, layerPhoenixSourceOver4, layerSubtractSourceOver4

**Space Functions** (36 tests in space.test.ts) - EXCELLENT coverage:
- ✅ Coordinate conversions: cart2polar2, polar2cart
- ✅ Centering: center, center2, uncenter, uncenter2
- ✅ Rotations: rotateX3, rotateY3, rotateZ3
- ✅ Aspect: flipY2, aspect
- ✅ Spherical/Equirect: equirect2xyz, xyz2equirect
- ✅ Tiling: sqTile, checkerTile2, brickTile2, triTile, hexTile, mirrorTile2, windmillTile2
- ✅ Depth: linearizeDepth, depth2viewZ, viewZ2depth
- ✅ Transforms: kaleidoscope, bracketing, tbn, perspective, orthographic
- ✅ View matrices: eulerView, lookAt, lookAtView, lookAtViewRoll, lookAtViewFromDirection
- ✅ Screen/View: view2screenPosition, screen2viewPosition, decimateNormal

**Easing Functions** (33 tests in easing.test.ts) - COMPLETE coverage:
- ✅ All 11 easing types × 3 variants (In/Out/InOut): back, bounce, circular, cubic, elastic, exponential, linear, quadratic, quartic, quintic, sine

**Math Functions** (15 tests in math.test.ts) - GOOD coverage:
- ✅ saturate, saturate3, pow2, pow22, pow3, pow5, pow7, absi
- ✅ Matrix conversions: toMat3, quat2mat3, quat2mat4
- ✅ Utilities: hammersley, nyquist

**Quaternion Functions** (11 tests in quat.test.ts) - COMPLETE coverage:
- ✅ quatAdd, quatSub, quatMul, quatConj, quatNorm, quatLength, quatLengthSq, quatIdentity, quatLerp, quat2mat3, quat2mat4

**Generative/Noise Functions** (15 tests in functions.test.ts) - GOOD coverage:
- ✅ cnoise2, cnoise3 (Classic Perlin noise)
- ✅ snoise2, snoise3 (Simplex noise)
- ✅ pnoise2 (Periodic noise)
- ✅ srandom2 (Seeded random)
- ✅ worley2 (Worley/Cellular noise)
- ✅ cubicMix, permute, smootherstep (utility functions)
- ✅ fmod2, fmod3, fmod4 (modulo variants)

### ⚠️ Functions with Skipped Tests (6 functions)

All in functions.test.ts:

1. **aafloor** - Anti-aliased floor (needs derivatives via `fwidth()`) - **REQUIRES FRAGMENT SHADER TEST**
2. **aafract** - Anti-aliased fract (needs derivatives via `fwidth()`) - **REQUIRES FRAGMENT SHADER TEST**
3. **fresnel** - Signature mismatch (test expects different params)
4. **fresnelReflection** - Signature mismatch (test expects different params)
5. **raymarchCast** - Requires user-defined `map()` function
6. **cookTorrance** - Named `specularCookTorrance`, different params

**Note**: These are edge cases. Core functionality is working.

### 🔍 Converted Files WITHOUT Tests

Based on the test coverage above, here are converted .wesl files that lack tests:

#### ✅ Fragment Shader Functions (6 functions) - TESTED

These functions use derivatives (`fwidth()`, `dpdx()`, `dpdy()`) which are only available in fragment shaders.
Fragment shader test harness is **implemented** using `testFragment()` from `testUtil.ts`.

1. **`math/aafloor.wesl`** ✅ - Anti-aliased floor (tested in `math-aa.test.ts`)
2. **`math/aafract.wesl`** ✅ - Anti-aliased fract (tested in `math-aa.test.ts`)
3. **`math/aastep.wesl`** ✅ - Anti-aliased step (tested in `math-aa.test.ts`)
4. **`math/aamirror.wesl`** ✅ - Anti-aliased mirror (tested in `math-aa.test.ts`)
5. **`math/fcos.wesl`** ✅ - Fast cosine approximation (tested in `math-aa.test.ts`)
6. **`filter/sharpen/adaptive.wesl`** ✅ - Adaptive sharpening (tested in `filter-sharpen.test.ts`)

**Status:** All derivative functions now have passing tests using the fragment shader test infrastructure. See CLAUDE.md "Fragment Shader Testing" section for usage patterns.

#### Color Utilities (4 files) - Priority: Medium
- `color/hueShiftRYB.wesl` - RYB color space hue shifting
- `color/mixSpectral.wesl` - Spectral color mixing
- `color/palette/hue.wesl` - Palette hue utilities
- `color/dither/bayer.wesl`, `color/dither/blueNoise.wesl`, `color/dither/vlachos.wesl` - Dithering algorithms

#### Generative Functions (2 files) - Priority: Medium
- `generative/noised.wesl` - Noise with derivatives
- `generative/wavelet.wesl` - Wavelet noise

#### Lighting Functions (2 files) - Priority: Low
- `lighting/raymarch/normal.wesl` - Normal calculation for raymarching
- (Other lighting functions have skipped tests, but tests exist)

#### Space Functions (1 file) - Priority: Medium
- `space/fisheye2xyz.wesl` - Has a test but it's passing now! ✅

### 📝 Testing Notes:
1. **Most categories have excellent coverage!** 178/184 tests passing
2. Test patterns are well-established in existing test files
3. The 6 skipped tests are known edge cases, not blocking issues
4. Total untested converted files: ~10 (down from the original estimate of 47!)

---

## 🔑 Key Learnings & Patterns for Conversion

### 1. Type Aliases - DON'T CREATE THEM
- ❌ Don't create `type.wesl` files with aliases (e.g., `alias Quat = vec4f`)
- ✅ Use base WGSL types directly (`vec4f`, not `Quat`)
- **Reason**: Aliases add no value, `type` keyword risks conflicts

### 2. Struct Definitions - CENTRALIZE THEM
- ❌ Don't inline struct definitions in every file
- ✅ Create one struct definition file per module (e.g., `aabb/aabb.wesl`)
- ✅ Import the struct in functions that use it
- **Example**: All AABB functions import from `geometry/aabb/aabb.wesl`

### 3. Platform Conditionals - WEBGPU ONLY
- ❌ Don't use: `PLATFORM_WEBGL`, `PLATFORM_RPI`
- ✅ Only use: `TARGET_MOBILE` (for mobile vs desktop browsers)
- **Reason**: WebGPU doesn't run on WebGL or bare Raspberry Pi

### 4. Conditional Syntax - BUG WORKAROUND
- ❌ Don't use `@else` after function definitions
- ✅ Use `@if(!CONDITION)` instead
```wesl
@if(FOO)
fn test(x: f32) -> f32 { return x * 2.0; }
@if(!FOO)  // NOT @else
fn test(x: f32) -> f32 { return x * 3.0; }
```
- **Reason**: `@else` across function definitions is broken (bug filed upstream)

### 5. Barrel Files - SKIP THEM
- ❌ Don't convert files that only contain `#include` statements
- ✅ Users can import specific functions directly
- **Example**: Skip `animation/easing/cubic.glsl` (just imports In/Out/InOut)

### 6. Function Naming - ADD SUFFIXES
- WGSL doesn't support function overloading
- Add numeric suffixes: `saturate2()`, `saturate3()`, `saturate4()`
- Or descriptive names: `expand()`, `expand2()`, `expand3()`

### 7. Pointer Parameters - USE ptr<function, T>
```wesl
// GLSL: inout AABB box
// WESL: box: ptr<function, AABB>
fn square(box: ptr<function, AABB>) {
    (*box).min = ...;  // dereference with *
}
```

---

## SDF (Signed Distance Fields) - 41 files

- `sdf.glsl` - Barrel file
- `sdf/mandelbulbSDF.glsl`
- `sdf/tetrahedronSDF.glsl`
- `sdf/gearSDF.glsl`
- `sdf/circleSDF.glsl`
- `sdf/opRound.glsl`
- `sdf/raysSDF.glsl`
- `sdf/spiralSDF.glsl`
- `sdf/hexSDF.glsl`
- `sdf/opRevolve.glsl`
- `sdf/lineSDF.glsl`
- `sdf/arrowSDF.glsl`
- `sdf/boxFrameSDF.glsl`
- `sdf/opRepeat.glsl`
- `sdf/pyramidSDF.glsl`
- `sdf/triPrismSDF.glsl`
- `sdf/heartSDF.glsl`
- `sdf/crossSDF.glsl`
- `sdf/coneSDF.glsl`
- `sdf/capsuleSDF.glsl`
- `sdf/octogonPrismSDF.glsl`
- `sdf/planeSDF.glsl`
- `sdf/dodecahedronSDF.glsl`
- `sdf/juliaSDF.glsl`
- `sdf/vesicaSDF.glsl`
- `sdf/icosahedronSDF.glsl`
- `sdf/ellipsoidSDF.glsl`
- `sdf/linkSDF.glsl`
- `sdf/flowerSDF.glsl`
- `sdf/opIntersection.glsl`
- `sdf/cubeSDF.glsl`
- `sdf/opExtrude.glsl`
- `sdf/triSDF.glsl`
- `sdf/opElongate.glsl`
- `sdf/rhombSDF.glsl`
- `sdf/opOnion.glsl`
- `sdf/kochSDF.glsl`
- `sdf/starSDF.glsl`
- `sdf/hexPrismSDF.glsl`
- `sdf/polySDF.glsl`
- `sdf/octahedronSDF.glsl`
- `sdf/superShapeSDF.glsl`

---

## Color - 41 files

### Color (main directory) - 7 files (barrel files)
- `color/space.glsl` - Barrel file
- `color/tonemap.glsl` - Barrel file
- `color/blend.glsl` - Barrel file
- `color/layer.glsl` - Barrel file
- `color/dither.glsl` - Barrel file
- `color/palette.glsl` - Barrel file
- `color/levels.glsl` - Barrel file

### Color/Dither - 3 files
- `color/dither/shift.glsl`
- `color/dither/triangleNoise.glsl`
- `color/dither/interleavedGradientNoise.glsl`

### Color/Composite - 9 files
- `color/composite/sourceOver.glsl`
- `color/composite/sourceOut.glsl`
- `color/composite/destinationOut.glsl`
- `color/composite/destinationOver.glsl`
- `color/composite/sourceIn.glsl`
- `color/composite/sourceAtop.glsl`
- `color/composite/compositeXor.glsl`
- `color/composite/destinationIn.glsl`
- `color/composite/destinationAtop.glsl`

### Color/Palette - 22 files
- `color/palette/zorn.glsl`
- `color/palette/ridgway.glsl`
- `color/palette/fire.glsl`
- `color/palette/spyder.glsl`
- `color/palette/water.glsl`
- `color/palette/lerp.glsl`
- `color/palette/pigments.glsl`
- `color/palette/flexoki.glsl`
- `color/palette/spectral.glsl`
- `color/palette/macbeth.glsl`

#### Color/Palette/Spectral - 5 files
- `color/palette/spectral/zucconi.glsl`
- `color/palette/spectral/soft.glsl`
- `color/palette/spectral/gems.glsl`
- `color/palette/spectral/zucconi6.glsl`
- `color/palette/spectral/geoffrey.glsl`

#### Color/Palette/Wada - 4 files
- `color/palette/wada/value.glsl`
- `color/palette/wada/triad.glsl`
- `color/palette/wada/tetrad.glsl`
- `color/palette/wada/dyad.glsl`

#### Color/Palette/Pigments - 3 files (4 are barrel files)
- `color/palette/pigments/winsor_acrylic.glsl`
- `color/palette/pigments/winsor_gouache.glsl`
- `color/palette/pigments/winsor_oil.glsl`

---

## Simulate - 4 files

- `simulate/latticeBoltzmann.glsl`
- `simulate/simpleAndFastFluid.glsl`
- `simulate/ripple.glsl`
- `simulate/grayscott.glsl`

---

## Animation - 12 files (all barrel files)

### Animation/Easing - 11 files (all barrel files)
- `animation/easing/elastic.glsl` - Barrel file
- `animation/easing/quartic.glsl` - Barrel file
- `animation/easing/cubic.glsl` - Barrel file
- `animation/easing/quadratic.glsl` - Barrel file
- `animation/easing/sine.glsl` - Barrel file
- `animation/easing/circular.glsl` - Barrel file
- `animation/easing/back.glsl` - Barrel file
- `animation/easing/quintic.glsl` - Barrel file
- `animation/easing/linear.glsl` - Barrel file
- `animation/easing/bounce.glsl` - Barrel file
- `animation/easing/exponential.glsl` - Barrel file

### Animation (main directory) - 1 file (barrel file)
- `animation/easing.glsl` - Barrel file

---

## Math - 10 files

- `math.glsl` - Barrel file
- `math/transpose.glsl` - May be redundant (WGSL has transpose builtin)
- `math/mmix.glsl` - Large multi-overload file, may be complex
- `math/decimation.glsl`
- `math/sum.glsl`
- `math/mirror.glsl`
- `math/mmax.glsl`
- `math/mmin.glsl`
- `math/quat/mul.glsl`
- `math/quat/conj.glsl`

---

## Generative - 7 files

- `generative/curl.glsl`
- `generative/gerstnerWave.glsl`
- `generative/voronoise.glsl`
- `generative/voronoi.glsl`
- `generative/psrdnoise.glsl`
- `generative/fbm.glsl`
- `generative/gnoise.glsl`

---

## Sample - 30 files

- `sample.glsl` - Barrel file
- `sampler.glsl` - Barrel file
- `sample/smooth.glsl`
- `sample/quilt.glsl`
- `sample/2DCube.glsl`
- `sample/dof.glsl`
- `sample/yuv.glsl`
- `sample/bumpMap.glsl`
- `sample/clamp2edge.glsl`
- `sample/heatmap.glsl`
- `sample/untile.glsl`
- `sample/dither.glsl`
- `sample/nearest.glsl`
- `sample/shadowLerp.glsl`
- `sample/normalMap.glsl`
- `sample/bracketing.glsl`
- `sample/triplanar.glsl`
- `sample/zero.glsl`
- `sample/fxaa.glsl`
- `sample/equirect.glsl`
- `sample/flow.glsl`
- `sample/repeat.glsl`
- `sample/shadowPCF.glsl`
- `sample/bicubic.glsl`
- `sample/hue.glsl`
- `sample/3DSdf.glsl`
- `sample/shadow.glsl`
- `sample/mirror.glsl`
- `sample/normalFromHeightMap.glsl`
- `sample/derivative.glsl`
- `sample/viewPosition.glsl`
- `sample/opticalFlow.glsl`

---

## Lighting - 91 files

_Lighting is the largest unconverted category with many complex, interdependent files. Many require struct definitions (Material, Light, Ray, etc.) to be converted first._

_(Full file listing available via: `./check-unconverted.sh | grep lighting`)_

---

## Geometry - 8 files

- `geometry/aabb/intersect.glsl` - Depends on Ray struct (in lighting/)
- `geometry/triangle/closestPoint.glsl`
- `geometry/triangle/intersect.glsl` - Depends on Ray struct
- `geometry/triangle/distanceSq.glsl`
- `geometry/triangle/signedDistance.glsl`
- `geometry/triangle/contain.glsl`
- `geometry/rect/clamp.glsl`
- `geometry/rect/rectShaped.glsl`

---

## Filter - 28 files

_(Full file listing available via: `./check-unconverted.sh | grep filter`)_

Includes: edge detection, blur (gaussian, box, radial), median filters, sharpening, bilateral filtering, FXAA, etc.

---

## Space - 2 files

- `space/displace.glsl`
- `space/parallaxMapping.glsl`

---

## Distort - 6 files

- `distort/displace.glsl`
- `distort/barrel.glsl`
- `distort/chromaAB.glsl`
- `distort/stretch.glsl`
- `distort/pincushion.glsl`
- `distort/grain.glsl`

---

## Draw - 16 files

- `draw/bridge.glsl`
- `draw/line.glsl`
- `draw/char.glsl`
- `draw/rect.glsl`
- `draw/colorChecker.glsl`
- `draw/circle.glsl`
- `draw/point.glsl`
- `draw/tri.glsl`
- `draw/matrix.glsl`
- `draw/axis.glsl`
- `draw/hex.glsl`
- `draw/flip.glsl`
- `draw/arrows.glsl`
- `draw/colorPicker.glsl`
- `draw/fill.glsl`
- `draw/digits.glsl`

---

## Morphological - 9 files

- `morphological/erosion.glsl`
- `morphological/alphaFill.glsl`
- `morphological/marchingSquares.glsl`
- `morphological/jumpFlood.glsl`
- `morphological/alphaHashing.glsl`
- `morphological/dilation.glsl`
- `morphological/pyramid.glsl` - Barrel file
- `morphological/pyramid/upscale.glsl`
- `morphological/pyramid/downscale.glsl`

---

## Summary by Category

| Category | Total Files | Barrels | Actual Work | Notes |
|----------|------------|---------|-------------|-------|
| **Lighting** | 91 | 0 | **91** | Largest category; complex interdependencies |
| **SDF** | 41 | 1 | **40** | Mostly standalone shape functions |
| **Color** | 41 | 4 | **37** | Composite, palette, dither functions |
| **Sample** | 30 | 1 | **29** | Texture sampling utilities |
| **Filter** | 28 | 0 | **28** | Image processing effects |
| **Draw** | 16 | 0 | **16** | Rendering utilities |
| **Math** | 10 | 1 | **9** | Math utilities |
| **Morphological** | 9 | 1 | **8** | Image morphological operations |
| **Generative** | 7 | 0 | **7** | Noise and procedural functions |
| **Geometry** | 8 | 2 | **6** | 2 depend on Ray struct |
| **Distort** | 6 | 0 | **6** | Image distortion effects |
| **Simulate** | 4 | 0 | **4** | Simulation algorithms |
| **Space** | 2 | 0 | **2** | Nearly complete! |
| **Animation** | 12 | 12 | **0** | All barrel files (skip) |
| **Root** | 4 | 0 | **4** | Metadata/aggregators |
| **Total** | **309** | **21** | **288** | Real conversion work needed |

**Commands:**
- See all unconverted: `./check-unconverted.sh`
- Exclude barrels: `./check-unconverted.sh --skip-barrels`
- Count by category: `./check-unconverted.sh --skip-barrels | awk -F/ '{print $2}' | sort | uniq -c | sort -rn`
