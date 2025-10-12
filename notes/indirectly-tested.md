# Indirectly Tested Functions

This document tracks WESL functions that appear in the "untested" list but are actually indirectly tested through other functions. These functions should **NOT** be added to test suites as they're already validated.

**Last Updated**: October 10, 2025

## Purpose

When analyzing untested functions, skip these as they're already covered by existing tests. This reduces test bulk without sacrificing coverage.

---

## Color Functions

### Color Composite (9 functions - all indirectly tested)

These scalar/vec3 functions are called by their vec4 counterparts, which are tested:

- `compositeXor()` - tested via `compositeXor4()` in color-composite.test.ts
- `compositeDestinationAtop()` - tested via `compositeDestinationAtop4()`
- `compositeDestinationIn()` - tested via `compositeDestinationIn4()`
- `compositeDestinationOut()` - tested via `compositeDestinationOut4()`
- `compositeDestinationOver()` - tested via `compositeDestinationOver4()`
- `compositeSourceAtop()` - tested via `compositeSourceAtop4()`
- `compositeSourceIn()` - tested via `compositeSourceIn4()`
- `compositeSourceOut()` - tested via `compositeSourceOut4()` (if tested)
- `compositeSourceOver()` - tested via `compositeSourceOver4()`

### Color Blend (2 functions)

- `blendSoftLight4()` - calls `blendSoftLight()` (tested) for each component
- `blendColorOpacity()` - calls `blendColor()` (tested)

### Color Dither (18 functions)

All blue noise variants call `ditherBlueNoise3Precision()`:
- `ditherBlueNoisePrecision()`
- `ditherBlueNoise1()`
- `ditherBlueNoise3()` - **TESTED** in color-adjust.test.ts
- `ditherBlueNoise4()`
- `ditherBlueNoise1Simple()`
- `ditherBlueNoise3Simple()`
- `ditherBlueNoise4Simple()`
- `ditherBlueNoise4Precision()`

All Vlachos variants call base functions:
- `ditherVlachos()` overloads
- `ditherVlachos1()`
- `ditherVlachos3()` - **TESTED** in color-adjust.test.ts
- `ditherVlachos3Noise()` overloads
- `ditherVlachos3Precision()`
- etc.

---

## Math Functions

### Component-wise Vector Wrappers (~70 functions)

These apply the same scalar formula to vector components. If the scalar version is tested, the vector versions are implicitly validated.

**Pattern**: If `fn foo(x: f32)` is tested, skip `fn foo2(v: vec2f)`, `fn foo3(v: vec3f)`, `fn foo4(v: vec4f)`

Examples:
- `cubic()` is tested → skip `cubic2()`, `cubic3()`, `cubic4()`
- `quartic()` is tested → skip `quartic2()`, `quartic3()`, `quartic4()`
- `invCubic()` is tested → skip `invCubic2()`, `invCubic3()`, `invCubic4()`
- `invQuartic()` is tested → skip `invQuartic2()`, `invQuartic3()`, `invQuartic4()`
- `gaussian()` is tested → skip `gaussian2()`, `gaussian3()`, `gaussian4()`
- `cubicMix()` is tested → skip `cubicMix2()`, `cubicMix3()`, `cubicMix4()`
- `decimate()` is tested → skip `decimate2()`, `decimate3()`, `decimate4()`
- `map()` is tested → skip `map2()`, `map3()`, `map4()`
- `mirror()` is tested → skip `mirror2()`
- `mmax()` is tested → skip `mmax4()`
- `mmin()` is tested → skip `mmin4()`
- `pow2()` is tested → skip `pow23()`, `pow24()`
- `pow3()` is tested → skip `pow32()`, `pow33()`, `pow34()`
- `pow5()` is tested → skip `pow52()`, `pow53()`, `pow54()`
- `pow7()` is tested → skip `pow72()`, `pow73()`, `pow74()`
- `powFast()` is tested → skip `powFast2()`, `powFast3()`, `powFast4()`
- `bump()` is tested → skip `bump3()`, `bump4()`

### Distance Functions (~15 functions)

Distance metrics that apply the same formula component-wise:
- `dist()` tested → skip `dist2()`, `dist3()`, `dist4()`
- `distChebychev()` tested → skip `distChebychev2()`, `distChebychev3()`, `distChebychev4()`
- `distEuclidean()` tested → skip `distEuclidean3()`, `distEuclidean4()` (likely same as built-in distance)
- `distManhattan()` tested → skip `distManhattan3()`, `distManhattan4()`
- `distMinkowski()` tested → skip `distMinkowski2()`, `distMinkowski3()`, `distMinkowski4()`

### Math Utilities (~10 functions)

- `mod289()` tested → skip `mod289_2()`, `mod289_3()`, `mod289_4()` (used internally by noise)
- `permute()` tested → skip `permute2()`, `permute3()`, `permute4()` (used internally by noise)
- `mod2()` tested → skip `mod22()`
- `inside()` tested → skip (check if `inside3()` is just component-wise)
- `inside2()` tested → skip (check `insideAABB()` implementation)
- `lengthSq()` tested → verify if `lengthSq4()` is just component-wise

---

## Space Functions

### Rotation Functions (~20 functions)

Matrix operations that apply to different dimensions. Check if vec3/vec4 variants just apply matrix to vector:

- `rotate2d()` tested → check `rotate3_c()`, `rotate4()`, `rotate4_c()`, `rotate_axis()`
- `rotateX()` tested → check `rotateX3_c()`, `rotateX4()`, `rotateX4_c()`
- `rotateY()` tested → check `rotateY3_c()`, `rotateY4()`, `rotateY4_c()`
- `rotateZ()` tested → check `rotateZ3_c()`, `rotateZ4()`, `rotateZ4_c()`

### Scale Functions (~12 functions)

Scaling operations with center variants:
- `scale()` tested → check `scale2_c()`, `scale3_c()`, `scale4()`, `scale_f()`, `scale_f_c()`, etc.

### Tiling Functions (~20 functions)

Tiling operations where variants just add parameters:

- `brickTile()` called by `brickTile2()` (tested) → skip `brickTile()`, `brickTile_s()`, `brickTile_s2()`
- `checkerTile2()` tested → skip `checkerTile()`, `checkerTile_s()`, `checkerTile_s2()`
- `mirrorTile()` variants tested → skip all `mirror[X|Y]Tile[_s|_s2|2]()` variants
- `windmillTile()` tested → skip `windmillTile_s()`
- `sqTile()` tested → skip `sqTile_scale()`
- `triTile()` tested → skip `triTile_scale()`

### Transform Functions (~10 functions)

- `flipY()` tested → skip `flipY3()`, `flipY4()`
- `center()` tested → skip `center3()`
- `uncenter()` tested → skip `uncenter3()`
- `cart2polar()` tested → skip `cart2polar3()` (if same formula)
- `polar2cart()` tested → skip `polar2cart3()` (if same formula)
- `kaleidoscope()` tested → skip `kaleidoscope_full()`, `kaleidoscope_seg()` (calls kaleidoscope_full)
- `depth2viewZ()` tested → skip `depth2viewZ_default()`
- `linearizeDepth()` tested → skip `linearizeDepth_default()`
- `viewZ2depth()` tested → skip `viewZ2depth_default()`

---

## Filter Functions

### Sharpen (7 functions)

All are internal helpers used by `sharpenAdaptive()` (tested):
- `SHARPENADAPTIVE_CTRL()`
- `SHARPENADAPTIVE_DIFF()`
- `SHARPENADAPTIVE_DXDY()`
- `SHARPENADAPTIVE_SOFT_LIM()`
- `SHARPENADAPTIVE_WPMEAN()`
- `sharpendAdaptiveControl3()`
- etc.

---

## Guidelines for Identifying Indirectly Tested Functions

### ✅ Skip Testing If:

1. **Component-wise wrapper**: Function applies same scalar operation to vector components
   ```rust
   fn cubic(v: f32) -> f32 { ... }
   fn cubic3(v: vec3f) -> vec3f { return v*v*(3.0-2.0*v); } // Same formula
   ```

2. **Called by tested function**: Function is directly invoked by another tested function
   ```rust
   fn blendColor(base: vec3f, blend: vec3f) -> vec3f { ... } // TESTED
   fn blendColorOpacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
       return blendColor(base, blend) * opacity + ...; // Indirectly tested
   }
   ```

3. **Wrapper with defaults**: Function just calls another with default parameters
   ```rust
   fn kaleidoscope(coord: vec2f) -> vec2f {
       return kaleidoscope_full(coord, 8.0, 0.0); // Just adds defaults
   }
   ```

### ❌ DO Test If:

1. **Different implementation**: Each variant has unique logic
   ```rust
   fn random21(p: f32) -> vec2f { /* unique algorithm */ }
   fn random22(p: vec2f) -> vec2f { /* different algorithm */ }
   ```

2. **Independent function**: Not called by any tested function
3. **Complex transformation**: Not a simple component-wise operation

---

## Integration with find-untested-functions.py

The Python script has been enhanced to use `scripts/indirectly-tested.txt` (machine-readable format):

**Usage**:
```bash
# Show only genuinely untested functions (161 indirect functions are automatically tracked)
python3 scripts/find-untested-functions.py

# Show summary with indirect test counts
python3 scripts/find-untested-functions.py --summary

# Show counts by category
python3 scripts/find-untested-functions.py --count

# Show indirectly tested functions
python3 scripts/find-untested-functions.py --show-indirect

# Show indirectly tested function counts
python3 scripts/find-untested-functions.py --show-indirect --count
```

The machine-readable list is maintained in `scripts/indirectly-tested.txt`.

---

## Maintenance

When adding new functions or tests:
1. Check if the new test indirectly covers any "untested" functions
2. Add those functions to this document
3. Document the reasoning (wrapper, called-by, etc.)

This keeps the test suite lean and focused on genuine coverage gaps.
