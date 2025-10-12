# Untested WESL Files

This document lists all WESL files that have been converted but do NOT have test coverage yet.

**Total WESL files**: 351
**Files with tests**: 171 (48.7%)
**Files without tests**: 180 (51.3%)

---

## 📊 Summary by Category

| Category | Untested Files |
|----------|----------------|
| color | 88 |
| math | 54 |
| geometry | 11 |
| space | 7 |
| sdf | 7 |
| lighting | 5 |
| filter | 3 |
| version | 1 |
| sample | 1 |
| generative | 1 |
| draw | 1 |
| animation | 1 |

---

## 🎯 Priority: High-Value Functions to Test

### Color Functions (88 untested) - **HIGH PRIORITY**
**Why**: Color operations are fundamental, widely used, and easy to test
**Includes**: blend modes, composite operations, layer blending

**Examples of untested**:
- `color/blend/add`, `color/blend/multiply`, `color/blend/screen`
- `color/composite/destinationAtop`, `color/composite/sourceOut`
- `color/layer/*SourceOver` functions (20+ functions)
- `color/dither/*` functions
- `color/mixSpectral`, `color/hueShiftRYB`

### Lighting Functions (5 untested) - **MEDIUM PRIORITY**
**Why**: Important for 3D rendering

```
lighting/common/ggx
lighting/common/schlick
lighting/diffuse/orenNayar
lighting/raymarch/normal
lighting/toShininess
```

### Geometry Functions (11 untested) - **MEDIUM PRIORITY**
**Why**: Spatial operations used in many applications

<details>
<summary>View list</summary>

```
geometry/aabb/aabb
geometry/aabb/size
geometry/rect/contains
geometry/rotate
geometry/scale
geometry/triangle/rotate
geometry/triangle/triangle
```

</details>

### Math Functions (54 untested) - **LOW-MEDIUM PRIORITY**
**Why**: Many may be simple wrappers; focus on complex ones
**Note**: Some may be WGSL built-in wrappers that need minimal testing

### SDF Functions (7 untested) - **MEDIUM PRIORITY**
**Why**: Useful for procedural graphics

```
sdf/boxSDF
sdf/circleSDF  
sdf/crossSDF
sdf/rectSDF
sdf/sphereSDF
sdf/torusSDF
sdf/opSubtraction
```

---

## 📝 Full List by Category

### Animation (1 untested)
animation/spriteLoop

### Color (88 untested)
<details>
<summary>Click to expand</summary>

color/blend/add
color/blend/average
color/blend/colorBurn
color/blend/colorDodge
color/blend/darken
color/blend/difference
color/blend/exclusion
color/blend/lighten
color/blend/linearBurn
color/blend/linearDodge
color/blend/multiply
color/blend/negation
color/blend/overlay
color/blend/phoenix
color/blend/reflect
color/blend/screen
color/blend/softLight
color/blend/subtract
color/brightnessContrast
color/composite/destinationAtop
color/composite/destinationIn
color/composite/destinationOut
color/composite/destinationOver
color/composite/sourceAtop
color/composite/sourceOut
color/exposure
color/hueShiftRYB
color/layer/averageSourceOver
color/layer/colorBurnSourceOver
color/layer/colorDodgeSourceOver
color/layer/colorSourceOver
color/layer/glowSourceOver
color/layer/hardLightSourceOver
color/layer/hardMixSourceOver
color/layer/hueSourceOver
color/layer/linearBurnSourceOver
color/layer/linearDodgeSourceOver
color/layer/linearLightSourceOver
color/layer/luminositySourceOver
color/layer/negationSourceOver
color/layer/pinLightSourceOver
color/layer/reflectSourceOver
color/layer/saturationSourceOver
color/layer/softLightSourceOver
color/layer/vividLightSourceOver
color/levels/outputRange
color/mixSpectral
color/palette/heatmap
color/palette/hue
color/saturationMatrix
color/space/YCbCr2rgb
color/space/YPbPr2rgb
color/space/cmyk2rgb
color/space/gamma2linear
color/space/hue2rgb
color/space/k2rgb
color/space/lab2lch
color/space/lab2xyz
color/space/lch2lab
color/space/linear2gamma
color/space/lms2rgb
color/space/oklab2rgb
color/space/rgb2YCbCr
color/space/rgb2cmyk
color/space/rgb2hcv
color/space/rgb2hue
color/space/rgb2lab
color/space/rgb2lch
color/space/rgb2lms
color/space/rgb2luma
color/space/rgb2oklab
color/space/rgb2srgb
color/space/rgb2yiq
color/space/srgb2luma
color/space/srgb2rgb
color/space/xyY2xyz
color/space/xyz2lab
color/space/xyz2rgb
color/space/xyz2srgb
color/space/xyz2xyY
color/space/yiq2rgb
color/tonemap/aces
color/tonemap/debug
color/tonemap/filmic
color/tonemap/reinhardJodie
color/tonemap/uncharted
color/tonemap/uncharted2
color/whiteBalance

</details>

### Math (54 untested)
<details>
<summary>Click to expand</summary>

math/aamirror
math/aastep
math/adaptiveThreshold
math/atan2
math/bump
math/consts
math/cubic
math/decimate
math/dist
math/fcos
math/gain
math/gaussian
math/grad4
math/highPass
math/inside
math/invCubic
math/invQuartic
math/inverse
math/lengthSq
math/map
math/mirror
math/mmax
math/mmin
math/mod2
math/mod289
math/pack
math/parabola
math/powFast
math/quartic
math/quat
math/quat/div
math/quat/inverse
math/quat/neg
math/quintic
math/rotate2d
math/rotate3d
math/rotate3dX
math/rotate3dY
math/rotate3dZ
math/rotate4d
math/rotate4dX
math/rotate4dY
math/rotate4dZ
math/round
math/saturateMediump
math/scale2d
math/scale3d
math/scale4d
math/sum
math/taylorInvSqrt
math/toMat4
math/translate4d
math/unpack
math/within

</details>

### Geometry (11 untested)
geometry/aabb/aabb
geometry/aabb/centroid
geometry/aabb/contain
geometry/aabb/diagonal
geometry/aabb/expand
geometry/aabb/square
geometry/triangle/area
geometry/triangle/barycentric
geometry/triangle/centroid
geometry/triangle/normal
geometry/triangle/triangle

### Space (7 untested)
space/nearest
space/ratio
space/rotate
space/scale
space/sprite
space/translate
space/unratio

### SDF (7 untested)
sdf/boxSDF
sdf/cylinderSDF
sdf/opSubtraction
sdf/opUnion
sdf/rectSDF
sdf/sphereSDF
sdf/torusSDF

### Lighting (5 untested)
lighting/common/ggx
lighting/common/schlick
lighting/diffuse/orenNayar
lighting/raymarch/normal
lighting/toShininess

### Filter (3 untested)
filter/edge/prewitt
filter/sharpen/adaptive
filter/sharpen/fast

### Other Categories (3 untested)
draw/stroke
generative/random
sample/sprite
version

---

## 🔧 Usage Commands

```bash
# List all untested WESL files
./find-untested-wesl.sh

# Count untested files by category  
./find-untested-wesl.sh --count

# Find untested files in a specific category
./find-untested-wesl.sh | grep "^color/"
./find-untested-wesl.sh | grep "^math/"

# Compare with total WESL files
find . -name "*.wesl" | grep -v node_modules | wc -l
```

---

## 💡 Recommended Testing Strategy

### Phase 1: Color Functions (Quick Wins) ⚡
**Time estimate**: 4-6 hours
**Impact**: High (88 files, ~49% of untested)

1. **Blend modes** (19 untested) - Use pattern from existing color tests
   - Test with known color pairs: `[0.5, 0.5, 0.5] + [0.3, 0.2, 0.1]`
   - Verify against expected results

2. **Layer blending** (20+ untested) - Similar pattern to blends
   - Many already have reference implementations

3. **Composite operations** (6 untested) - Alpha compositing tests
   - Test with semi-transparent colors

### Phase 2: Geometry & SDF (Medium Impact) 🎯
**Time estimate**: 2-3 hours  
**Impact**: Medium (18 files total)

- Geometry functions: Test with known shapes and transformations
- SDF functions: Test distance calculations with known points

### Phase 3: Math Functions (Case-by-Case) 🔢
**Time estimate**: Variable
**Impact**: Low-Medium (many may be trivial)

- Review each function individually
- Skip trivial WGSL built-in wrappers
- Focus on complex calculations

### Phase 4: Fix Skipped Tests 🛠️
**Time estimate**: 2-4 hours
**Impact**: High (improve reliability)

See [Skipped Tests](#-skipped-tests) section below

---

## ⏭️ Skipped Tests (6 tests)

These tests exist but are currently skipped due to known issues:

### Fragment Shader Required (2 tests)
**Issue**: Use `fwidth()` which requires derivatives, only available in fragment shaders

1. **`aafloor`** - Anti-aliased floor function
   - File: `math/aafloor.wesl`
   - Test: `test/wesl/functions.test.ts`
   - Solution: Implement fragment shader test harness (see [test-fragshader.md](test-fragshader.md))

2. **`aafract`** - Anti-aliased fract function
   - File: `math/aafract.wesl`
   - Test: `test/wesl/functions.test.ts`
   - Solution: Same as aafloor

### Signature Mismatches (3 tests)
**Issue**: Test expectations don't match actual function signatures

3. **`fresnel`** - Fresnel effect
   - Test expects: `(vec3f, vec3f)`
   - Actual signature: `(f0: f32, NoV: f32)`
   - Solution: Fix test to match actual signature or create wrapper

4. **`fresnelReflection`** - Fresnel reflection
   - Similar signature mismatch
   - Solution: Review function and update test

5. **`cookTorrance`** / **`specularCookTorrance`**
   - Function is named `specularCookTorrance` with different parameters
   - Solution: Update test to use correct name and parameters

### Scene-Specific Function (1 test)

6. **`raymarchCast`** - Raymarch casting
   - Issue: Requires user-defined `map()` function (scene SDF)
   - Solution: Define a sample scene in the test

---

## 📚 Test File Templates

### Color Blend Test Pattern
```typescript
import { testColor } from "./testUtil.ts";

test("blendAdd", async () => {
  await testColor("blend/add", "blendAdd3", 
    [0.5, 0.3, 0.2], // color1
    [0.2, 0.4, 0.6], // color2
    [0.7, 0.7, 0.8]  // expected result
  );
});
```

### SDF Test Pattern  
```typescript
import { testCompute } from "./testUtil.ts";

test("sphereSDF", async () => {
  const result = await testCompute("sdf/sphereSDF", "sphereSDF",
    [[0, 0, 0], // point
     [0, 0, 0], // sphere center
     1.0]       // radius
  );
  expect(result).toBeCloseTo(-1.0); // inside sphere
});
```

### Geometry Test Pattern
```typescript
import { testCompute } from "./testUtil.ts";

test("triangleArea", async () => {
  const result = await testCompute("geometry/triangle/area", "triangleArea",
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]] // three vertices
  );
  expect(result).toBeCloseTo(0.5);
});
```

---

## Related Files
- [next-steps.md](next-steps.md) - Overall project roadmap
- [test-fragshader.md](test-fragshader.md) - Fragment shader testing plan
- [GlslToConvert.md](GlslToConvert.md) - Conversion tracking
- Test utilities: [test/wesl/testUtil.ts](test/wesl/testUtil.ts)
- Test files: [test/wesl/](test/wesl/)

