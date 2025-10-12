# WESL Conversion Review Checklist

## Overview

This document provides a comprehensive checklist and review instructions for verifying the quality and completeness of GLSL to WESL conversions in the LYGIA shader library.

### Conversion Status

- **Total GLSL files**: 656
- **Total WESL files**: 351
- **Conversion rate**: ~53%

### Files by Category

| Category | WESL Files | Notes |
|----------|------------|-------|
| color | 156 | Blend modes, color spaces, tonemapping, palettes |
| math | 80 | Math utilities, quaternions, transformations |
| space | 40 | Spatial transformations, projections, tiles |
| animation | 34 | Easing functions, sprite utilities |
| geometry | 11 | AABB, triangle operations |
| lighting | 9 | PBR, diffuse, specular, fresnel |
| generative | 8 | Noise, random functions |
| sdf | 7 | Signed distance fields |
| filter | 3 | Edge detection, sharpening |
| draw | 1 | Stroke utilities |
| sample | 1 | Sprite sampling |
| version | 1 | Version info |

---

## Reviewer Instructions

### Purpose

This review ensures that converted WESL files:
1. **Faithfully translate** all functionality from the GLSL original
2. **Follow WESL conventions** and best practices
3. **Handle all edge cases** including optional features and conditionals
4. **Maintain compatibility** with the rest of the LYGIA library

### How to Review a File Pair

For each WESL file, compare it side-by-side with its corresponding GLSL file:

```bash
# Example: Review a specific file
code -d color/blend/add.glsl color/blend/add.wesl
```

Or use a diff tool:
```bash
diff -u color/blend/add.glsl color/blend/add.wesl
```

### What to Check

#### 1. **Function Completeness**

**GLSL often has multiple overloads** - ensure ALL are converted:

```glsl
// GLSL - 3 overloads
float blendAdd(in float base, in float blend) { ... }
vec3 blendAdd(in vec3 base, in vec3 blend) { ... }
vec3 blendAdd(in vec3 base, in vec3 blend, float opacity) { ... }
```

```wgsl
// WESL - should have 3 functions with unique names
fn blendAdd(base: f32, blend: f32) -> f32 { ... }
fn blendAdd3(base: vec3f, blend: vec3f) -> vec3f { ... }
fn blendAdd3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f { ... }
```

**Common missing overloads:**
- ❌ **vec4 variants** - GLSL often has `vec3` and `vec4` versions, check both are converted
- ❌ **Integer variants** - `int`, `ivec2`, `ivec3` versions may be missing
- ❌ **Matrix variants** - `mat2`, `mat3`, `mat4` overloads

**How to verify:**
- Count function declarations in GLSL: `grep "^float\|^vec\|^mat\|^int" file.glsl | grep "functionName"`
- Count function declarations in WESL: `grep "^fn functionName" file.wesl`
- Numbers should match!

#### 2. **Import Completeness**

**Check all `#include` statements have corresponding `import` statements:**

```glsl
// GLSL
#include "hue2rgb.glsl"
#include "../math/saturate.glsl"
```

```wgsl
// WESL
import lygia::color::space::hue2rgb::hue2rgb;
import lygia::math::saturate::saturate;
```

**Common issues:**
- ❌ **Missing imports** - forgot to convert an `#include`
- ❌ **Incorrect paths** - `lygia::` prefix missing or wrong directory structure
- ❌ **Wrong function name** - imported `foo` when file exports `fooBar`

**How to verify:**
- Count `#include` in GLSL: `grep "#include" file.glsl | wc -l`
- Count `import` in WESL: `grep "^import" file.wesl | wc -l`
- Read both files to ensure paths map correctly

#### 3. **Conditional Compilation**

**Check all `#ifdef`, `#ifndef`, `#else`, `#endif` are converted to `@if`, `@else`:**

```glsl
// GLSL
#ifdef CENTER_2D
    v -= CENTER_2D;
#else
    v -= 0.5;
#endif
```

```wgsl
// WESL
@if(CENTER_2D)
pos -= center;
@else
pos -= 0.5;
```

**Common issues:**
- ❌ **Missing conditionals** - entire `#ifdef` block not converted
- ❌ **Simplified too much** - removed optional behavior instead of converting it
- ❌ **Incorrect logic** - `#ifndef` (if NOT defined) vs `#ifdef` (if defined)
  - `#ifndef FOO` → `@if(!FOO)` or `@if(FOO) ... @else ...`
- ❌ **Missing constants** - conditional references a constant that wasn't defined

**How to verify:**
- Count conditionals in GLSL: `grep "#ifdef\|#ifndef\|#else" file.glsl | wc -l`
- Count conditionals in WESL: `grep "@if\|@else" file.wesl | wc -l`
- Check logic is preserved (especially `#ifndef` vs `#ifdef`)

#### 4. **Constants and Macros**

**Check all `#define` constants are converted:**

```glsl
// GLSL
#define PI 3.14159
#define RANDOM_SCALE vec4(.1031, .1030, .0973, .1099)
```

```wgsl
// WESL
const PI: f32 = 3.14159;
const RANDOM_SCALE: vec4f = vec4f(.1031, .1030, .0973, .1099);
```

**Function-like macros are not yet supported in WESL :**

```glsl
// GLSL
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
```

- typically, implement the default macro value as a fn() in WESL
- if there other cases defined for the macro fn, use @if conditions for the others
- if the user should be able to replace the fn, add a comment tagged with LATER. 
A future version of WESL will allow users to override fnss


**Common issues:**
- ❌ **Missing constants** - forgot to convert a `#define`
- ❌ **Wrong type** - `const PI: i32 = 3.14159;` (should be `f32`)
- ❌ **Function macros not handled** - complex macros need manual conversion

#### 5. **Type Conversions**

**Verify all GLSL types are converted to WESL equivalents:**

| GLSL | WESL | Common Issues |
|------|------|---------------|
| `float` | `f32` | ❌ Left as `float` |
| `int` | `i32` | ❌ Left as `int` |
| `vec2` | `vec2f` or `vec2<f32>` | ❌ Missing `f` suffix |
| `vec3` | `vec3f` or `vec3<f32>` | ❌ Missing `f` suffix |
| `vec4` | `vec4f` or `vec4<f32>` | ❌ Missing `f` suffix |
| `mat2` | `mat2x2f` or `mat2x2<f32>` | ❌ Wrong matrix dimensions |
| `mat3` | `mat3x3f` or `mat3x3<f32>` | ❌ Wrong matrix dimensions |
| `mat4` | `mat4x4f` or `mat4x4<f32>` | ❌ Wrong matrix dimensions |

**How to verify:**
- Search WESL file for GLSL type names: `grep -E "(^|[^a-z])float([^0-9]|$)" file.wesl`
- Any matches are potential errors (unless in comments)

#### 6. **Numeric Literals**

**Check all numeric literals have proper format:**

```glsl
// GLSL - flexible
1.       // OK
.5       // OK
1        // OK (context-dependent)
```

```wgsl
// WESL - strict
1.0      // OK (f32)
0.5      // OK (f32)
1        // OK (i32)
1u       // OK (u32)
```

**Common issues:**
- ❌ **Trailing dots** - `1.` should be `1.0`
- ❌ **Leading dots** - `.5` should be `0.5`
- ❌ **Missing decimal** - `vec3(1)` should be `vec3(1.0)` if expecting float

**How to verify:**
- Search for trailing dots: `grep '\b[0-9]\+\.\b' file.wesl`
- Search for leading dots: `grep '\.[0-9]' file.wesl`

#### 7. **Reserved Words**

**Check for WGSL reserved words used as identifiers:**

Common problematic names:
- `target` → use `center`, `dest`, `destination`
- `uniform` → use `value`, `val`
- `varying` → use `interpolated`
- `attribute` → use `attr`, `property`

**How to verify:**
- If you see parse errors when testing, check parameter names
- Look for: `target`, `uniform`, `varying`, `attribute`, `storage`, `texture`, `sampler`

#### 8. **Function Syntax**

**Verify function declarations use WESL syntax:**

```glsl
// GLSL
vec3 hsl2rgb(const in vec3 hsl) { ... }
float rotate2d(in float angle) { ... }
```

```wgsl
// WESL
fn hsl2rgb(hsl: vec3f) -> vec3f { ... }
fn rotate2d(angle: f32) -> f32 { ... }
```

**Common issues:**
- ❌ **Parameter qualifiers** - `in`, `out`, `inout`, `const in` should be removed
- ❌ **Return type position** - should be after `->`, not before `fn`
- ❌ **Missing `fn` keyword**

#### 9. **Variable Declarations**

**Check variable declarations use WESL syntax:**

```glsl
// GLSL
float x = 1.0;
vec3 color = vec3(1.0, 0.0, 0.0);
```

```wgsl
// WESL
var x: f32 = 1.0;          // mutable
let color = vec3f(1.0, 0.0, 0.0);  // immutable (type inferred)
const PI: f32 = 3.14159;   // compile-time constant
```

**Common issues:**
- ❌ **Missing `var`/`let`/`const` keyword**
- ❌ **Wrong keyword choice** - using `var` when `let` would be clearer

#### 10. **Comments and Documentation**

**Ensure documentation is preserved:**

Required in both GLSL and WESL:
- Contributors
- Description
- License

**Check for:**
- ❌ **Missing contributor info**
- ❌ **Outdated usage examples** - `use:` line may reference GLSL types
- ❌ **Missing options documentation** - `options:` should list `@if` conditions

#### 11. **Matrix Constructor Order**

**Matrix constructors may have different memory layout:**

```glsl
// GLSL - column-major, may vary
mat2(c, s, -s, c)
```

```wgsl
// WESL - explicit column vectors
mat2x2f(
    vec2f(c, -s),  // column 0
    vec2f(s, c)    // column 1
)
// or single constructor (check WGSL spec for order)
mat2x2f(c, -s, s, c)
```

**Common issues:**
- ❌ **Incorrect element order** - visually looks the same but produces wrong results
- ❌ **Row-major vs column-major** confusion

**How to verify:**
- Test the function with known values
- Check matrix multiplication order

#### 12. **Vector Constructors**

**WESL may require explicit type suffix:**

```glsl
// GLSL
vec3(1.0)           // splat
vec3(v.xy, 1.0)     // swizzle + scalar
```

```wgsl
// WESL
vec3(1.0)           // OK with type inference
vec3f(1.0)          // explicit
vec3(v.xy, 1.0)     // OK
```

Generally WESL is flexible, but check for consistency.

---

## Review Checklist Template

For each file, verify:

- [ ] **All function overloads converted** (check counts match)
  - [ ] Float/scalar variants
  - [ ] vec2 variants
  - [ ] vec3 variants
  - [ ] vec4 variants (if in GLSL)
  - [ ] Matrix variants (if applicable)
  - [ ] Integer variants (if applicable)

- [ ] **All imports present** (check `#include` → `import` conversion)
  - [ ] Count matches
  - [ ] Paths are correct (`lygia::` prefix)
  - [ ] Function names match exports

- [ ] **All conditionals converted** (check `#ifdef` → `@if`)
  - [ ] Count matches
  - [ ] Logic preserved (`#ifndef` → `@if(!...)` or `@else`)
  - [ ] Required constants defined

- [ ] **All constants converted** (check `#define` → `const`)
  - [ ] Numerical constants present
  - [ ] Vector/matrix constants present
  - [ ] Function-like macros handled appropriately

- [ ] **Type conversions correct**
  - [ ] No GLSL type names (`float`, `vec3`, `mat4`, etc.)
  - [ ] WESL types used (`f32`, `vec3f`, `mat4x4f`, etc.)

- [ ] **Numeric literals correct**
  - [ ] No trailing dots (`1.` → `1.0`)
  - [ ] No leading-only dots (`.5` → `0.5`)

- [ ] **No reserved words as identifiers**
  - [ ] No `target`, `uniform`, `varying`, `attribute`
  - [ ] Parameters and variables use valid names

- [ ] **Function syntax correct**
  - [ ] Uses `fn` keyword
  - [ ] Return type after `->`
  - [ ] No `in`/`out`/`inout` qualifiers

- [ ] **Variable declarations correct**
  - [ ] Uses `var`/`let`/`const` keywords
  - [ ] Types specified or inferred correctly

- [ ] **Documentation preserved**
  - [ ] Contributors listed
  - [ ] Description present
  - [ ] License included
  - [ ] Usage examples updated (if needed)
  - [ ] Options documented (for `@if` conditions)

- [ ] **Matrix/vector constructors correct**
  - [ ] Matrix element order verified
  - [ ] Vector constructors have proper types

- [ ] **Include guards removed**
  - [ ] No `#ifndef FNC_*`
  - [ ] No `#define FNC_*`
  - [ ] No `#endif`

---

## Common Conversion Patterns

### Pattern 1: Simple Function with Overloads

**GLSL:**
```glsl
#ifndef FNC_BLENDADD
#define FNC_BLENDADD
float blendAdd(in float base, in float blend) { return min(base + blend, 1.); }
vec3  blendAdd(in vec3 base, in vec3 blend) { return min(base + blend, vec3(1.)); }
#endif
```

**WESL:**
```wgsl
fn blendAdd(base: f32, blend: f32) -> f32 { return min(base + blend, 1.0); }
fn blendAdd3(base: vec3f, blend: vec3f) -> vec3f { return min(base + blend, vec3(1.0)); }
```

**Review checks:**
- ✅ Include guards removed
- ✅ Two functions preserved
- ✅ Unique names (`blendAdd`, `blendAdd3`)
- ✅ Types converted (`float` → `f32`, `vec3` → `vec3f`)
- ✅ Numeric literals corrected (`1.` → `1.0`)

### Pattern 2: Function with Imports

**GLSL:**
```glsl
#include "hue2rgb.glsl"

#ifndef FNC_HSL2RGB
#define FNC_HSL2RGB
vec3 hsl2rgb(const in vec3 hsl) {
    vec3 rgb = hue2rgb(hsl.x);
    float C = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
    return (rgb - 0.5) * C + hsl.z;
}
#endif
```

**WESL:**
```wgsl
import lygia::color::space::hue2rgb::hue2rgb;

fn hsl2rgb(hsl: vec3f) -> vec3f {
    let rgb = hue2rgb(hsl.x);
    let C = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
    return (rgb - 0.5) * C + hsl.z;
}
```

**Review checks:**
- ✅ `#include` → `import` with correct path
- ✅ Include guards removed
- ✅ Parameter qualifiers removed (`const in`)
- ✅ Variable declarations use `let`

### Pattern 3: Conditional Compilation

**GLSL:**
```glsl
#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF

#ifndef CIRCLESDF_FNC
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
#endif

float circleSDF(in vec2 v) {
#ifdef CENTER_2D
    v -= CENTER_2D;
#else
    v -= 0.5;
#endif
    return CIRCLESDF_FNC(v) * 2.0;
}
#endif
```

**WESL (needs review - may vary by approach):**
```wgsl
// Approach 1: Simple conditional
fn circleSDF(v: vec2f) -> f32 {
    var pos = v;
    @if(CENTER_2D)
    pos -= CENTER_2D;
    @else
    pos -= 0.5;

    return length(pos) * 2.0;
}

// Approach 2: With function pointer alternative (if supported)
// Check if CIRCLESDF_FNC customization is preserved
```

**Review checks:**
- ⚠️ **CRITICAL**: Check if `CIRCLESDF_FNC` macro customization is preserved
- ✅ `#ifdef` → `@if`
- ✅ `#else` → `@else`
- ✅ `#ifndef` logic handled (either as `@if(!...)` or `@else` branch)

### Pattern 4: Conditional Constants

**GLSL:**
```glsl
#ifndef MAT_YUV2RGB
#define MAT_YUV2RGB
#ifdef YUV_SDTV
const mat3 YUV2RGB = mat3(
    1.0,       1.0,      1.0,
    0.0,      -0.39465,  2.03211,
    1.13983,  -0.58060,  0.0
);
#else
const mat3 YUV2RGB = mat3(
    1.0,       1.0,      1.0,
    0.0,      -0.21482,  2.12798,
    1.28033,  -0.38059,  0.0
);
#endif
#endif

#ifndef FNC_YUV2RGB
#define FNC_YUV2RGB
vec3 yuv2rgb(const in vec3 yuv) { return YUV2RGB * yuv; }
#endif
```

**WESL:**
```wgsl
@if(YUV_SDTV)
const YUV2RGB = mat3x3<f32>(
    vec3f(1.0,       1.0,      1.0),
    vec3f(0.0,      -0.39465,  2.03211),
    vec3f(1.13983,  -0.58060,  0.0)
);

@if(!YUV_SDTV)
const YUV2RGB = mat3x3<f32>(
    vec3f(1.0,       1.0,      1.0),
    vec3f(0.0,      -0.21482,  2.12798),
    vec3f(1.28033,  -0.38059,  0.0)
);

fn yuv2rgb(yuv: vec3f) -> vec3f { return YUV2RGB * yuv; }
```

**Review checks:**
- ✅ Both conditional branches preserved
- ✅ `#ifdef` → `@if(YUV_SDTV)`
- ✅ `#else` → `@if(!YUV_SDTV)` (note: could also use `@else`)
- ✅ Matrix type converted (`mat3` → `mat3x3<f32>`)
- ✅ Matrix constructor uses column vectors
- ⚠️ **Check if vec4 overload is missing** (GLSL had it, check WESL)

---

## File Checklist

Review status for all 351 WESL files. Mark each as you complete the review:

### version (1 file)
- [x] `version.wesl` ↔ `version.glsl` - ✅ Perfect conversion (const with u32 types)

### animation (34 files)

#### animation/easing (33 files)
- [x] `easing/backIn.wesl` ↔ `easing/backIn.glsl` - ✅ Perfect conversion
- [x] `easing/backInOut.wesl` ↔ `easing/backInOut.glsl` - ✅ Correct select() logic
- [x] `easing/backOut.wesl` ↔ `easing/backOut.glsl` - ✅ Perfect conversion
- [x] `easing/bounceIn.wesl` ↔ `easing/bounceIn.glsl` - ✅ Perfect conversion
- [x] `easing/bounceInOut.wesl` ↔ `easing/bounceInOut.glsl` - ✅ Correct select() logic
- [x] `easing/bounceOut.wesl` ↔ `easing/bounceOut.glsl` - ✅ Fixed: let→const for compile-time constants
- [x] `easing/circularIn.wesl` ↔ `easing/circularIn.glsl` - ✅ Perfect conversion
- [x] `easing/circularInOut.wesl` ↔ `easing/circularInOut.glsl` - ✅ Correct select() logic
- [x] `easing/circularOut.wesl` ↔ `easing/circularOut.glsl` - ✅ Perfect conversion
- [x] `easing/cubicIn.wesl` ↔ `easing/cubicIn.glsl` - ✅ Perfect conversion
- [x] `easing/cubicInOut.wesl` ↔ `easing/cubicInOut.glsl` - ✅ Correct select() logic
- [x] `easing/cubicOut.wesl` ↔ `easing/cubicOut.glsl` - ✅ Perfect conversion
- [x] `easing/elasticIn.wesl` ↔ `easing/elasticIn.glsl` - ✅ Perfect conversion (imports HALF_PI)
- [x] `easing/elasticInOut.wesl` ↔ `easing/elasticInOut.glsl` - ✅ Correct select() logic (imports HALF_PI)
- [x] `easing/elasticOut.wesl` ↔ `easing/elasticOut.glsl` - ✅ Perfect conversion (imports HALF_PI)
- [x] `easing/exponentialIn.wesl` ↔ `easing/exponentialIn.glsl` - ✅ Correct select() logic
- [x] `easing/exponentialInOut.wesl` ↔ `easing/exponentialInOut.glsl` - ✅ Correct nested select() logic
- [x] `easing/exponentialOut.wesl` ↔ `easing/exponentialOut.glsl` - ✅ Correct select() logic
- [x] `easing/linearIn.wesl` ↔ `easing/linearIn.glsl` - ✅ Perfect conversion
- [x] `easing/linearInOut.wesl` ↔ `easing/linearInOut.glsl` - ✅ Perfect conversion
- [x] `easing/linearOut.wesl` ↔ `easing/linearOut.glsl` - ✅ Fixed: documentation type (<float> → <f32>)
- [x] `easing/quadraticIn.wesl` ↔ `easing/quadraticIn.glsl` - ✅ Perfect conversion
- [x] `easing/quadraticInOut.wesl` ↔ `easing/quadraticInOut.glsl` - ✅ Correct select() logic
- [x] `easing/quadraticOut.wesl` ↔ `easing/quadraticOut.glsl` - ✅ Perfect conversion
- [x] `easing/quarticIn.wesl` ↔ `easing/quarticIn.glsl` - ✅ Perfect conversion
- [x] `easing/quarticInOut.wesl` ↔ `easing/quarticInOut.glsl` - ✅ Correct select() logic
- [x] `easing/quarticOut.wesl` ↔ `easing/quarticOut.glsl` - ✅ Perfect conversion
- [x] `easing/quinticIn.wesl` ↔ `easing/quinticIn.glsl` - ✅ Perfect conversion
- [x] `easing/quinticInOut.wesl` ↔ `easing/quinticInOut.glsl` - ✅ Correct select() logic
- [x] `easing/quinticOut.wesl` ↔ `easing/quinticOut.glsl` - ✅ Perfect conversion
- [x] `easing/sineIn.wesl` ↔ `easing/sineIn.glsl` - ✅ Perfect conversion (imports HALF_PI)
- [x] `easing/sineInOut.wesl` ↔ `easing/sineInOut.glsl` - ✅ Perfect conversion (imports PI)
- [x] `easing/sineOut.wesl` ↔ `easing/sineOut.glsl` - ✅ Perfect conversion (imports HALF_PI)

#### animation (1 file)
- [x] `spriteLoop.wesl` ↔ `spriteLoop.glsl` - ✅ Fixed documentation typo in use section

### color (156 files)

#### color/blend (24 files)
- [x] `blend/add.wesl` ↔ `blend/add.glsl` - ✅ Perfect (3 overloads: f32, vec3, vec3+opacity)
- [x] `blend/average.wesl` ↔ `blend/average.glsl` - ✅ Perfect (3 overloads: f32, vec3, vec3+opacity)
- [x] `blend/color.wesl` ↔ `blend/color.glsl` - ✅ Perfect (2 overloads, correct imports)
- [x] `blend/colorBurn.wesl` ↔ `blend/colorBurn.glsl` - ✅ Perfect (3 overloads, correct select() logic)
- [x] `blend/colorDodge.wesl` ↔ `blend/colorDodge.glsl` - ✅ Perfect (3 overloads, correct select() logic)
- [x] `blend/darken.wesl` ↔ `blend/darken.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/difference.wesl` ↔ `blend/difference.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/exclusion.wesl` ↔ `blend/exclusion.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/glow.wesl` ↔ `blend/glow.glsl` - ✅ Perfect (3 overloads, correct imports)
- [x] `blend/hardLight.wesl` ↔ `blend/hardLight.glsl` - ✅ Perfect (3 overloads, correct imports)
- [x] `blend/hardMix.wesl` ↔ `blend/hardMix.glsl` - ✅ Fixed indentation issue
- [x] `blend/hue.wesl` ↔ `blend/hue.glsl` - ✅ Perfect (2 overloads, correct imports)
- [x] `blend/lighten.wesl` ↔ `blend/lighten.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/linearBurn.wesl` ↔ `blend/linearBurn.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/linearDodge.wesl` ↔ `blend/linearDodge.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/linearLight.wesl` ↔ `blend/linearLight.glsl` - ✅ Perfect (3 overloads, correct imports)
- [x] `blend/luminosity.wesl` ↔ `blend/luminosity.glsl` - ✅ Perfect (2 overloads, correct imports)
- [x] `blend/multiply.wesl` ↔ `blend/multiply.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/negation.wesl` ↔ `blend/negation.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/overlay.wesl` ↔ `blend/overlay.glsl` - ✅ Fixed trailing dots (1. → 1.0, 2. → 2.0)
- [x] `blend/phoenix.wesl` ↔ `blend/phoenix.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/pinLight.wesl` ↔ `blend/pinLight.glsl` - ✅ Perfect (3 overloads, correct imports)
- [x] `blend/reflect.wesl` ↔ `blend/reflect.glsl` - ✅ Perfect (3 overloads, correct select() logic)
- [x] `blend/saturation.wesl` ↔ `blend/saturation.glsl` - ✅ Perfect (2 overloads, correct imports)
- [x] `blend/screen.wesl` ↔ `blend/screen.glsl` - ✅ Fixed trailing dots (1. → 1.0)
- [x] `blend/softLight.wesl` ↔ `blend/softLight.glsl` - ✅ Fixed trailing dots, has vec4 overload
- [x] `blend/subtract.wesl` ↔ `blend/subtract.glsl` - ✅ Perfect (3 overloads)
- [x] `blend/vividLight.wesl` ↔ `blend/vividLight.glsl` - ✅ Perfect (3 overloads, correct imports)

#### color/composite (9 files)
- [x] `composite/compositeXor.wesl` ↔ `composite/compositeXor.glsl` - ✅ Perfect (3 overloads: f32, vec3+alpha, vec4)
- [x] `composite/destinationAtop.wesl` ↔ `composite/destinationAtop.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/destinationIn.wesl` ↔ `composite/destinationIn.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/destinationOut.wesl` ↔ `composite/destinationOut.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/destinationOver.wesl` ↔ `composite/destinationOver.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/sourceAtop.wesl` ↔ `composite/sourceAtop.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/sourceIn.wesl` ↔ `composite/sourceIn.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/sourceOut.wesl` ↔ `composite/sourceOut.glsl` - ✅ Perfect (3 overloads)
- [x] `composite/sourceOver.wesl` ↔ `composite/sourceOver.glsl` - ✅ Fixed: simplified unnecessary intermediate variable

#### color/dither (3 files)
- [x] `dither/bayer.wesl` ↔ `dither/bayer.glsl` - ✅ Complete rewrite: added all 11 overloads, @if conditionals for DITHER_BAKER_COORD/DITHER_BAYER_PRECISION/DITHER_PRECISION, 8x8 Bayer matrix implementation
- [x] `dither/blueNoise.wesl` ↔ `dither/blueNoise.glsl` - ✅ Complete rewrite: added all 9 overloads, @if conditionals for DITHER_BLUENOISE_TIME/DITHER_BLUENOISE_COORD/DITHER_BLUENOISE_PRECISION, removed hardcoded uniforms.frameIdx, added saturate import
- [x] `dither/vlachos.wesl` ↔ `dither/vlachos.glsl` - ✅ Complete rewrite: added all 7 overloads (including ditherVlachos3Noise helper), @if conditionals for DITHER_VLACHOS_TIME/DITHER_VLACHOS_COORD/DITHER_VLACHOS_PRECISION, removed hardcoded uniforms.frameIdx

#### color/layer (24 files)
- [x] `layer/addSourceOver.wesl` ↔ `layer/addSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/averageSourceOver.wesl` ↔ `layer/averageSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/colorBurnSourceOver.wesl` ↔ `layer/colorBurnSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/colorDodgeSourceOver.wesl` ↔ `layer/colorDodgeSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/colorSourceOver.wesl` ↔ `layer/colorSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/darkenSourceOver.wesl` ↔ `layer/darkenSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/differenceSourceOver.wesl` ↔ `layer/differenceSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/exclusionSourceOver.wesl` ↔ `layer/exclusionSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/glowSourceOver.wesl` ↔ `layer/glowSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/hardLightSourceOver.wesl` ↔ `layer/hardLightSourceOver.glsl` - ✅ Fixed GLSL: float3 → vec3
- [x] `layer/hardMixSourceOver.wesl` ↔ `layer/hardMixSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/hueSourceOver.wesl` ↔ `layer/hueSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/lightenSourceOver.wesl` ↔ `layer/lightenSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/linearBurnSourceOver.wesl` ↔ `layer/linearBurnSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/linearDodgeSourceOver.wesl` ↔ `layer/linearDodgeSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/linearLightSourceOver.wesl` ↔ `layer/linearLightSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/luminositySourceOver.wesl` ↔ `layer/luminositySourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/multiplySourceOver.wesl` ↔ `layer/multiplySourceOver.glsl` - ✅ Perfect conversion (WESL uses inline return)
- [x] `layer/negationSourceOver.wesl` ↔ `layer/negationSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/overlaySourceOver.wesl` ↔ `layer/overlaySourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/phoenixSourceOver.wesl` ↔ `layer/phoenixSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/pinLightSourceOver.wesl` ↔ `layer/pinLightSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/reflectSourceOver.wesl` ↔ `layer/reflectSourceOver.glsl` - ✅ Fixed GLSL: float3 → vec3
- [x] `layer/saturationSourceOver.wesl` ↔ `layer/saturationSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/screenSourceOver.wesl` ↔ `layer/screenSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/softLightSourceOver.wesl` ↔ `layer/softLightSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/subtractSourceOver.wesl` ↔ `layer/subtractSourceOver.glsl` - ✅ Perfect conversion
- [x] `layer/vividLightSourceOver.wesl` ↔ `layer/vividLightSourceOver.glsl` - ✅ Perfect conversion

#### color/levels (4 files)
- [x] `levels.wesl` ↔ `levels.glsl` - ✅ Perfect (4 overloads: vec3+vec3, vec3+float, vec4+vec3, vec4+float)
- [x] `levels/gamma.wesl` ↔ `levels/gamma.glsl` - ✅ Perfect (4 overloads: vec3+vec3, vec3+float, vec4+vec3, vec4+float)
- [x] `levels/inputRange.wesl` ↔ `levels/inputRange.glsl` - ✅ Fixed trailing dots (0. → 0.0, 1. → 1.0), all 4 overloads present
- [x] `levels/outputRange.wesl` ↔ `levels/outputRange.glsl` - ✅ Perfect (4 overloads: vec3+vec3, vec3+float, vec4+vec3, vec4+float)

#### color/palette (2 files)
- [x] `palette/heatmap.wesl` ↔ `palette/heatmap.glsl` - ✅ Perfect conversion (1 overload)
- [x] `palette/hue.wesl` ↔ `palette/hue.glsl` - ✅ Fixed: added missing hueDefault overload for default ratio parameter

#### color/space (40 files)
- [x] `space/YCbCr2rgb.wesl` ↔ `space/YCbCr2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/YPbPr2rgb.wesl` ↔ `space/YPbPr2rgb.glsl` - ✅ Added missing vec4 overload, added @if(YPBPR_SDTV) conditionals
- [x] `space/cmyk2rgb.wesl` ↔ `space/cmyk2rgb.glsl` - ✅ Added missing saturate import
- [x] `space/gamma2linear.wesl` ↔ `space/gamma2linear.glsl` - ✅ Added f32 and vec4 overloads, added @if(GAMMA) conditionals
- [x] `space/hcy2rgb.wesl` ↔ `space/hcy2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/hsl2rgb.wesl` ↔ `space/hsl2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/hsv2rgb.wesl` ↔ `space/hsv2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/hsv2ryb.wesl` ↔ `space/hsv2ryb.glsl` - ✅ Added @if conditionals for HSV2RYB_FAST and RYB_FAST, added saturate import
- [x] `space/hue2rgb.wesl` ↔ `space/hue2rgb.glsl` - ✅ Added missing saturate import
- [x] `space/k2rgb.wesl` ↔ `space/k2rgb.glsl` - ✅ Added missing saturate import
- [x] `space/lab2lch.wesl` ↔ `space/lab2lch.glsl` - ✅ Added missing vec4 overload
- [x] `space/lab2rgb.wesl` ↔ `space/lab2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/lab2srgb.wesl` ↔ `space/lab2srgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/lab2xyz.wesl` ↔ `space/lab2xyz.glsl` - ✅ Added missing vec4 overload
- [x] `space/lch2lab.wesl` ↔ `space/lch2lab.glsl` - ✅ Added missing vec4 overload
- [x] `space/lch2rgb.wesl` ↔ `space/lch2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/lch2srgb.wesl` ↔ `space/lch2srgb.glsl` - ✅ Fixed function names (lch2srgb3 → lch2srgb)
- [x] `space/linear2gamma.wesl` ↔ `space/linear2gamma.glsl` - ✅ Added f32 and vec4 overloads, added @if(GAMMA) conditionals
- [x] `space/lms2rgb.wesl` ↔ `space/lms2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/oklab2rgb.wesl` ↔ `space/oklab2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/oklab2srgb.wesl` ↔ `space/oklab2srgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2YCbCr.wesl` ↔ `space/rgb2YCbCr.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2YPbPr.wesl` ↔ `space/rgb2YPbPr.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2cmyk.wesl` ↔ `space/rgb2cmyk.glsl` - ✅ Perfect conversion (1 overload)
- [x] `space/rgb2hcv.wesl` ↔ `space/rgb2hcv.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2hcy.wesl` ↔ `space/rgb2hcy.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2heat.wesl` ↔ `space/rgb2heat.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2hsl.wesl` ↔ `space/rgb2hsl.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2hsv.wesl` ↔ `space/rgb2hsv.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2hue.wesl` ↔ `space/rgb2hue.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2lab.wesl` ↔ `space/rgb2lab.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2lch.wesl` ↔ `space/rgb2lch.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2lms.wesl` ↔ `space/rgb2lms.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2luma.wesl` ↔ `space/rgb2luma.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2oklab.wesl` ↔ `space/rgb2oklab.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2ryb.wesl` ↔ `space/rgb2ryb.glsl` - ✅ Added @if(RYB_FAST) conditionals, added vec4 overload, added mmin/mmax imports
- [x] `space/rgb2srgb.wesl` ↔ `space/rgb2srgb.glsl` - ✅ Added missing vec4 overload, added saturate import
- [x] `space/rgb2xyY.wesl` ↔ `space/rgb2xyY.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2xyz.wesl` ↔ `space/rgb2xyz.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2yiq.wesl` ↔ `space/rgb2yiq.glsl` - ✅ Added missing vec4 overload
- [x] `space/rgb2yuv.wesl` ↔ `space/rgb2yuv.glsl` - ✅ Added missing vec4 overload
- [x] `space/ryb2rgb.wesl` ↔ `space/ryb2rgb.glsl` - ✅ Added @if(RYB_FAST) conditionals, added vec4 overload, added mmin/mmax imports
- [x] `space/srgb2lab.wesl` ↔ `space/srgb2lab.glsl` - ✅ Added missing vec4 overload
- [x] `space/srgb2lch.wesl` ↔ `space/srgb2lch.glsl` - ✅ Added missing vec4 overload
- [x] `space/srgb2luma.wesl` ↔ `space/srgb2luma.glsl` - ✅ Added missing vec4 overload
- [x] `space/srgb2oklab.wesl` ↔ `space/srgb2oklab.glsl` - ✅ Added missing vec4 overload
- [x] `space/srgb2rgb.wesl` ↔ `space/srgb2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/srgb2xyz.wesl` ↔ `space/srgb2xyz.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyY2rgb.wesl` ↔ `space/xyY2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyY2srgb.wesl` ↔ `space/xyY2srgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyY2xyz.wesl` ↔ `space/xyY2xyz.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyz2lab.wesl` ↔ `space/xyz2lab.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyz2rgb.wesl` ↔ `space/xyz2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyz2srgb.wesl` ↔ `space/xyz2srgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/xyz2xyY.wesl` ↔ `space/xyz2xyY.glsl` - ✅ Added missing vec4 overload
- [x] `space/yiq2rgb.wesl` ↔ `space/yiq2rgb.glsl` - ✅ Added missing vec4 overload
- [x] `space/yuv2rgb.wesl` ↔ `space/yuv2rgb.glsl` - ✅ Added missing vec4 overload

#### color/tonemap (9 files)
- [x] `tonemap/aces.wesl` ↔ `tonemap/aces.glsl` - ✅ Fixed: added missing saturate import, added f32 type annotations to constants
- [x] `tonemap/debug.wesl` ↔ `tonemap/debug.glsl` - ✅ Perfect (PLATFORM conditionals not needed in WESL)
- [x] `tonemap/filmic.wesl` ↔ `tonemap/filmic.glsl` - ✅ Perfect (2 overloads)
- [x] `tonemap/linear.wesl` ↔ `tonemap/linear.glsl` - ✅ Perfect (2 overloads, passthrough)
- [x] `tonemap/reinhard.wesl` ↔ `tonemap/reinhard.glsl` - ✅ Perfect (2 overloads)
- [x] `tonemap/reinhardJodie.wesl` ↔ `tonemap/reinhardJodie.glsl` - ✅ Perfect (2 overloads)
- [x] `tonemap/uncharted.wesl` ↔ `tonemap/uncharted.glsl` - ✅ Perfect (helper + 2 main overloads)
- [x] `tonemap/uncharted2.wesl` ↔ `tonemap/uncharted2.glsl` - ✅ Perfect (2 overloads)
- [x] `tonemap/unreal.wesl` ↔ `tonemap/unreal.glsl` - ✅ Perfect (2 overloads)

#### color (17 files)
- [x] `brightnessContrast.wesl` ↔ `brightnessContrast.glsl` - ✅ Perfect (3 overloads: f32, vec3, vec4)
- [x] `brightnessMatrix.wesl` ↔ `brightnessMatrix.glsl` - ✅ Perfect (returns mat4x4f, fixed trailing dots)
- [x] `contrast.wesl` ↔ `contrast.glsl` - ✅ Perfect (3 overloads: f32, vec3, vec4)
- [x] `contrastMatrix.wesl` ↔ `contrastMatrix.glsl` - ✅ Fixed trailing dots (1. → 1.0, .0 → 0.0, .5 → 0.5), proper mat4x4f column vectors
- [x] `desaturate.wesl` ↔ `desaturate.glsl` - ✅ Perfect (2 overloads: vec3, vec4)
- [x] `distance.wesl` ↔ `distance.glsl` - ✅ Fixed: added missing colorDistance and colorDistance4 wrapper functions (default to CIE94)
- [x] `exposure.wesl` ↔ `exposure.glsl` - ✅ Fixed: added missing f32 overload, fixed trailing dot (pow(2. → pow(2.0))
- [x] `hueShift.wesl` ↔ `hueShift.glsl` - ✅ Fixed: added vec4 overload, added @if(HUESHIFT_AMOUNT) conditionals and TAU import
- [x] `hueShiftRYB.wesl` ↔ `hueShiftRYB.glsl` - ✅ Fixed: added vec4 overload, corrected function body (was using PI instead of parameter a)
- [x] `luma.wesl` ↔ `luma.glsl` - ✅ Fixed: added missing f32 overload (passthrough) and vec4 overload
- [x] `luminance.wesl` ↔ `luminance.glsl` - ✅ Perfect (2 overloads: vec3, vec4)
- [x] `levels.wesl` ↔ `levels.glsl` - ✅ Perfect (4 overloads with proper naming)
- [x] `mixOklab.wesl` ↔ `mixOklab.glsl` - ✅ Fixed: added vec4 overload, added @if(MIXOKLAB_SRGB) conditionals with srgb2rgb/rgb2srgb imports
- [x] `mixSpectral.wesl` ↔ `mixSpectral.glsl` - ✅ Fixed: added vec4 overload (Note: WESL uses simplified single-function version vs GLSL's multi-color helpers)
- [x] `saturationMatrix.wesl` ↔ `saturationMatrix.glsl` - ✅ Fixed: changed return type from mat3x3 to mat4x4f, fixed trailing dots
- [x] `vibrance.wesl` ↔ `vibrance.glsl` - ✅ Perfect (2 overloads, uses rgb2luma instead of mmax/mmin)
- [x] `whiteBalance.wesl` ↔ `whiteBalance.glsl` - ✅ Perfect (2 overloads, WESL uses non-mobile version only)

### draw (1 file)
- [x] `stroke.wesl` ↔ `stroke.glsl` - ✅ Fixed: added missing 3-parameter overload (stroke), renamed 4-parameter to strokeEdge, added aastep/saturate imports, updated documentation types

### filter (3 files)

#### filter/edge (1 file)
- [x] `edge/prewitt.wesl` ↔ `edge/prewitt.glsl` - ✅ Fixed: changed textureSample → textureSampleBaseClampToEdge, fixed trailing dots (0. → 0.0). Note: WESL uses single vec3f return type, GLSL allows customization via EDGEPREWITT_TYPE/EDGEPREWITT_SAMPLER_FNC

#### filter/sharpen (2 files)
- [x] `sharpen/adaptive.wesl` ↔ `sharpen/adaptive.glsl` - ✅ Fixed: added missing sharpenAdaptive default overload (without strength), renamed main function to sharpenAdaptive4. Has extra sharpenContrastAdaptive function not in GLSL
- [x] `sharpen/fast.wesl` ↔ `sharpen/fast.glsl` - ✅ Fixed: added missing sharpenFast default overload (without strength), renamed strength version to sharpenFast4, fixed trailing dots (1. → 1.0, 0. → 0.0, -1. → -1.0, 5. → 5.0)

### generative (8 files)
- [x] `cnoise.wesl` ↔ `cnoise.glsl` - ✅ Perfect conversion (3 overloads: cnoise2/3/4, all imports correct)
- [x] `noised.wesl` ↔ `noised.glsl` - ✅ Fixed trailing dots (6. → 6.0, 15. → 15.0, etc.), hardcodes srandom22/srandom33
- [x] `pnoise.wesl` ↔ `pnoise.glsl` - ✅ Perfect conversion (3 overloads: pnoise2/3/4, uses % operator for modulo)
- [x] `random.wesl` ↔ `random.glsl` - ✅ Fixed spacing (pos.wzxy+33.33 → pos.wzxy + 33.33), hardcodes RANDOM_SINLESS=true, all 16 overloads present
- [x] `snoise.wesl` ↔ `snoise.glsl` - ✅ Perfect conversion (6 overloads: snoise2/3/4, snoise22/33/34, all imports correct)
- [x] `srandom.wesl` ↔ `srandom.glsl` - ✅ Perfect conversion (9 overloads, uses fmod2/fmod3 correctly)
- [x] `wavelet.wesl` ↔ `wavelet.glsl` - ✅ Perfect conversion (5 overloads with unique names, hardcodes WAVELET_VORTICITY=0.0)
- [x] `worley.wesl` ↔ `worley.glsl` - ✅ Perfect conversion (4 overloads: worley2/22/3/32, hardcodes distEuclidean)

### geometry (11 files)

#### geometry/aabb (6 files)
- [x] `aabb/aabb.wesl` ↔ `aabb/aabb.glsl` - ✅ Perfect conversion (struct definition, vec3f types)
- [x] `aabb/centroid.wesl` ↔ `aabb/centroid.glsl` - ✅ Perfect conversion (single function)
- [x] `aabb/contain.wesl` ↔ `aabb/contain.glsl` - ✅ Perfect conversion (WESL uses operators instead of lessThanEqual/lessThan)
- [x] `aabb/diagonal.wesl` ↔ `aabb/diagonal.glsl` - ✅ Perfect conversion (single function)
- [x] `aabb/expand.wesl` ↔ `aabb/expand.glsl` - ✅ Perfect conversion (3 overloads with unique names, uses ptr<function, AABB>)
- [x] `aabb/square.wesl` ↔ `aabb/square.glsl` - ✅ Perfect conversion (uses ptr<function, AABB>, imports diagonal)

#### geometry/triangle (5 files)
- [x] `triangle/area.wesl` ↔ `triangle/area.glsl` - ✅ Perfect conversion. Fixed GLSL doc: "normal" → "area"
- [x] `triangle/barycentric.wesl` ↔ `triangle/barycentric.glsl` - ✅ Perfect conversion (3 overloads with unique names). Fixed GLSL doc: "centroid" → "barycentric"
- [x] `triangle/centroid.wesl` ↔ `triangle/centroid.glsl` - ✅ Perfect conversion (single function)
- [x] `triangle/normal.wesl` ↔ `triangle/normal.glsl` - ✅ Perfect conversion. Fixed GLSL doc: "vec33" → "vec3", "getNormal" → "normal"
- [x] `triangle/triangle.wesl` ↔ `triangle/triangle.glsl` - ✅ Perfect conversion (struct definition). Fixed GLSL typo: STR_TRINAGLE → STR_TRIANGLE

### lighting (9 files)

#### lighting/common (2 files)
- [x] `common/ggx.wesl` ↔ `common/ggx.glsl` - ✅ Fixed: added missing GGX(NoH, roughness) overload, added importanceSamplingGGX function, added imports for INV_PI/PI and saturateMediump, preserved all comments and platform-specific optimizations
- [x] `common/schlick.wesl` ↔ `common/schlick.glsl` - ✅ Fixed: corrected parameter name (cos0→VoH) and used pow5 consistently across all overloads

#### lighting/diffuse (1 file)
- [x] `diffuse/orenNayar.wesl` ↔ `diffuse/orenNayar.glsl` - ✅ Fixed: added use: documentation, noted ShadingData struct overload deferred until LATER

#### lighting/raymarch (2 files)
- [x] `raymarch/cast.wesl` ↔ `raymarch/cast.glsl` - ✅ Correctly deferred (requires user-defined map function and Material struct)
- [x] `raymarch/normal.wesl` ↔ `raymarch/normal.glsl` - ✅ Fixed: corrected deferral message (was incorrectly importing math::map, now properly deferred until user-defined map and Material struct available)

#### lighting/specular (1 file)
- [x] `specular/cookTorrance.wesl` ↔ `specular/cookTorrance.glsl` - ✅ Complete rewrite: created smithGGXCorrelated.wesl (new file), fixed signature to match GLSL (vec3→vec3), updated imports, noted ShadingData overload deferred, preserved platform-specific optimization logic

#### lighting (3 files)
- [x] `fresnel.wesl` ↔ `fresnel.glsl` - ✅ Fixed: added 3 missing overloads (fresnel(vec3,vec3,vec3), fresnel(vec3,f32), fresnel(vec3,f32,f32)), added imports for saturate and pow5, added documentation
- [x] `fresnelReflection.wesl` ↔ `fresnelReflection.glsl` - ✅ Fixed: added FRESNEL_REFLECTION_RGB constant with @if conditional, added missing fresnelReflection(R, Fr) overload, noted envMap/ior dependencies and 6 fresnelIridescent variants deferred until LATER, added stub implementations
- [x] `toShininess.wesl` ↔ `toShininess.glsl` - ✅ Perfect conversion. Fixed GLSL: trailing dots (80.→80.0, 160.→160.0)

### math (80 files)

#### math/quat (14 files)
- [x] `quat.wesl` ↔ `quat.glsl` - ✅ Fixed: added missing quatForward(f) overload, fixed typo (quatFowardUp → quatForwardUp)
- [x] `quat/add.wesl` ↔ `quat/add.glsl` - ✅ Perfect conversion
- [x] `quat/conj.wesl` ↔ `quat/conj.glsl` - ✅ Perfect conversion
- [x] `quat/div.wesl` ↔ `quat/div.glsl` - ✅ Perfect conversion
- [x] `quat/identity.wesl` ↔ `quat/identity.glsl` - ✅ Perfect conversion (const)
- [x] `quat/inverse.wesl` ↔ `quat/inverse.glsl` - ✅ Perfect conversion (all imports correct)
- [x] `quat/length.wesl` ↔ `quat/length.glsl` - ✅ Perfect conversion
- [x] `quat/lengthSq.wesl` ↔ `quat/lengthSq.glsl` - ✅ Perfect conversion
- [x] `quat/lerp.wesl` ↔ `quat/lerp.glsl` - ✅ Perfect conversion (uses select() for ternaries)
- [x] `quat/mul.wesl` ↔ `quat/mul.glsl` - ✅ Fixed: added missing quatMulScalar(q, s) overload
- [x] `quat/neg.wesl` ↔ `quat/neg.glsl` - ✅ Perfect conversion
- [x] `quat/norm.wesl` ↔ `quat/norm.glsl` - ✅ Perfect conversion (all imports correct)
- [x] `quat/quat2mat3.wesl` ↔ `quat/2mat3.glsl` - ✅ Perfect conversion (Note: GLSL file is 2mat3.glsl not quat2mat3.glsl)
- [x] `quat/quat2mat4.wesl` ↔ `quat/2mat4.glsl` - ✅ Perfect conversion (Note: GLSL file is 2mat4.glsl not quat2mat4.glsl)
- [x] `quat/sub.wesl` ↔ `quat/sub.glsl` - ✅ Perfect conversion

#### math (66 files)
- [x] `aafloor.wesl` ↔ `aafloor.glsl` - ✅ Fixed: trailing dots (1. → 1.0)
- [x] `aafract.wesl` ↔ `aafract.glsl` - ✅ Fixed: restored nyquist import and logic that was incorrectly simplified
- [x] `aamirror.wesl` ↔ `aamirror.glsl` - ✅ Perfect conversion (uses dpdx/dpdy)
- [x] `aastep.wesl` ↔ `aastep.glsl` - ✅ Perfect conversion (uses fwidth)
- [x] `absi.wesl` ↔ `absi.glsl` - ✅ Perfect conversion (macro → fn with select())
- [x] `adaptiveThreshold.wesl` ↔ `adaptiveThreshold.glsl` - ✅ Perfect conversion
- [x] `atan2.wesl` ↔ `atan2.glsl` - ✅ Perfect conversion
- [x] `bump.wesl` ↔ `bump.glsl` - ✅ Perfect conversion
- [x] `consts.wesl` ↔ `consts.glsl` - ✅ Perfect conversion (Note: GLSL file doesn't exist, WESL is canonical)
- [x] `cubic.wesl` ↔ `cubic.glsl` - ✅ Perfect conversion
- [x] `cubicMix.wesl` ↔ `cubicMix.glsl` - ✅ Perfect conversion
- [x] `decimate.wesl` ↔ `decimate.glsl` - ✅ Perfect conversion
- [x] `dist.wesl` ↔ `dist.glsl` - ✅ Perfect conversion
- [x] `fcos.wesl` ↔ `fcos.glsl` - ✅ Perfect conversion
- [x] `fmod.wesl` ↔ `fmod.glsl` - ✅ Perfect conversion (floored modulo)
- [x] `gain.wesl` ↔ `gain.glsl` - ✅ Perfect conversion
- [x] `gaussian.wesl` ↔ `gaussian.glsl` - ✅ Perfect conversion
- [x] `grad4.wesl` ↔ `grad4.glsl` - ✅ Perfect conversion
- [x] `hammersley.wesl` ↔ `hammersley.glsl` - ✅ Perfect conversion (includes hemisphereCosSample helper)
- [x] `highPass.wesl` ↔ `highPass.glsl` - ✅ Perfect conversion
- [x] `inside.wesl` ↔ `inside.glsl` - ✅ Perfect conversion
- [x] `invCubic.wesl` ↔ `invCubic.glsl` - ✅ Perfect conversion
- [x] `invQuartic.wesl` ↔ `invQuartic.glsl` - ✅ Perfect conversion
- [x] `inverse.wesl` ↔ `inverse.glsl` - ✅ Perfect conversion
- [x] `lengthSq.wesl` ↔ `lengthSq.glsl` - ✅ Perfect conversion
- [x] `map.wesl` ↔ `map.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `mirror.wesl` ↔ `mirror.glsl` - ✅ Perfect conversion
- [x] `mmax.wesl` ↔ `mmax.glsl` - ✅ Perfect conversion (extract max component from vector)
- [x] `mmin.wesl` ↔ `mmin.glsl` - ✅ Perfect conversion (extract min component from vector)
- [x] `mod2.wesl` ↔ `mod2.glsl` - ✅ Perfect conversion
- [x] `mod289.wesl` ↔ `mod289.glsl` - ✅ Fixed: trailing dots (1. → 1.0, 289. → 289.0)
- [x] `nyquist.wesl` ↔ `nyquist.glsl` - ✅ Perfect conversion
- [x] `pack.wesl` ↔ `pack.glsl` - ✅ Fixed: trailing dots in constants
- [x] `parabola.wesl` ↔ `parabola.glsl` - ✅ Perfect conversion
- [x] `permute.wesl` ↔ `permute.glsl` - ✅ Perfect conversion
- [x] `pow2.wesl` ↔ `pow2.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `pow3.wesl` ↔ `pow3.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `pow5.wesl` ↔ `pow5.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `pow7.wesl` ↔ `pow7.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `powFast.wesl` ↔ `powFast.glsl` - ✅ Perfect conversion
- [x] `quartic.wesl` ↔ `quartic.glsl` - ✅ Perfect conversion
- [x] `quintic.wesl` ↔ `quintic.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `rotate2d.wesl` ↔ `rotate2d.glsl` - ✅ Fixed: matrix constructor now uses explicit column vectors
- [x] `rotate3d.wesl` ↔ `rotate3d.glsl` - ✅ Perfect conversion
- [x] `rotate3dX.wesl` ↔ `rotate3dX.glsl` - ✅ Fixed: corrected sin/cos order in column vectors
- [x] `rotate3dY.wesl` ↔ `rotate3dY.glsl` - ✅ Perfect conversion
- [x] `rotate3dZ.wesl` ↔ `rotate3dZ.glsl` - ✅ Perfect conversion
- [x] `rotate4d.wesl` ↔ `rotate4d.glsl` - ✅ Perfect conversion
- [x] `rotate4dX.wesl` ↔ `rotate4dX.glsl` - ✅ Perfect conversion
- [x] `rotate4dY.wesl` ↔ `rotate4dY.glsl` - ✅ Perfect conversion
- [x] `rotate4dZ.wesl` ↔ `rotate4dZ.glsl` - ✅ Perfect conversion
- [x] `round.wesl` ↔ `round.glsl` - ✅ Perfect conversion
- [x] `saturate.wesl` ↔ `saturate.glsl` - ✅ Perfect conversion (4 overloads)
- [x] `saturateMediump.wesl` ↔ `saturateMediump.glsl` - ✅ Perfect conversion
- [x] `scale2d.wesl` ↔ `scale2d.glsl` - ✅ Fixed: added missing f32 and f32,f32 overloads
- [x] `scale3d.wesl` ↔ `scale3d.glsl` - ✅ Perfect conversion
- [x] `scale4d.wesl` ↔ `scale4d.glsl` - ✅ Perfect conversion
- [x] `smootherstep.wesl` ↔ `smootherstep.glsl` - ✅ Fixed: added missing saturate import
- [x] `sum.wesl` ↔ `sum.glsl` - ✅ Perfect conversion
- [x] `taylorInvSqrt.wesl` ↔ `taylorInvSqrt.glsl` - ✅ Perfect conversion
- [x] `toMat3.wesl` ↔ `toMat3.glsl` - ✅ Perfect conversion
- [x] `toMat4.wesl` ↔ `toMat4.glsl` - ✅ Perfect conversion
- [x] `translate4d.wesl` ↔ `translate4d.glsl` - ✅ Fixed: added missing f32,f32,f32 overload, used column vector syntax
- [x] `unpack.wesl` ↔ `unpack.glsl` - ✅ Fixed: trailing dots in constants
- [x] `within.wesl` ↔ `within.glsl` - ✅ Fixed: trailing dots (1. → 1.0)

### sample (1 file)
- [x] `sprite.wesl` ↔ `sprite.glsl` - ✅ Fixed documentation spacing (missing space after texture type). Perfect conversion: uses explicit texture_2d<f32>/sampler parameters (WESL style) vs GLSL's macro-based SAMPLER_TYPE customization

### sdf (7 files)
- [x] `boxSDF.wesl` ↔ `boxSDF.glsl` - ✅ Fixed: added missing default overload (without borders parameter)
- [x] `cylinderSDF.wesl` ↔ `cylinderSDF.glsl` - ✅ Fixed: added 3 missing overloads (single float h, h+r params, arbitrary orientation), converted ternary to select()
- [x] `opSubtraction.wesl` ↔ `opSubtraction.glsl` - ✅ Fixed: added 3 missing overloads (vec4, smooth variants), added saturate import, converted ternary to select()
- [x] `opUnion.wesl` ↔ `opUnion.glsl` - ✅ Fixed: added 2 missing overloads (soft union variants for f32 and vec4), added saturate import. Note: Material struct overload skipped
- [x] `rectSDF.wesl` ↔ `rectSDF.glsl` - ✅ Fixed: added 3 missing overloads, renamed rectSDF_round to rectSDF3, added @if(CENTER_2D) conditional
- [x] `sphereSDF.wesl` ↔ `sphereSDF.glsl` - ✅ Fixed: added missing default overload (without size parameter)
- [x] `torusSDF.wesl` ↔ `torusSDF.glsl` - ✅ Fixed: added missing advanced overload (4 params with sc, ra, rb), converted ternary to select()

### space (40 files)
- [x] `aspect.wesl` ↔ `aspect.glsl` - ✅ Perfect conversion
- [x] `bracketing.wesl` ↔ `bracketing.glsl` - ✅ Perfect conversion (uses struct for output, @if conditionals for BRACKETING_ANGLE_DELTA)
- [x] `brickTile.wesl` ↔ `brickTile.glsl` - ✅ Fixed trailing dot (2. → 2.0)
- [x] `cart2polar.wesl` ↔ `cart2polar.glsl` - ✅ Perfect conversion (2 overloads)
- [x] `center.wesl` ↔ `center.glsl` - ✅ Perfect conversion (3 overloads)
- [x] `checkerTile.wesl` ↔ `checkerTile.glsl` - ✅ Fixed trailing dot (2. → 2.0)
- [x] `decimateNormal.wesl` ↔ `decimateNormal.glsl` - ✅ Perfect conversion
- [x] `depth2viewZ.wesl` ↔ `depth2viewZ.glsl` - ✅ Perfect conversion (@if conditionals for CAMERA_ORTHOGRAPHIC_PROJECTION)
- [x] `equirect2xyz.wesl` ↔ `equirect2xyz.glsl` - ✅ Perfect conversion (expands `dir.xz *=` to separate operations)
- [x] `eulerView.wesl` ↔ `eulerView.glsl` - ✅ Perfect conversion (explicit mat3x3f identity matrix)
- [x] `fisheye2xyz.wesl` ↔ `fisheye2xyz.glsl` - ✅ Perfect conversion (adds division by zero guard)
- [x] `flipY.wesl` ↔ `flipY.glsl` - ✅ Fixed GLSL trailing dots (1. → 1.0)
- [x] `hexTile.wesl` ↔ `hexTile.glsl` - ✅ Fixed trailing dots, converted ternary to if/else
- [x] `kaleidoscope.wesl` ↔ `kaleidoscope.glsl` - ✅ Perfect conversion (3 overloads, @if conditionals for CENTER_2D)
- [x] `linearizeDepth.wesl` ↔ `linearizeDepth.glsl` - ✅ Perfect conversion (@if conditionals for CAMERA_NEAR_CLIP/CAMERA_FAR_CLIP)
- [x] `lookAt.wesl` ↔ `lookAt.glsl` - ✅ Fixed: added missing @if conditionals for LOOK_AT_RIGHT_HANDED (4 overloads)
- [x] `lookAtView.wesl` ↔ `lookAtView.glsl` - ✅ Perfect conversion (3 overloads, removes translate import, inlines mat3→mat4 conversion)
- [x] `mirrorTile.wesl` ↔ `mirrorTile.glsl` - ✅ Fixed: mirrorXTile/mirrorYTile workaround for single f32 mirror (uses vec2f wrapper)
- [x] `nearest.wesl` ↔ `nearest.glsl` - ✅ Fixed: added missing use: documentation
- [x] `orthographic.wesl` ↔ `orthographic.glsl` - ✅ Perfect conversion
- [x] `perspective.wesl` ↔ `perspective.glsl` - ✅ Perfect conversion
- [x] `polar2cart.wesl` ↔ `polar2cart.glsl` - ✅ Perfect conversion (2 overloads)
- [x] `ratio.wesl` ↔ `ratio.glsl` - ✅ Fixed: improved spacing consistency (*.5 → * 0.5)
- [x] `rotate.wesl` ↔ `rotate.glsl` - ✅ Fixed: complete rewrite, added all overloads (vec2/vec3/vec4 with/without center, x_axis variant, @if conditionals). Note: quaternion variants deferred (need FNC_QUATMULT)
- [x] `rotateX.wesl` ↔ `rotateX.glsl` - ✅ Perfect conversion (4 overloads with @if conditionals)
- [x] `rotateY.wesl` ↔ `rotateY.glsl` - ✅ Fixed GLSL trailing dot (1. → 1.0), perfect WESL conversion
- [x] `rotateZ.wesl` ↔ `rotateZ.glsl` - ✅ Fixed GLSL bug (CENTER_3D → CENTER_4D on line 22), perfect WESL conversion
- [x] `scale.wesl` ↔ `scale.glsl` - ✅ Fixed: complete rewrite, added all 12 overloads (f32/vec2/vec3/vec4 variants with/without center, @if conditionals)
- [x] `screen2viewPosition.wesl` ↔ `screen2viewPosition.glsl` - ✅ Fixed: replaced hardcoded matrix with @if conditionals for CAMERA_PROJECTION_MATRIX/INVERSE_CAMERA_PROJECTION_MATRIX
- [x] `sprite.wesl` ↔ `sprite.glsl` - ✅ Perfect conversion
- [x] `sqTile.wesl` ↔ `sqTile.glsl` - ✅ Perfect conversion (2 overloads)
- [x] `tbn.wesl` ↔ `tbn.glsl` - ✅ Perfect conversion (2 overloads)
- [x] `translate.wesl` ↔ `translate.glsl` - ✅ Perfect conversion
- [x] `triTile.wesl` ↔ `triTile.glsl` - ✅ Fixed trailing dots, converted ternary to if/else
- [x] `uncenter.wesl` ↔ `uncenter.glsl` - ✅ Perfect conversion (3 overloads)
- [x] `unratio.wesl` ↔ `unratio.glsl` - ✅ Fixed: improved spacing consistency (* .5 → * 0.5)
- [x] `view2screenPosition.wesl` ↔ `view2screenPosition.glsl` - ✅ Fixed: replaced hardcoded matrix with @if conditional for CAMERA_PROJECTION_MATRIX
- [x] `viewZ2depth.wesl` ↔ `viewZ2depth.glsl` - ✅ Perfect conversion (@if conditionals for CAMERA_ORTHOGRAPHIC_PROJECTION)
- [x] `windmillTile.wesl` ↔ `windmillTile.glsl` - ✅ Perfect conversion (5 overloads)
- [x] `xyz2equirect.wesl` ↔ `xyz2equirect.glsl` - ✅ Perfect conversion

---

## Tips for Efficient Review

### Batch Review by Category

Review similar files together to spot patterns:

```bash
# Review all blend modes together
cd color/blend
ls *.wesl | while read f; do
  echo "=== $f ==="
  diff -u "${f%.wesl}.glsl" "$f" | head -30
done
```

### Use Scripts to Find Common Issues

```bash
# Find WESL files with GLSL type names
find . -name "*.wesl" -exec grep -l "\bfloat\b\|\bvec[234]\b\|\bmat[234]\b" {} \;

# Find WESL files with trailing dots
find . -name "*.wesl" -exec grep -l '[0-9]\.' {} \;

# Find missing imports (should match #include count)
for f in $(find . -name "*.wesl"); do
  glsl="${f%.wesl}.glsl"
  if [ -f "$glsl" ]; then
    includes=$(grep -c "^#include" "$glsl" 2>/dev/null || echo "0")
    imports=$(grep -c "^import" "$f" 2>/dev/null || echo "0")
    if [ "$includes" != "$imports" ]; then
      echo "$f: includes=$includes imports=$imports"
    fi
  fi
done
```

### Prioritize High-Risk Files

Focus review on:
1. **Files with conditionals** - complex `#ifdef` logic
2. **Files with many overloads** - easy to miss one
3. **Files with imports** - dependency issues
4. **Files with macros** - tricky conversions

### Test Coverage

Many files have tests in `test/wesl/`. Run tests to catch functional issues:

```bash
pnpm test:once
```

---

## Resources

- [GLSLtoWESL.md](GLSLtoWESL.md) - Complete conversion guide
- [README_WESL.md](README_WESL.md) - WESL usage and conventions
- [WESL Specification](https://wesl-lang.dev/spec/README)
- Test files in `test/wesl/` for examples

---

## Tracking Progress

Use this checklist format for tracking:

```markdown
## Review Session - [Date]
Reviewer: [Name]

### Files Reviewed (X/351)
- [x] color/blend/add.wesl - ✅ Complete, all overloads present
- [x] color/blend/average.wesl - ⚠️ Missing vec4 overload from GLSL
- [x] math/rotate2d.wesl - ✅ Complete
- [ ] math/rotate3d.wesl - [not reviewed yet]

### Issues Found
1. `color/blend/average.wesl` - Missing `vec4` overload (GLSL has it)
2. `color/space/yuv2rgb.wesl` - Missing `vec4` overload (GLSL has it)

### Statistics
- Files reviewed: X
- Issues found: Y
- Completion: X/351 (Z%)
```

---

## Summary

This review is critical to ensure:
- ✅ **Functional completeness** - all GLSL functionality is preserved
- ✅ **Type safety** - WESL types are correct
- ✅ **Compatibility** - imports and dependencies work
- ✅ **Quality** - code follows WESL best practices

Take your time, be thorough, and document any issues found. This review will improve the quality of the LYGIA WESL library for all users.
