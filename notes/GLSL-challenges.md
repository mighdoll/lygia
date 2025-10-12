# GLSL to WESL Conversion Challenges

This document identifies specific conversion challenges found in the Lygia GLSL files that require special attention or alternative approaches when converting to WESL.

**Total files to convert: 432** (see [GlslToConvert.md](GlslToConvert.md))

**Note:** Files requiring texture sampling or macro-based function composition are deferred for later conversion.

---

## 1. Texture Sampling and Sampler Abstraction ⚠️ DEFER

### Challenge
GLSL files use macro-based sampler abstraction to support different GLSL versions. WGSL has a fundamentally different texture system with separate texture and sampler objects. This requires significant design work to handle properly.

### Examples
- **File:** `/Users/lee/wesl/lygia/sampler.glsl`
- **Pattern:**
  ```glsl
  #ifndef SAMPLER_FNC
  #if __VERSION__ >= 300
  #define SAMPLER_FNC(TEX, UV) texture(TEX, UV)
  #else
  #define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
  #endif
  #endif

  #ifndef SAMPLER_TYPE
  #define SAMPLER_TYPE sampler2D
  #endif
  ```

### Affected Files (30+ files) - **DEFERRED**
- All files in `sample/` directory (~30 files)
- `/Users/lee/wesl/lygia/sample/yuv.glsl`
- `/Users/lee/wesl/lygia/filter/` directory (many blur/filter files)
- `/Users/lee/wesl/lygia/lighting/material/new.glsl`
- All files using `SAMPLER_FNC` or `SAMPLER_TYPE`

### Why Defer
The texture/sampler system in WGSL is fundamentally different from GLSL and requires architectural decisions about how to expose textures and samplers to shader functions. This is beyond the scope of initial conversion work.

### Status
**DEFERRED** - These files are marked in GlslToConvert.md and will be converted later after establishing a texture/sampler design pattern.

---

## 2. Struct Definitions with Conditional Members ✅ SUPPORTED

### Challenge
GLSL uses preprocessor directives to conditionally include struct members. WESL supports `@if` on struct members!

### Examples
- **File:** `/Users/lee/wesl/lygia/lighting/material.glsl`
- **Pattern:**
  ```glsl
  struct Material {
      vec4    albedo;
      vec3    emissive;
      vec3    position;
      vec3    normal;

  #if defined(RENDER_RAYMARCHING)
      float   sdf;
      bool    valid;
  #endif

  #if defined(SHADING_MODEL_CLEAR_COAT)
      float   clearCoat;
      float   clearCoatRoughness;
      #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
      vec3    clearCoatNormal;
      #endif
  #endif

  #if defined(SHADING_MODEL_SUBSURFACE)
      vec3    subsurfaceColor;
      float   subsurfacePower;
      float   subsurfaceThickness;
  #endif
  };
  ```

### Affected Files
- `/Users/lee/wesl/lygia/lighting/material.glsl`
- `/Users/lee/wesl/lygia/lighting/light/directional.glsl`
- `/Users/lee/wesl/lygia/lighting/light/point.glsl`
- `/Users/lee/wesl/lygia/geometry/aabb/aabb.glsl` (simpler case)

### WESL Support
WESL supports `@if` conditionals on struct members! This makes conversion straightforward.

### Solution Strategy
Convert GLSL preprocessor conditionals to WESL `@if` attributes on struct members:

```wgsl
struct Material {
    // Core fields (always present)
    albedo: vec4f,
    emissive: vec3f,
    position: vec3f,
    normal: vec3f,

    // Optional fields with @if
    @if(RENDER_RAYMARCHING)
    sdf: f32,

    @if(RENDER_RAYMARCHING)
    valid: bool,

    @if(SHADING_MODEL_CLEAR_COAT)
    clearCoat: f32,

    @if(SHADING_MODEL_CLEAR_COAT)
    clearCoatRoughness: f32,

    // Nested conditions work too
    @if(SHADING_MODEL_CLEAR_COAT && MATERIAL_HAS_CLEAR_COAT_NORMAL)
    clearCoatNormal: vec3f,

    // ... other conditional fields
}
```

This maintains the same conditional structure as the GLSL version while using WESL's native conditional compilation.

---

## 3. Output Parameters and Struct Initialization

### Challenge
GLSL commonly uses `out` parameters to initialize structs. WGSL doesn't have `out` parameters; it uses return values or pointer parameters.

### Examples
- **File:** `/Users/lee/wesl/lygia/lighting/light/directionalNew.glsl` (renamed from `new.glsl`)
- **Pattern:**
  ```glsl
  void lightNew(out LightDirectional _L) {
      _L.direction = normalize(LIGHT_DIRECTION);
      _L.color = LIGHT_COLOR;
      _L.intensity = LIGHT_INTENSITY;
  }

  LightDirectional LightDirectionalNew() {
      LightDirectional l;
      lightNew(l);
      return l;
  }
  ```

### Affected Files (~20+ files)
- `/Users/lee/wesl/lygia/lighting/light/new.glsl` → rename to `directionalNew.wesl` or `pointNew.wesl`
- `/Users/lee/wesl/lygia/lighting/material/new.glsl` → rename to `materialNew.wesl`
- `/Users/lee/wesl/lygia/lighting/shadingData/new.glsl` → rename to `shadingDataNew.wesl`
- `/Users/lee/wesl/lygia/lighting/ray/new.glsl` → rename to `rayNew.wesl`
- Many constructors throughout the lighting system

### Important: Avoid `new.wesl` Filename
**Do not use `new.wesl` as a filename** - this creates potential keyword conflicts. Use descriptive names like:
- `materialNew.wesl`
- `lightDirectionalNew.wesl`
- `shadingDataNew.wesl`
- `rayNew.wesl`

### WGSL Approach
```wgsl
// Return the struct directly (preferred approach)
fn lightDirectionalNew() -> LightDirectional {
    var l: LightDirectional;
    l.direction = normalize(LIGHT_DIRECTION);
    l.color = LIGHT_COLOR;
    l.intensity = LIGHT_INTENSITY;
    return l;
}

// Option 2: Use pointer parameter (more verbose, avoid if possible)
fn lightNew(l: ptr<function, LightDirectional>) {
    (*l).direction = normalize(LIGHT_DIRECTION);
    // ...
}
```

### Solution Strategy
1. Convert `out` parameter functions to return the struct directly
2. Eliminate the dual pattern (both void with `out` and returning version)
3. Keep only the return-based version for clarity
4. **Rename files from `new.glsl` to `<type>New.wesl`**

---

## 4. Platform-Specific Code Paths

### Challenge
GLSL files contain conditional compilation for platform differences (WebGL, desktop GL, version differences). WGSL/WESL may need different conditions.

### Examples
- **File:** `/Users/lee/wesl/lygia/filter/gaussianBlur/1D.glsl` (uses texture sampling - DEFERRED)
- **File:** `/Users/lee/wesl/lygia/color/palette/lerp.glsl` (uses macro functions - DEFERRED)
- **Pattern:**
  ```glsl
  #ifdef PLATFORM_WEBGL
  // Fixed loop limit of 16 for WebGL
  for (int i = 0; i < 16; i++) {
      if (i >= kernelSize) break;
      // ...
  }
  #else
  // Dynamic loop for desktop
  for (int i = 0; i < kernelSize; i++) {
      // ...
  }
  #endif
  ```

### Affected Files
- `/Users/lee/wesl/lygia/filter/gaussianBlur/1D.glsl` (DEFERRED - texture sampling)
- `/Users/lee/wesl/lygia/color/palette/lerp.glsl` (DEFERRED - macro functions)
- `/Users/lee/wesl/lygia/math/transpose.glsl` (version-based: `__VERSION__ < 120`)

### WGSL Consideration
- WebGPU is more uniform across platforms
- WGSL has stricter loop requirements in some contexts
- May not need platform-specific paths, but might need variants for different shader stages or capabilities

### Solution Strategy
1. **For loop limits:** Use WESL `@if` to provide alternative implementations
   ```wgsl
   @if(REQUIRE_CONSTANT_LOOP)
   fn blur(...) -> vec4f {
       // Version with constant loop limit
       for (var i = 0; i < 16; i++) {
           if (i >= kernelSize) { break; }
           // ...
       }
   }
   @else
   fn blur(...) -> vec4f {
       // Version with dynamic loop
       for (var i = 0; i < kernelSize; i++) {
           // ...
       }
   }
   ```
2. **For built-in functions:** Check if WGSL has the built-in (e.g., `transpose()` is built-in in WGSL)
3. Document which features may have performance implications on different platforms

---

## 5. Version-Based Feature Detection

### Challenge
GLSL uses `__VERSION__` macro for feature detection. WGSL doesn't have version macros.

### Examples
- **File:** `/Users/lee/wesl/lygia/math/transpose.glsl`
- **Pattern:**
  ```glsl
  #if !defined(FNC_TRANSPOSE) && (__VERSION__ < 120)
  #define FNC_TRANSPOSE
  mat3 transpose(in mat3 m) {
      return mat3(m[0][0], m[1][0], m[2][0], ...);
  }
  #endif
  ```

### Affected Files
- `/Users/lee/wesl/lygia/math/transpose.glsl` (likely can be skipped - WGSL has built-in)
- `/Users/lee/wesl/lygia/sampler.glsl` (DEFERRED - texture sampling)

### WGSL Approach
WGSL already has `transpose()` as a built-in function. Many GLSL polyfills are unnecessary in WGSL.

### Solution Strategy
1. Check WGSL specification for built-in functions
2. Skip files that only provide polyfills for WGSL built-ins
3. If polyfill is still needed for some reason, use WESL `@if` with a feature flag instead of version check

---

## 6. Macro-Based Function Composition ⚠️ DEFER

### Challenge
GLSL uses macros to allow users to customize function behavior. WGSL doesn't have macros or function pointers, making this pattern difficult to replicate without significant design changes.

### Examples
- **File:** `/Users/lee/wesl/lygia/color/palette/lerp.glsl`
- **Pattern:**
  ```glsl
  #ifndef PALETTE_LERP_MIX_FNC
  #define PALETTE_LERP_MIX_FNC(A, B, T) mix(A, B, T)
  #endif

  vec3 paletteLerp(vec3 a[PALETTE_LERP_SIZE], float t) {
      // ...
      vec3 result = PALETTE_LERP_MIX_FNC(paletteLerp_get(a, int(index1)),
                                         paletteLerp_get(a, int(index)),
                                         index1 - index);
      // ...
  }
  ```
- **File:** `/Users/lee/wesl/lygia/sdf/circleSDF.glsl`
  ```glsl
  #ifndef CIRCLESDF_FNC
  #define CIRCLESDF_FNC(POS_UV) length(POS_UV)
  #endif

  float circleSDF(in vec2 v) {
      return CIRCLESDF_FNC(v) * 2.0;
  }
  ```

### Affected Files - **DEFERRED**
- `/Users/lee/wesl/lygia/color/palette/lerp.glsl` (`PALETTE_LERP_MIX_FNC`)
- `/Users/lee/wesl/lygia/sdf/circleSDF.glsl` (`CIRCLESDF_FNC`)
- Various other files with customizable function hooks via macros

### Why Defer
WGSL doesn't support macros or higher-order functions. While `@if` can provide some alternatives, this requires careful design to maintain usability without the flexibility that macros provide. These files need a clear pattern established before conversion.

### Status
**DEFERRED** - These files are marked in GlslToConvert.md. Note: Many files use texture sampling macros (see Challenge #1), which are already deferred.

---

## 7. Array Parameters with Compile-Time Size

### Challenge
GLSL allows array parameters where size is defined by preprocessor. WGSL requires explicit array sizes at compile time.

### Examples
- **File:** `/Users/lee/wesl/lygia/color/palette/lerp.glsl` (also uses macro functions - DEFERRED)
- **Pattern:**
  ```glsl
  #define PALETTE_LERP_SIZE 5

  vec3 paletteLerp_get(vec3 a[PALETTE_LERP_SIZE], int index) {
      index = int(mod(float(index), float(PALETTE_LERP_SIZE)));
      #if defined(PLATFORM_WEBGL)
      for (int i = 0; i < PALETTE_LERP_SIZE; i++)
          if (i == index) return a[i];
      #else
      return a[index];
      #endif
  }

  vec3 paletteLerp(vec3 a[PALETTE_LERP_SIZE], float t) { ... }
  ```

### Affected Files
- `/Users/lee/wesl/lygia/color/palette/lerp.glsl` (DEFERRED for other reasons)
- Potentially others using array parameters

### WGSL Approach
```wgsl
// Option 1: Use const for size
const PALETTE_LERP_SIZE: i32 = 5;

fn paletteLerp(a: array<vec3f, 5>, t: f32) -> vec3f { ... }

// Option 2: Use @if to generate multiple variants
@if(PALETTE_SIZE_5)
fn paletteLerp5(a: array<vec3f, 5>, t: f32) -> vec3f { ... }

@if(PALETTE_SIZE_10)
fn paletteLerp10(a: array<vec3f, 10>, t: f32) -> vec3f { ... }
```

### Solution Strategy
1. Define array size as `const` in WESL
2. Use explicit array size in function signatures
3. For multiple size support, use `@if` to create variants or use separate function names
4. Document required constants in file header

---

## 8. Barrel/Aggregator Files - SKIP

### Challenge
Many GLSL files are just collections of `#include` statements (barrel files). These should be skipped for WESL conversion as they provide no functional code.

### Examples
- **File:** `/Users/lee/wesl/lygia/animation/easing/cubic.glsl`
- **Pattern:**
  ```glsl
  #include "cubicIn.glsl"
  #include "cubicOut.glsl"
  #include "cubicInOut.glsl"
  ```

### Affected Files - **NOT CONVERTING**
These files should not appear in GlslToConvert.md as they were filtered out:
- `/Users/lee/wesl/lygia/animation/easing.glsl`
- `/Users/lee/wesl/lygia/animation/easing/cubic.glsl` (and other easing categories)
- `/Users/lee/wesl/lygia/color/tonemap.glsl`
- `/Users/lee/wesl/lygia/color/space.glsl`
- `/Users/lee/wesl/lygia/color/palette.glsl`
- `/Users/lee/wesl/lygia/color/dither.glsl`

### Solution
**Skip these files entirely.** Users should import the actual implementation files directly:

```wgsl
// Instead of importing a barrel file, import specific functions:
import lygia::animation::easing::cubicIn::cubicIn;
import lygia::animation::easing::cubicOut::cubicOut;
import lygia::animation::easing::cubicInOut::cubicInOut;
```

### Status
**NOT CONVERTING** - These files should already be excluded from GlslToConvert.md.

---

## 9. Multiple Related Overloads (Function Families)

### Challenge
GLSL files often define many related overloads. WGSL requires unique names, leading to potential naming complexity.

### Examples
- **File:** `/Users/lee/wesl/lygia/generative/random.glsl`
- **Pattern:**
  ```glsl
  float random(in float x) { ... }
  float random(in vec2 st) { ... }
  float random(in vec3 pos) { ... }
  float random(in vec4 pos) { ... }

  vec2 random2(float p) { ... }
  vec2 random2(vec2 p) { ... }
  vec2 random2(vec3 p) { ... }

  vec3 random3(float p) { ... }
  vec3 random3(vec2 p) { ... }
  vec3 random3(vec3 p) { ... }

  vec4 random4(float p) { ... }
  vec4 random4(vec2 p) { ... }
  vec4 random4(vec3 p) { ... }
  vec4 random4(vec4 p) { ... }
  ```

### Affected Files (Many!)
- `/Users/lee/wesl/lygia/generative/random.glsl` (16 functions!)
- `/Users/lee/wesl/lygia/color/blend/add.glsl` (and all blend modes ~27 files)
- Most SDF functions (~40 files) have 2-3 overloads
- Many math utility functions

### WGSL Naming Convention
From the existing WESL files, the pattern is:
```wgsl
fn random(p: f32) -> f32 { ... }
fn random2(st: vec2f) -> f32 { ... }
fn random3(pos: vec3f) -> f32 { ... }
fn random4(p: vec4f) -> f32 { ... }

fn random21(p: f32) -> vec2f { ... }
fn random22(p: vec2f) -> vec2f { ... }
fn random23(p: vec3f) -> vec2f { ... }

fn random31(p: f32) -> vec3f { ... }
fn random32(p: vec2f) -> vec3f { ... }
fn random33(p: vec3f) -> vec3f { ... }

fn random41(p: f32) -> vec4f { ... }
fn random42(p: vec2f) -> vec4f { ... }
fn random43(p: vec3f) -> vec4f { ... }
fn random44(p: vec4f) -> vec4f { ... }
```

### Solution Strategy
1. Use systematic naming: `functionName[OutputDim][InputDim]`
2. For same return type: `functionName`, `functionName2`, `functionName3`, `functionName4`
3. For different return types: `functionNameXY` where X=output dim, Y=input dim
4. Document the naming convention clearly
5. Consider if all variants are needed or if some can be omitted

---

## 10. Complex Include Dependencies

### Challenge
Some files have deep include chains that might create circular dependencies or require careful ordering in WESL.

### Examples
- **File:** `/Users/lee/wesl/lygia/lighting/pbr.glsl`
- **Pattern:**
  ```glsl
  #include "../math/saturate.glsl"
  #include "shadingData/new.glsl"
  #include "material.glsl"
  #include "light/new.glsl"
  #include "light/resolve.glsl"
  #include "light/iblEvaluate.glsl"
  ```
  And each of those files includes others...

### Affected Files
- Most files in `lighting/` directory (~74 files)
- Complex shader pipelines
- Files with many dependencies

### WESL Import System
WESL handles imports differently:
- Imports are resolved recursively
- Cyclic imports are allowed
- Everything is public by default
- No include guards needed
- Use `lygia::` prefix for Lygia library imports

### Solution Strategy
1. Convert include chains to explicit imports using `lygia::` prefix
   ```wgsl
   import lygia::math::saturate::saturate;
   import lygia::lighting::shadingData::shadingDataNew::shadingDataNew;
   import lygia::lighting::material::Material;
   import lygia::lighting::light::directionalNew::lightDirectionalNew;
   ```
2. Import only what's needed from each module
3. Let WESL handle circular dependencies
4. Test thoroughly to ensure all dependencies resolve
5. Consider flattening some deeply nested dependencies

---

## Summary of Conversion Priorities

### Deferred / Skip
1. **Texture Sampling** - Requires design work ⚠️ DEFER
2. **Macro Function Composition** - Requires design work ⚠️ DEFER
3. **Barrel Files** - Skip entirely ⏭️ SKIP

### Straightforward Conversions
4. **Struct Conditional Members** - Use `@if` on struct members ✅
5. **Output Parameters** - Return structs directly, rename `new.glsl` files ✅
6. **Platform-Specific Code** - Use `@if` for variants ✅
7. **Version Detection** - Often unnecessary (built-ins exist) ✅
8. **Array Size Parameters** - Use `const` and explicit sizes ✅
9. **Function Overloading** - Systematic renaming ✅
10. **Include Dependencies** - Convert to `lygia::` imports ✅

---

## Recommended Conversion Order

1. **Start with:** Simple math utilities (no texture sampling, few dependencies)
   - `math/` directory functions (excluding those with texture dependencies)
   - Simple `color/space/` conversions

2. **Next:** SDF functions (straightforward, self-contained)
   - `sdf/` directory (~40 files) - excluding those with macro customization

3. **Then:** Structs and simple constructors
   - `geometry/aabb/`, `lighting/light/`, etc.
   - Remember to rename `new.glsl` → `<type>New.wesl`

4. **Later:** Animation easing functions
   - `animation/easing/` directory

5. **Defer:** Functions with texture sampling or macro composition
   - `sample/`, `filter/` directories
   - Files with `SAMPLER_FNC`, `SAMPLER_TYPE`, or function macros

---

## Testing Strategy

1. **Unit Tests:** Create WESL test for each converted function
2. **Integration Tests:** Test function combinations
3. **Comparison Tests:** Compare output with GLSL version where possible
4. **Feature Flags:** Test all `@if` branches
5. **Platform Tests:** Verify on different WebGPU implementations

---

## Additional Resources

- [GLSLtoWESL.md](GLSLtoWESL.md) - Detailed conversion guide
- [README_WESL.md](README_WESL.md) - WESL contribution guidelines
- [WESL Specification](https://wesl-lang.dev/spec/README)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
