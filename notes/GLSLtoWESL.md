# GLSL to WESL Conversion Guide

This guide helps convert GLSL shader files from the Lygia library to WESL (WebGPU Enhanced Shading Language).

## Overview

WESL is a strict superset of WGSL (WebGPU Shading Language) with added features for imports and conditional compilation. Since WGSL and GLSL have different syntax, conversion requires systematic changes.

---

## Key Syntax Differences

### 1. Type Names

GLSL types must be converted to WGSL equivalents:

| GLSL | WESL/WGSL |
|------|-----------|
| `float` | `f32` |
| `int` | `i32` |
| `uint` | `u32` |
| `bool` | `bool` (same) |
| `vec2` | `vec2f` or `vec2<f32>` |
| `vec3` | `vec3f` or `vec3<f32>` |
| `vec4` | `vec4f` or `vec4<f32>` |
| `ivec2` | `vec2i` or `vec2<i32>` |
| `ivec3` | `vec3i` or `vec3<i32>` |
| `ivec4` | `vec4i` or `vec4<i32>` |
| `uvec2` | `vec2u` or `vec2<u32>` |
| `uvec3` | `vec3u` or `vec3<u32>` |
| `uvec4` | `vec4u` or `vec4<u32>` |
| `mat2` | `mat2x2f` or `mat2x2<f32>` |
| `mat3` | `mat3x3f` or `mat3x3<f32>` |
| `mat4` | `mat4x4f` or `mat4x4<f32>` |

### 2. Vector Constructors

Vector constructors require the type suffix:

```glsl
// GLSL
vec3(1.0)
vec3(1.0, 2.0, 3.0)
vec2(0.5)
```

```wgsl
// WESL
vec3(1.0)          // shorthand for vec3f(1.0)
vec3f(1.0, 2.0, 3.0)
vec2(0.5)          // shorthand for vec2f(0.5)
```

### 3. Function Declarations

WGSL/WESL uses different function syntax:

```glsl
// GLSL
float circleSDF(in vec2 v) {
    return length(v) * 2.0;
}

vec3 blendAdd(in vec3 base, in vec3 blend) {
    return min(base + blend, vec3(1.));
}
```

```wgsl
// WESL
fn circleSDF(v: vec2f) -> f32 {
    return length(v) * 2.0;
}

fn blendAdd3(base: vec3f, blend: vec3f) -> vec3f {
    return min(base + blend, vec3(1.0));
}
```

**Key differences:**
- Use `fn` keyword instead of return type first
- Parameter syntax: `name: type` instead of `type name`
- Return type after `->` instead of before function name
- No `in`, `out`, `inout` qualifiers (use pointers with `&` for output parameters if needed)
- Function overloading by unique names (e.g., `blendAdd`, `blendAdd3`, `blendAdd3Opacity`)

### 4. Variable Declarations

```glsl
// GLSL
float x = 1.0;
vec3 color = vec3(1.0, 0.0, 0.0);
const float PI = 3.14159;
```

```wgsl
// WESL
var x: f32 = 1.0;
let color: vec3f = vec3(1.0, 0.0, 0.0);
const PI: f32 = 3.14159;
```

**Key differences:**
- Use `var` for mutable variables (can change)
- Use `let` for immutable variables (cannot change, runtime constant)
- Use `const` for compile-time constants
- Type annotation with `:` after variable name
- Type inference works, so you can often omit the type: `let x = 1.0;`

### 5. Numeric Literals

WGSL requires explicit type suffixes or decimal points:

```glsl
// GLSL
1.       // float with trailing dot
.5       // float with leading dot
1        // int or float depending on context
```

```wgsl
// WESL
1.0      // f32 (always use .0 for floats)
0.5      // f32
1        // i32 (integer)
1u       // u32 (unsigned)
1.0f     // explicit f32 suffix (optional)
```

**Best practice:** Always use `.0` for float literals to avoid ambiguity.

---

## Import System

### GLSL Includes → WESL Imports

```glsl
// GLSL
#include "hue2rgb.glsl"
#include "../math/saturate.glsl"
```

```wgsl
// WESL (Lygia library)
import lygia::color::space::hue2rgb::hue2rgb;
import lygia::math::saturate::saturate;
```

**Import rules:**
- Directory hierarchy maps to `::` separated paths
- Must import specific items (functions, structs, constants)
- Can import multiple items: `import path::{ item1, item2 };`
- Can rename imports: `import path::item as new_name;`
- Imports must be at the top of the file
- Use `lygia::` to reference the root of the Lygia library
- Use `super::` to reference parent directory

**Path mapping examples:**
- `color/space/hue2rgb.wesl` → `import lygia::color::space::hue2rgb::hue2rgb;`
- `math/saturate.wesl` → `import lygia::math::saturate::saturate;`
- `./foo.wesl` (same directory) → `import super::foo::functionName;`

---

## Conditional Compilation

### Preprocessor Directives → @if Attributes

```glsl
// GLSL
#ifndef FNC_CIRCLERSDF
#define FNC_CIRCLESDF

#ifdef CENTER_2D
    v -= CENTER_2D;
#else
    v -= 0.5;
#endif

#ifndef CIRCLESDF_FNC
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
#endif

#endif
```

```wgsl
// WESL
fn circleSDF(v: vec2f) -> f32 {
    var pos = v;
    @if(USE_CUSTOM_CENTER)
    pos -= vec2f(0.3, 0.7);  // custom center point
    @else
    pos -= 0.5;              // default center

    return length(pos) * 2.0;
}
```

**Key differences:**
- Use `@if(feature)` instead of `#ifdef feature`
- Use `@else` instead of `#else`
- Use `@elif(condition)` instead of `#elif condition`
- **Conditions are strictly boolean expressions** (true/false), unlike GLSL's `#ifdef` which tests if something is defined
- Supports boolean expressions: `@if(DEBUG && !RELEASE)`
- Supports logical operators: `@if(A || B)`, `@if(!C)`
- Condition values are set via the WESL linker API (see "How to set conditions" below)
- No include guards needed (WESL handles this automatically)
- No macro functions like `#define FUNC(x) expr` - use regular functions instead

**Common patterns:**

```wgsl
// Simple feature flag
@if(USE_TEXTURE)
fn textured_version() -> vec4f { /* ... */ }

// Complex boolean expression
@if(LEGACY_MODE || (WEB_VERSION && !XYZ_SUPPORTED))
fn legacy_implementation() -> f32 { /* ... */ }

// Alternative implementations (use @else for clarity)
@if(FAST_PATH)
fn compute() -> f32 { /* fast version */ }
@else
fn compute() -> f32 { /* accurate version */ }

// Inline conditionals for single statements
fn scale(v: vec2f, s: f32) -> vec2f {
    @if(USE_CUSTOM_CENTER)
    let center = constants::CENTER;
    @else
    let center = vec2f(0.5);

    return (v - center) * s + center;
}
```

**How to set conditions:**

Conditions are set via the WESL linker API:

```typescript
// JavaScript/TypeScript
import { link } from "wesl";

const linked = await link(shaderSource, {
  conditions: {
    USE_TEXTURE: true,
    LEGACY_MODE: false,
    FAST_PATH: true,
  }
});
```

```rust
// Rust
let shader = Wesl::new("src/shaders")
    .with_conditions([("USE_TEXTURE", true), ("FAST_PATH", true)])
    .compile("main.wesl")?;
```

See the [WESL Conditional Translation spec](https://wesl-lang.dev/spec/ConditionalTranslation) for more details.

---

## Function Overloading

GLSL supports function overloading; WGSL/WESL requires unique function names.

```glsl
// GLSL - same function name, different signatures
float blendAdd(in float base, in float blend) { ... }
vec3 blendAdd(in vec3 base, in vec3 blend) { ... }
vec3 blendAdd(in vec3 base, in vec3 blend, float opacity) { ... }
```

```wgsl
// WESL - unique names with suffixes
fn blendAdd(base: f32, blend: f32) -> f32 { ... }
fn blendAdd3(base: vec3f, blend: vec3f) -> vec3f { ... }
fn blendAdd3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f { ... }
```

**Naming conventions:**
- Append vector dimension: `random`, `random2`, `random3`, `random4`
- Append input/output dimensions: `random21` (f32 → vec2f), `random32` (vec2f → vec3f)
- Append variant name: `blendAdd3Opacity`

---

## Common Patterns

### 1. Include Guards → Not Needed

```glsl
// GLSL
#ifndef FNC_RANDOM
#define FNC_RANDOM
// ... code ...
#endif
```

```wgsl
// WESL - no include guards needed
// WESL automatically handles duplicate imports
// ... code ...
```

### 2. Macro Definitions → Constants or Functions

```glsl
// GLSL
#define PI 3.14159
#define RANDOM_SCALE vec4(.1031, .1030, .0973, .1099)
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
```

```wgsl
// WESL
const PI: f32 = 3.14159;
const RANDOM_SCALE: vec4f = vec4f(.1031, .1030, .0973, .1099);

// For function-like macros, use actual functions:
fn circlesdf_fnc(pos_uv: vec2f) -> f32 {
    return length(pos_uv);
}
```

### 3. Options/Configuration

Many GLSL files use `#ifdef` for optional features. In WESL, use `@if()` with conditions set via the linker:

```glsl
// GLSL
#ifdef RANDOM_SINLESS
    // sinless implementation
#else
    // sin-based implementation
#endif
```

```wgsl
// WESL
// RANDOM_SINLESS is set via linker API (see "How to set conditions" above)

fn random(p: f32) -> f32 {
    var x = p;
    @if(RANDOM_SINLESS)
    x = fract(x * RANDOM_SCALE.x);
    x *= x + 33.33;
    x *= x + x;
    return fract(x);
    @else
    return fract(sin(x) * 43758.5453);
}
```

### 4. Const Parameters

GLSL's `const in` parameters become regular parameters in WESL:

```glsl
// GLSL
vec3 hsl2rgb(const in vec3 hsl) { ... }
```

```wgsl
// WESL
fn hsl2rgb(hsl: vec3f) -> vec3f { ... }
```

Parameters in WESL are immutable by default (similar to `const in`).

---

## File Structure Template

Here's a typical WESL file structure for converted Lygia functions:

```wgsl
// 1. Imports (if needed)
import lygia::color::space::hue2rgb::hue2rgb;

// 2. Comments/documentation
/*
contributors: [Author Names]
description: Function description
use: functionName(<type> param1, <type> param2)
options:
    - OPTION_NAME: description
license: License text
*/

// 3. Constants (replacing #define)
const SOME_CONSTANT: f32 = 1.0;

// 4. Feature flags as const (if used)
@if(FEATURE_NAME)
const feature_enabled: bool = true;

// 5. Function implementations
fn functionName(param1: type1, param2: type2) -> returnType {
    // Implementation
}

// 6. Overloaded variants with unique names
fn functionName2(param: vec2f) -> f32 { ... }
fn functionName3(param: vec3f) -> f32 { ... }
```

---

## Common Gotchas

### 1. WGSL Reserved Words

WGSL has reserved keywords that cannot be used as identifiers (variable names, parameter names, function names, etc.). Some common reserved words that differ from GLSL include:

**Commonly problematic reserved words:**
- `target` - use `center`, `dest`, `destination` instead
- `uniform` - use `value`, `val` instead
- `varying` - use `interpolated` instead
- `attribute` - use `attr`, `property` instead

**Other WGSL reserved words to avoid:**
- Storage qualifiers: `storage`, `workgroup`, `private`, `function`
- Sampler/texture: `sampler`, `texture`, `depth`
- Address spaces: `read`, `write`, `read_write`
- Built-in types: `array`, `mat`, `vec`, `atomic`, `ptr`, `ref`

**Example issue:**
```wgsl
// ❌ ERROR: 'target' is reserved
fn lookAt(eye: vec3f, target: vec3f, up: vec3f) -> mat3x3f { ... }

// ✅ CORRECT: use 'center' instead
fn lookAt(eye: vec3f, center: vec3f, up: vec3f) -> mat3x3f { ... }
```

**Best practice:** If you get cryptic parsing errors like "invalid fn, expected function parameters", check if any parameter or variable names are WGSL reserved words.

### 2. Swizzling works the same

```glsl
vec3 p3 = vec3(p.xyx);  // GLSL
```

```wgsl
let p3 = vec3(p.xyx);  // WESL - works the same!
```

### 2. Built-in functions

Most GLSL built-in functions work in WGSL/WESL:
- `length()`, `dot()`, `cross()`, `normalize()`
- `sin()`, `cos()`, `tan()`, `abs()`, `min()`, `max()`
- `fract()`, `floor()`, `ceil()`, `clamp()`, `mix()`, `step()`, `smoothstep()`
- `fwidth()`, `dFdx()`, `dFdy()`

### 3. No implicit type conversions

WGSL is stricter about types:

```glsl
// GLSL - works
float x = 1;  // implicit int to float conversion
```

```wgsl
// WESL - error! must be explicit
let x: f32 = 1;     // Error: type mismatch
let x: f32 = 1.0;   // Correct
let x = f32(1);     // Also correct
```

### 4. Texture sampling

GLSL texture functions become separate texture and sampler in WGSL - this is complex and may require additional work beyond simple conversion.

### 5. Arrays

```glsl
// GLSL
float values[3] = float[](1.0, 2.0, 3.0);
```

```wgsl
// WESL
var values: array<f32, 3> = array<f32, 3>(1.0, 2.0, 3.0);
// or with type inference:
let values = array(1.0, 2.0, 3.0);
```

---

## Conversion Checklist

When converting a GLSL file to WESL:

- [ ] Replace `#include` with `import` statements
- [ ] Convert all type names (`float` → `f32`, `vec3` → `vec3f`, etc.)
- [ ] Update function syntax (`fn name(param: type) -> returnType`)
- [ ] Rename overloaded functions to unique names
- [ ] **Check for WGSL reserved words** (e.g., `target`, `uniform`, `varying`) and rename identifiers
- [ ] Convert `#define` constants to `const` declarations
- [ ] Replace `#ifdef`/`#ifndef`/`#endif` with `@if`/`@else`
- [ ] Remove include guards (`#ifndef FNC_*`)
- [ ] Update numeric literals (add `.0` to floats)
- [ ] Change `var` declarations to use `: type` syntax
- [ ] Convert `const in` parameters to regular parameters
- [ ] Replace macro functions with actual functions
- [ ] Test with WESL compiler/translator
- [ ] Verify imports resolve correctly

---

## Example Conversion

### Before (GLSL):
```glsl
#include "hue2rgb.glsl"

#ifndef FNC_HSL2RGB
#define FNC_HSL2RGB
vec3 hsl2rgb(const in vec3 hsl) {
    vec3 rgb = hue2rgb(hsl.x);
    float C = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
    return (rgb - 0.5) * C + hsl.z;
}
vec4 hsl2rgb(const in vec4 hsl) {
    return vec4(hsl2rgb(hsl.xyz), hsl.w);
}
#endif
```

### After (WESL):
```wgsl
import lygia::color::space::hue2rgb::hue2rgb;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a HSL color to linear RGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License
*/

fn hsl2rgb(hsl: vec3f) -> vec3f {
    let rgb = hue2rgb(hsl.x);
    let C = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
    return (rgb - 0.5) * C + hsl.z;
}

fn hsl2rgb4(hsl: vec4f) -> vec4f {
    return vec4(hsl2rgb(hsl.xyz), hsl.w);
}
```

---

## Resources

- [WESL Specification](https://wesl-lang.dev/spec/README)
- [WESL Imports](https://wesl-lang.dev/spec/Imports)
- [WESL Conditional Translation](https://wesl-lang.dev/spec/ConditionalTranslation)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Lygia WESL README](README_WESL.md)

---

## Tips for Batch Conversion

1. **Start simple**: Begin with utility functions (math, color space conversions)
2. **Test incrementally**: Verify each file compiles before moving to the next
3. **Watch for dependencies**: Some files import others - convert dependencies first
4. **Handle options**: Document which `#ifdef` features are enabled by default
5. **Maintain comments**: Preserve contributor, description, and license information
6. **Update tests**: Ensure corresponding test files are also converted/updated
