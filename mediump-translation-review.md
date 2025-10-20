# Mediump Translation Review - LYGIA GLSL to WESL

## Executive Summary

This document summarizes the findings from a comprehensive review of mediump precision handling in LYGIA's GLSL to WESL translations.

### Key Findings

1. **GLSLtoWESL.md has been updated** with comprehensive mediump translation guidelines
2. **saturateMediump pattern is correctly implemented** in most translated files
3. **Several critical issues found** that need to be addressed
4. **No f16 usage detected** (good - following the recommended strategy)

## Translation Strategy (Now Documented in GLSLtoWESL.md)

✅ **Use f32 throughout** - No f16 or enable f16
✅ **Keep saturateMediump pattern** - Clamps to 65504.0 on mobile
✅ **Preserve mathematical workarounds** - Like Lagrange's identity in GGX
✅ **Use @if(TARGET_MOBILE)** - For platform-specific optimizations

## Critical Issues Found

### 1. Missing WESL Conversions

| File | Priority | Issue |
|------|----------|-------|
| **kelemen.wesl** | CRITICAL | File doesn't exist - GLSL version uses saturateMediump |
| **ashikhmin.wesl** | HIGH | Contains fp16 precision workaround (0.0078125 constant) |
| **charlie.wesl** | HIGH | Contains fp16 precision workaround |
| **clampNoV.wesl** | HIGH | Division by zero prevention (MIN_N_DOT_V = 1e-4) |
| **beckmann.wesl** | MEDIUM | Division safety with NoH clamping |
| **envBRDFApprox.wesl** | MEDIUM | Contains carefully calibrated approximation constants |

### 2. Platform Conditional Issues in Existing Files

| File | Issue | Status |
|------|-------|--------|
| **ggx.wesl** | ✅ FIXED | Added @if(TARGET_MOBILE) conditional in GGXPrecise() - Lagrange's identity now only used on mobile |
| **cookTorrance.wesl** | ✅ FIXED | Added @if(PLATFORM_RPI) conditional to use smithGGXCorrelated_Fast on RPI |

### 3. Correctly Implemented Files

✅ **saturateMediump.wesl** - All 4 overloads with proper TARGET_MOBILE conditionals
✅ **smithGGXCorrelated.wesl** - Perfect translation with saturateMediump
✅ **snoise.wesl** - Precision comments and Taylor approximations preserved
✅ **taylorInvSqrt.wesl** - Fast inverse square root correctly implemented

## Precision-Sensitive Patterns Found

### 1. fp16 Magic Constants
```glsl
// In ashikhmin.glsl and charlie.glsl:
float sin2h = max(1.0 - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
```
**Critical:** This constant ensures values remain positive in half-precision

### 2. Lagrange's Identity for Numerical Stability
```glsl
// In ggx.glsl - avoids floating point cancellation:
vec3 NxH = cross(N, H);
float oneMinusNoHSquared = dot(NxH, NxH); // Instead of 1.0 - NoH * NoH
```
**Status:** Preserved in ggx.wesl but conditional was removed

### 3. Division Safety Clamping
```glsl
// Various files:
return saturateMediump(0.25 / (LoH * LoH));  // kelemen
let v = 0.5 / (lambdaV + lambdaL);            // smithGGX
```

### 4. Minimum Value Clamping
```glsl
#define MIN_N_DOT_V 1e-4        // clampNoV.glsl
#define MIN_PERCEPTUAL_ROUGHNESS 0.045  // perceptual2linearRoughness.glsl
```

## Recommendations

### Immediate Actions Required

1. **Create kelemen.wesl**:
```wgsl
import lygia::math::saturateMediump::saturateMediump;

fn kelemen(LoH: f32) -> f32 {
    return saturateMediump(0.25 / (LoH * LoH));
}
```

2. **Fix ggx.wesl platform conditional**:
   - Add @if(TARGET_MOBILE) around Lagrange's identity path
   - Provide standard computation for desktop platforms

3. **Consider cookTorrance.wesl optimization**:
   - Add conditional for low-end devices to use smithGGXCorrelated_Fast

### Priority Conversions Needed

**High Priority** (contain critical precision workarounds):
- ashikhmin.glsl → ashikhmin.wesl (fp16 workaround)
- charlie.glsl → charlie.wesl (fp16 workaround)
- clampNoV.glsl → clampNoV.wesl (division safety)

**Medium Priority** (contain calibrated constants):
- beckmann.glsl → beckmann.wesl
- envBRDFApprox.glsl → envBRDFApprox.wesl
- perceptual2linearRoughness.glsl → perceptual2linearRoughness.wesl

## Verification Checklist

When reviewing or creating new mediump translations:

- [ ] Use f32 for all floating-point types
- [ ] Import saturateMediump for overflow-prone calculations
- [ ] Preserve mathematical workarounds and precision tricks
- [ ] Use @if(TARGET_MOBILE) not PLATFORM_WEBGL or PLATFORM_RPI
- [ ] Keep precision-related comments
- [ ] Preserve magic constants exactly (e.g., 0.0078125 for fp16)
- [ ] Test division operations for potential infinity
- [ ] Never use enable f16 or f16 types

## Summary

The mediump translation strategy for LYGIA is well-defined and mostly well-implemented. The main issues are:
1. A few missing conversions (kelemen being most critical)
2. ~~Some platform conditionals that got lost in translation (ggx.wesl)~~ ✅ FIXED
3. Several precision-sensitive functions that haven't been converted yet

The recommended f32-only approach with saturateMediump clamping is the correct strategy for maximum compatibility.

## Fixes Applied (2025-10-20)

### ✅ Fixed Platform Conditionals

**ggx.wesl** (lighting/common/ggx.wesl:15-45)
- Added `@if(TARGET_MOBILE)` conditional within `GGXPrecise()` function
- Lagrange's identity optimization now only used on mobile platforms (matching GLSL behavior)
- Desktop platforms use standard computation path
- Tests passing: test/wesl/lighting-common.test.ts

**cookTorrance.wesl** (lighting/specular/cookTorrance.wesl:11-15)
- Added `@if(PLATFORM_RPI)` conditional for visibility term selection
- Raspberry Pi now uses `smithGGXCorrelated_Fast()` for better performance
- Other platforms use full `smithGGXCorrelated()` implementation
- Tests passing: test/wesl/lighting-misc.test.ts

Both fixes ensure the WESL translations now match the GLSL platform-specific optimizations.