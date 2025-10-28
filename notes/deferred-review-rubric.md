# DEFERRED Files Review Rubric

**Purpose:** Determine which deferred files can now be converted given new test infrastructure capabilities.

**Date:** 2025-10-27
**Context:** We now have fragment shader testing with texture/sampler support and visual regression testing.

---

## New Capabilities

### Test Infrastructure (Implemented)
1. **Fragment shader tests** - `testFragment()` with derivatives support
2. **Input textures** - `inputTextures` parameter for testing texture operations
3. **Texture helpers** - gradientTexture, checkerboardTexture, radialGradientTexture, etc.
4. **Visual regression** - `toMatchImage()` for visual validation
5. **Sampler creation** - `createSampler()` for test setup

### Conversion Patterns Available
- Explicit texture/sampler parameters instead of SAMPLER_FNC macro
- Standard WGSL texture types: `texture_2d<f32>`, `sampler`
- Standard WGSL texture functions: `textureSample()`, `textureSampleLevel()`, etc.

---

## Decision Rubric

### ✅ CAN CONVERT NOW if ALL of:

1. **Straightforward texture operation**
   - Simple texture sampling (no complex macro composition)
   - Clear signature: `fn(texture_2d<f32>, sampler, coords, ...) -> result`
   - Doesn't require user-pluggable function selection

2. **Testable with current infrastructure**
   - Can create meaningful test with texture helpers
   - Visual output can be validated (either numeric or snapshot)
   - No external dependencies (scene, camera, etc.)

3. **No complex macro composition**
   - SAMPLER_FNC usage is ONLY for texture sampling
   - No FBM_FNC, DITHER_FNC, or other user-defined macros
   - No PLATFORM-specific code

4. **Clear conversion path**
   - Obvious how to replace macro with explicit parameters
   - No breaking changes to API
   - Examples available in similar converted files

### ⚠️ MEDIUM PRIORITY - Review carefully:

1. **Moderate complexity**
   - Multiple texture inputs or complex sampling patterns
   - Iterative/adaptive algorithms
   - May benefit from visual regression tests

2. **Limited dependencies**
   - Calls other functions but they're available in WESL
   - Platform-agnostic

### ❌ STILL DEFER if ANY of:

1. **User-pluggable functions required**
   - FBM_FNC with user-defined noise function
   - DITHER_FNC with user-defined dithering
   - No sensible default implementation

2. **Platform-specific**
   - PLATFORM_RPI, PLATFORM_WEBGL conditionals
   - OpenGL-specific workarounds

3. **Complex scene dependencies**
   - Requires full scene setup (camera matrices, light positions)
   - Multi-pass rendering dependencies
   - Global state assumptions

4. **Blocked by other deferred files**
   - Heavy interdependencies with deferred files
   - Would require converting many files together

5. **Uncertain API design**
   - Not clear how to expose without macros
   - Breaking changes would be significant
   - Needs design discussion

---

## Review Process

For each category:

1. **Read representative GLSL files** (2-3 samples per category)
2. **Check for:**
   - Macro usage patterns
   - Dependencies on other files
   - Complexity of implementation
   - Testability

3. **Classify files:**
   - HIGH PRIORITY: Can convert immediately
   - MEDIUM PRIORITY: Needs careful review
   - DEFER: Keep deferred (with reason)

4. **Document findings:**
   - Which files to un-defer
   - Conversion approach for each
   - Testing strategy

---

## Categories to Review

**Total Deferred:** 105 files

### High-Value Categories (Good test cases available)

1. **Filter/** (23 files) - Image processing operations
   - Edge detection (Sobel, etc.)
   - Blur operations (Gaussian, box blur, etc.)
   - Sharpening, median filters
   - **Testability:** HIGH - Can test with gradientTexture, checkerboardTexture
   - **Value:** HIGH - Common operations

2. **Sample/** (27 files) - Texture sampling utilities
   - Sampling patterns (bicubic, triplanar, etc.)
   - Wrapping modes (repeat, mirror, clamp)
   - Texture utilities (normal maps, bump maps)
   - **Testability:** HIGH - Direct texture operations
   - **Value:** HIGH - Foundational utilities

3. **Distort/** (6 files) - Image distortion effects
   - Barrel, pincushion distortion
   - Chromatic aberration
   - **Testability:** MEDIUM - Visual regression needed
   - **Value:** MEDIUM - Specialized effects

### Medium-Value Categories

4. **Morphological/** (8 files) - Image morphology
   - Erosion, dilation
   - Alpha fill, jump flood
   - **Testability:** MEDIUM - Iterative algorithms
   - **Value:** MEDIUM - Specialized operations

5. **Lighting/** (22 files) - Lighting calculations
   - Material properties (8 files - just texture sampling)
   - Advanced effects (SSAO, SSR - complex)
   - **Testability:** MIXED - Materials: high, Effects: low
   - **Value:** MEDIUM

### Lower Priority

6. **Generative/** (3 files) - FBM, Voronoi variants
   - Likely need user-pluggable functions
   - **Priority:** LOW

7. **Simulate/** (4 files) - Simulation algorithms
   - Complex multi-pass operations
   - **Priority:** LOW

8. **Space/** (3 files) - Spatial transforms with textures
   - **Priority:** LOW - Special cases

9. **Color/** (5 files), **Draw/** (2 files), **SDF/** (1 file)
   - Small categories, mixed complexity
   - **Priority:** CASE-BY-CASE

---

## Expected Outcome

**Optimistic estimate:** 30-50 files can be un-deferred
**Conservative estimate:** 15-25 files can be un-deferred immediately

**Focus areas:**
- Filter operations (edge, blur, sharpen)
- Simple sample operations (bicubic, triplanar)
- Material texture sampling (albedo, normal, roughness)
