# Fragment Shader Testing - Lygia Integration Plan

## Overview

This document describes how to integrate fragment shader testing into the Lygia shader library test suite, **after** the `testFragmentShader()` utility has been implemented in the `wesl-debug` package.

## Prerequisites

⚠️ **This integration depends on wesl-debug implementation being complete first!**

See [frag-shader-impl.md](frag-shader-impl.md) for the wesl-debug package implementation requirements. The wesl-debug implementation must:
- Include `testFragmentShader()` function
- Pass all standalone tests
- Be published/linked so lygia can use it

## Problem Statement

Lygia has **6 shader functions** that require derivatives (`fwidth`, `dFdx`, `dFdy`) which are only available in fragment shaders, not compute shaders. Current test infrastructure (`testComputeShader`) cannot test these functions, so they are currently skipped.

## Functions Requiring Fragment Shader Testing

Based on codebase analysis, these 6 functions need fragment shader tests:

1. **`math/aafloor.wesl`** - Anti-aliased floor (uses `fwidth()` on line 12)
2. **`math/aafract.wesl`** - Anti-aliased fract (uses `fwidth()` on line 11)
3. **`math/aastep.wesl`** - Anti-aliased step (uses `fwidth()` on line 7)
4. **`math/aamirror.wesl`** - Anti-aliased mirror (uses `dpdx()/dpdy()` on line 11)
5. **`math/fcos.wesl`** - Fast cosine approximation (uses `fwidth()` on line 13)
6. **`filter/sharpen/adaptive.wesl`** - Adaptive sharpening filter (uses `fwidth()` on line 78)

Currently these tests are skipped in `test/wesl/functions.test.ts` (see lines 9-22 and 24-37).

## Implementation Tasks

### 1. Add Fragment Shader Test Utility Wrapper

**Location:** `/Users/lee/wesl/lygia/test/wesl/testUtil.ts`

**Add function:**

```typescript
import { testFragmentShader as testFragmentShaderBase } from "wesl-debug";

/** utility function to test WGSL fragment shader */
export function testFragmentShader(
  src: string,
  elem: WgslElementType = "f32",
  conditions?: Record<string, boolean>,
) {
  if (!sharedGpu) {
    throw new Error("GPU not initialized. Call setupWebGPU() in beforeAll()");
  }
  return testFragmentShaderBase(import.meta.url, sharedGpu, src, elem, conditions);
}
```

**Purpose:**
- Wraps `testFragmentShader()` from wesl-debug
- Mirrors the existing `testShader()` wrapper pattern
- Automatically provides `import.meta.url` (project directory) and `sharedGpu`
- Simplifies test code by reducing boilerplate

### 2. Create Derivative Function Tests

**Location:** `/Users/lee/wesl/lygia/test/wesl/derivatives.test.ts` (new file)

**Purpose:** Dedicated test file for derivative-dependent functions.

**Test structure:**

```typescript
import { beforeAll, expect, test } from "vitest";
import { expectCloseTo, setupWebGPU, testFragmentShader } from "./testUtil.ts";

beforeAll(async () => {
  await setupWebGPU();
});

test("aafloor with derivatives", async () => {
  const src = `
    import lygia::math::aafloor::aafloor;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // Create varying value for derivatives to work
      // For 1x1 texture, pos.x will be ~0.5 at pixel center
      let x = pos.x / 10.0 + 2.5; // Gives ~2.55
      let result = aafloor(x);

      // Return result in red channel for validation
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  // Result in first component (red channel)
  // aafloor should produce something close to 2.0
  expectCloseTo([2.0], [result[0]], 0.1);
});

test("aafract with derivatives", async () => {
  const src = `
    import lygia::math::aafract::aafract;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 10.0 + 2.5;
      let result = aafract(x);
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  // aafract should produce fractional part, close to 0.5
  expectCloseTo([0.5], [result[0]], 0.1);
});

test("aastep with derivatives", async () => {
  const src = `
    import lygia::math::aastep::aastep;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let value = pos.x / 10.0; // ~0.05
      let threshold = 0.5;
      let result = aastep(threshold, value);
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  // aastep returns 0 or 1 with smooth transition
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
});

test("aamirror with derivatives", async () => {
  const src = `
    import lygia::math::aamirror::aamirror;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 10.0 + 0.5;
      let result = aamirror(x);
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  // aamirror should return a valid mirrored value
  expect(result[0]).toBeDefined();
  expect(isFinite(result[0])).toBe(true);
});

test("fcos with derivatives", async () => {
  const src = `
    import lygia::math::fcos::fcos;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = 0.0; // Test at x=0, cos(0) = 1.0
      let result = fcos(x);
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  // fcos(0) should be close to 1.0
  expectCloseTo([1.0], [result[0]], 0.1);
});

test("adaptive sharpen filter with derivatives", async () => {
  const src = `
    import lygia::filter::sharpen::adaptive::sharpenAdaptive;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // This function typically needs texture sampling
      // For now, just verify it compiles and runs
      // TODO: Implement proper test with texture input
      return vec4f(1.0, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  expect(result[0]).toBe(1.0);
});
```

**Testing strategy:**
- Each test imports the specific Lygia function
- Uses `@builtin(position)` to create varying values for derivatives
- Returns result in color channels (typically red channel for f32)
- Validates results with appropriate tolerance (usually 0.1)

**Note on test values:**
- For 1x1 texture, `@builtin(position)` gives approximately (0.5, 0.5) at pixel center
- Divide by 10 and add offsets to create specific test scenarios
- Anti-aliasing functions smooth edges, so expect approximate results

### 3. Update Skipped Tests in functions.test.ts

**Location:** `/Users/lee/wesl/lygia/test/wesl/functions.test.ts`

**Current state (lines 9-37):**

```typescript
test.skip("aafloor", async () => {
  const src = `
    import lygia::math::aafloor::aafloor;

    @compute @workgroup_size(1)
    fn foo() {
      let result = aafloor(2.7);
      test::results[0] = result;
    }
  `;
  const result = await testShader(src);
  expectCloseTo([2.0], result, 0.1);
});

test.skip("aafract", async () => {
  const src = `
    import lygia::math::aafract::aafract;

    @compute @workgroup_size(1)
    fn foo() {
      let result = aafract(2.7);
      test::results[0] = result;
    }
  `;
  const result = await testShader(src);
  expectCloseTo([0.7], result, 0.1);
});
```

**Two options:**

#### Option A: Remove skipped tests (recommended)
Since we're creating `derivatives.test.ts`, simply delete these skipped tests from `functions.test.ts` to avoid duplication.

#### Option B: Convert to fragment shader tests
Replace `.skip` with actual fragment shader tests:

```typescript
test("aafloor", async () => {
  const src = `
    import lygia::math::aafloor::aafloor;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 10.0 + 2.7;
      let result = aafloor(x);
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  expectCloseTo([2.0], [result[0]], 0.1);
});

test("aafract", async () => {
  const src = `
    import lygia::math::aafract::aafract;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let x = pos.x / 10.0 + 2.7;
      let result = aafract(x);
      return vec4f(result, 0.0, 0.0, 1.0);
    }
  `;
  const result = await testFragmentShader(src);
  expectCloseTo([0.7], [result[0]], 0.1);
});
```

**Recommendation:** Use Option A (remove skipped tests) and keep all derivative tests in the dedicated `derivatives.test.ts` file for better organization.

### 4. Update Test Import Statements

If converting tests in `functions.test.ts`, add import:

```typescript
import { expectCloseTo, setupWebGPU, testShader, testFragmentShader } from "./testUtil.ts";
```

If creating separate `derivatives.test.ts`, use:

```typescript
import { expectCloseTo, setupWebGPU, testFragmentShader } from "./testUtil.ts";
```

## Running Tests

### Run all Lygia tests:

```bash
cd /Users/lee/wesl/lygia
pnpm test
```

### Run only derivative tests:

```bash
pnpm test derivatives
```

### Expected results:
- All 6 derivative function tests should pass
- No regressions in existing compute shader tests
- Total test count increases from 184 to 190 (6 new tests)

## Test Validation Strategy

### Understanding Anti-Aliasing Functions

Anti-aliasing functions like `aafloor`, `aafract`, `aastep` smooth edges by using derivatives:

- **Without AA:** Sharp transition (e.g., floor(2.7) = 2.0)
- **With AA:** Smooth gradient near edges using fwidth()
- **Test tolerance:** Use 0.1 (same as existing skipped tests) because results depend on derivative values

### Common Issues to Watch For

1. **Constant values don't produce derivatives**
   - ❌ `let x = 2.7; aafloor(x)` - derivative is 0
   - ✅ `let x = pos.x; aafloor(x)` - derivative exists

2. **Component extraction**
   - Remember: `testFragmentShader(src)` returns array of numbers
   - First element is red channel: `result[0]`
   - For vec3f: extract first 3 elements: `[result[0], result[1], result[2]]`

3. **Pixel position in 1x1 texture**
   - `@builtin(position)` gives approximately (0.5, 0.5)
   - Scale and offset as needed for specific test values

## Success Criteria

1. ✅ All 6 derivative-dependent functions have working tests
2. ✅ Tests pass with existing tolerance (`epsilon = 0.1`)
3. ✅ No regressions in existing compute shader tests (178 tests still pass)
4. ✅ Total test count: 184 → 190 (6 new passing tests, 6 removed skipped tests)
5. ✅ Test execution time remains reasonable (<5s total for fragment shader tests)

## Files Summary

### New Files
- `/Users/lee/wesl/lygia/test/wesl/derivatives.test.ts` (~150 lines)

### Modified Files
- `/Users/lee/wesl/lygia/test/wesl/testUtil.ts` (+13 lines) - Add `testFragmentShader()` wrapper
- `/Users/lee/wesl/lygia/test/wesl/functions.test.ts` (-28 lines if removing skipped tests)

### Dependencies
- `wesl-debug` package with `testFragmentShader()` implemented (see [frag-shader-impl.md](frag-shader-impl.md))
- No new package dependencies required (wesl-debug already linked via pnpm workspace)

## Future Enhancements (Out of Scope)

Ideas for future improvements:

1. **Larger render targets for spatial testing**
   - Test derivatives across multiple pixels
   - Verify smooth gradients visually

2. **Texture sampling tests**
   - Some filters require texture inputs
   - Would need to create test textures

3. **Visual debugging mode**
   - Save rendered textures as images
   - Inspect anti-aliasing visually

4. **Performance benchmarks**
   - Compare fragment vs compute shader performance
   - Optimize hot paths

## References

- **wesl-debug implementation:** [frag-shader-impl.md](frag-shader-impl.md)
- **Existing compute shader tests:** `/Users/lee/wesl/lygia/test/wesl/functions.test.ts`
- **Test utility helpers:** `/Users/lee/wesl/lygia/test/wesl/testUtil.ts`
- **Function tracking:** [GlslToConvert.md](GlslToConvert.md) (see "Functions Requiring Fragment Shader Testing" section)
