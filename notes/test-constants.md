# Proposal: Add `constants` Support to wesl-debug Test APIs

## Summary
Add support for the `constants` parameter in `testComputeShader()` and `testFragmentShader()` to enable testing WESL code that uses the `constants::` namespace.

## Background
WESL's linker supports a `constants` parameter in `LinkParams` that allows host applications to inject constant values into shaders via the magic `constants::` namespace:

```typescript
// From wesl LinkParams
constants?: Record<string, string | number>;
```

This enables patterns like:

```wgsl
// shader.wesl
@if(USE_CUSTOM_VALUE)
import constants::CUSTOM_VALUE;

fn compute() -> f32 {
    @if(USE_CUSTOM_VALUE)
    return CUSTOM_VALUE * 2.0;
    @else
    return 1.0;
}
```

```typescript
// application code
const shader = await link(src, {
    conditions: { USE_CUSTOM_VALUE: true },
    constants: { CUSTOM_VALUE: 42.0 }
});
```

## Problem
Currently, `testComputeShader()` and `testFragmentShader()` support the `conditions` parameter but not `constants`. This makes it impossible to test WESL code that relies on the `constants::` namespace.

## Proposed Change

### 1. Update Type Definitions

```typescript
// src/TestComputeShader.ts
export interface ComputeTestParams {
  src: string;
  projectDir: string;
  device: GPUDevice;
  resultFormat?: WgslElementType;
  conditions?: Record<string, boolean>;
  constants?: Record<string, string | number>;  // ← ADD THIS
}

// src/TestFragmentShader.ts
export interface FragmentTestParams {
  src: string;
  projectDir: string;
  device: GPUDevice;
  textureFormat?: GPUTextureFormat;
  size?: [number, number];
  conditions?: Record<string, boolean>;
  constants?: Record<string, string | number>;  // ← ADD THIS
}
```

### 2. Pass Through to Linker

The implementation should simply pass the `constants` parameter through to the WESL linker:

```typescript
// Wherever link() is called in the test functions
const linked = await link(weslSrc, {
  conditions: params.conditions,
  constants: params.constants,  // ← ADD THIS
  // ... other params
});
```

## Example Usage

### Testing with constants

```typescript
test("custom center point", async () => {
  const src = `
    import constants::CENTER;

    @compute @workgroup_size(1)
    fn main() {
      let result = computeDistance(CENTER);
      test::results[0] = result;
    }
  `;

  const result = await testComputeShader({
    src,
    device,
    projectDir,
    constants: { CENTER: "vec2f(0.5, 0.5)" }
  });

  expect(result[0]).toBeCloseTo(0.707);
});
```

### Testing conditional code with constants

```typescript
test("with custom matrix", async () => {
  const src = `
    @if(USE_PROJECTION)
    import constants::PROJECTION_MATRIX;

    @compute @workgroup_size(1)
    fn main() {
      @if(USE_PROJECTION)
      let result = PROJECTION_MATRIX * vec4f(1.0, 2.0, 3.0, 1.0);
      @else
      let result = vec4f(1.0, 2.0, 3.0, 1.0);

      test::results[0] = result;
    }
  `;

  const result = await testComputeShader({
    src,
    device,
    projectDir,
    conditions: { USE_PROJECTION: true },
    constants: {
      PROJECTION_MATRIX: `mat4x4f(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
      )`
    }
  });

  expectCloseTo([1.0, 2.0, 3.0, 1.0], result);
});
```

## Implementation Notes

1. **Backward Compatibility**: The change is fully backward compatible since `constants` is optional
2. **Type Safety**: Constants are `string | number` - strings for complex types (vectors, matrices), numbers for scalars
3. **No Validation Needed**: The WESL linker already handles validation and error reporting for constants
4. **Minimal Change**: This is just plumbing - all the hard work is already done in the linker

## Benefits

- Enables testing of WESL libraries that use the `constants::` namespace
- Matches the full capabilities of the WESL linker
- Maintains consistency between test APIs and production linking APIs
- Essential for testing conditional code paths that depend on host-provided values

## Alternative Approaches Considered

**None**. This is the only straightforward way to support this WESL feature in tests.

## Related Files

The change likely touches:
- `src/TestComputeShader.ts` - Add to interface and pass to linker
- `src/TestFragmentShader.ts` - Add to interface and pass to linker
- `src/CompileShader.ts` - Possibly needs the same update if used independently
- `dist/index.d.ts` - Will be regenerated with new types

## Testing

After implementing, test with both simple and complex constants:

```typescript
// Scalar constant
constants: { PI: 3.14159 }

// Vector constant
constants: { CENTER: "vec2f(0.5, 0.5)" }

// Matrix constant
constants: { TRANSFORM: "mat3x3f(...)" }

// Multiple constants
constants: {
  WIDTH: 800,
  HEIGHT: 600,
  ASPECT_RATIO: "f32(800.0 / 600.0)"
}
```
