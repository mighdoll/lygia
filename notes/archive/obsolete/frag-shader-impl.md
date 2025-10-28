# Fragment Shader Test Harness - wesl-debug Implementation

## Overview

Implement `testFragmentShader()` in the `wesl-debug` package to enable testing of fragment-shader-only functionality (primarily derivative functions like `fwidth()`, `dFdx()`, `dFdy()`).

This mirrors the existing `testComputeShader()` API but uses a render pipeline instead of compute pipeline.

## Target Package

**Package:** `wesl-debug`
**Location:** `/Users/lee/wesl/wesl-js/tools/packages/wesl-debug/`

## Implementation Tasks

### 1. Create `SimpleFragmentShader.ts`

**Location:** `/Users/lee/wesl/wesl-js/tools/packages/wesl-debug/src/SimpleFragmentShader.ts`

**Estimated lines:** ~250 lines

#### Function 1: `testFragmentShader()`

High-level API for running fragment shader tests.

```typescript
/**
 * Executes a fragment shader test and returns pixel values for validation.
 *
 * Similar to testComputeShader(), but runs a render pipeline to enable
 * derivative functions (fwidth, dFdx, dFdy) which are only available in
 * fragment shaders.
 *
 * Renders to a 1x1 RGBA32Float texture and reads back the pixel value.
 * A built-in fullscreen triangle vertex shader is provided automatically.
 *
 * @param projectDir - The project directory, used for resolving dependencies
 * @param gpu - GPU instance for device creation
 * @param fragmentSrc - WGSL/WESL fragment shader source code
 * @param resultFormat - Format for interpreting the result (default "f32")
 * @param conditions - Optional conditions for shader compilation
 * @returns Array of numbers from the rendered pixel (length depends on resultFormat)
 *
 * @example
 * ```typescript
 * const src = `
 *   @fragment
 *   fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
 *     let x = pos.x;
 *     let derivative = fwidth(x); // Only works in fragment shaders!
 *     return vec4f(derivative, 0.0, 0.0, 1.0);
 *   }
 * `;
 * const result = await testFragmentShader(projectDir, gpu, src, "f32");
 * // result[0] contains the red channel value (derivative)
 * ```
 */
export async function testFragmentShader(
  projectDir: string,
  gpu: GPU,
  fragmentSrc: string,
  resultFormat: WgslElementType = "f32",
  conditions: Record<string, boolean> = {}
): Promise<number[]> {
  const adapter = await gpu.requestAdapter();
  const device = await requestWeslDevice(adapter);

  try {
    // Generate combined vertex + fragment shader source
    const vertexSrc = generateFullscreenTriangleVertex();
    const completeSrc = vertexSrc + "\n\n" + fragmentSrc;

    // Compile shader (reuse compileShader from SimpleComputeShader)
    const params = { projectDir, device, src: completeSrc, conditions };
    const module = await compileShader(params);

    // Execute render pipeline and read back result
    const result = await runSimpleRenderPipeline(device, module, resultFormat);
    return result;
  } finally {
    device.destroy();
  }
}
```

**Implementation notes:**
- Reuse `compileShader()` from `SimpleComputeShader.ts` for dependency resolution
- Generate a simple vertex shader automatically (fullscreen triangle technique)
- Combine vertex + fragment source before compilation

#### Function 2: `generateFullscreenTriangleVertex()`

Helper function to generate built-in vertex shader.

```typescript
/**
 * Generates a simple vertex shader that renders a fullscreen triangle.
 *
 * Uses the fullscreen triangle technique: draws 3 vertices without vertex
 * buffers by computing positions from vertex_index. The triangle covers
 * the entire viewport with minimal overdraw.
 */
function generateFullscreenTriangleVertex(): string {
  return `
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  // Fullscreen triangle technique
  // idx 0: (-1, -1), idx 1: (3, -1), idx 2: (-1, 3)
  let uv = vec2f(f32((idx << 1) & 2), f32(idx & 2));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
`;
}
```

**Why fullscreen triangle?**
- No vertex buffers needed (positions computed from `vertex_index`)
- Covers entire viewport with just 3 vertices
- Standard technique for fullscreen post-processing effects

#### Function 3: `runSimpleRenderPipeline()`

Core function to execute render pipeline and read back texture.

```typescript
/**
 * Executes a render pipeline with the given shader module and returns pixel data.
 *
 * Creates a 1x1 RGBA32Float texture, renders to it, and reads back the pixel value.
 * The small texture size is sufficient for testing shader logic, and derivatives
 * still work due to 2x2 quad execution (helper invocations).
 *
 * @param device - GPU device to use
 * @param module - Compiled shader module with vertex and fragment shaders
 * @param resultFormat - Format for interpreting the result (determines how many components to extract)
 * @returns Array of numbers from the rendered pixel
 */
export async function runSimpleRenderPipeline(
  device: GPUDevice,
  module: GPUShaderModule,
  resultFormat: WgslElementType = "f32"
): Promise<number[]> {
  // Push error scopes to catch all types of errors
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");

  // Create 1x1 RGBA32Float texture as render target
  const texture = device.createTexture({
    label: "fragment-test-output",
    size: { width: 1, height: 1, depthOrArrayLayers: 1 },
    format: "rgba32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main"
    },
    fragment: {
      module,
      entryPoint: "fs_main", // TODO: Auto-detect entry point if needed
      targets: [{
        format: "rgba32float"
      }]
    },
    primitive: {
      topology: "triangle-list"
    }
  });

  // Render pass
  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: texture.createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      storeOp: "store"
    }]
  });

  renderPass.setPipeline(pipeline);
  renderPass.draw(3); // Draw 3 vertices (fullscreen triangle)
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);

  // Check for errors (pop in reverse order of push)
  const validationError = await device.popErrorScope();
  const oomError = await device.popErrorScope();
  const internalError = await device.popErrorScope();

  if (validationError) {
    throw new Error(`WebGPU validation error: ${validationError.message}`);
  }
  if (oomError) {
    throw new Error(`WebGPU out-of-memory error: ${oomError.message}`);
  }
  if (internalError) {
    throw new Error(`WebGPU internal error: ${internalError.message}`);
  }

  // Read back texture data using thimbleberry's withTextureCopy
  const data = await withTextureCopy(device, texture, (texData) => {
    // texData is Sliceable<number> containing RGBA values
    // Extract components based on resultFormat
    const numComponents = getNumComponents(resultFormat);
    return Array.from(texData.slice(0, numComponents));
  });

  // Clean up
  texture.destroy();

  return data;
}

/**
 * Helper to determine how many components to extract based on result format.
 */
function getNumComponents(format: WgslElementType): number {
  if (format === "f32" || format === "u32" || format === "i32") return 1;
  if (format === "vec2f" || format === "vec2u" || format === "vec2i") return 2;
  if (format === "vec3f" || format === "vec3u" || format === "vec3i") return 3;
  if (format === "vec4f" || format === "vec4u" || format === "vec4i") return 4;
  return 1; // default to scalar
}
```

**Key implementation details:**
- Use `rgba32float` texture format for full 32-bit float precision
- 1x1 texture is sufficient (derivatives still work via quad execution)
- Use `withTextureCopy()` from thimbleberry for texture readback
- Extract appropriate number of components based on `resultFormat`
- Follow same error handling pattern as `testComputeShader()`

### 2. Update Package Exports

**Location:** `/Users/lee/wesl/wesl-js/tools/packages/wesl-debug/src/index.ts`

Add exports for new functions:

```typescript
export {
  testFragmentShader,
  runSimpleRenderPipeline
} from './SimpleFragmentShader.ts';
```

**Note:** Don't export `generateFullscreenTriangleVertex()` - it's an internal helper.

### 3. Update Package Dependencies

**Location:** `/Users/lee/wesl/wesl-js/tools/packages/wesl-debug/package.json`

Verify `thimbleberry` is listed in dependencies (it should already be there):

```json
{
  "dependencies": {
    "thimbleberry": "^0.2.10",
    "wesl": "workspace:*"
  }
}
```

## Testing the Implementation

### Create Standalone Tests

**Location:** `/Users/lee/wesl/wesl-js/tools/packages/wesl-debug/test/SimpleFragmentShader.test.ts`

Create comprehensive tests that DO NOT depend on external shader libraries (like lygia).

```typescript
import { describe, test, expect, beforeAll } from "vitest";
import { testFragmentShader } from "../src/SimpleFragmentShader.js";

let gpu: GPU;

beforeAll(async () => {
  // Setup WebGPU for testing
  const webgpu = await import("webgpu");
  Object.assign(globalThis, webgpu.globals);
  gpu = webgpu.create([]);
});

describe("SimpleFragmentShader", () => {
  test("renders simple constant color", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.5, 0.25, 0.75, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "vec4f");

    expect(result).toHaveLength(4);
    expect(result[0]).toBeCloseTo(0.5, 5);
    expect(result[1]).toBeCloseTo(0.25, 5);
    expect(result[2]).toBeCloseTo(0.75, 5);
    expect(result[3]).toBeCloseTo(1.0, 5);
  });

  test("extracts single component for f32 format", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.123, 0.456, 0.789, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "f32");

    expect(result).toHaveLength(1);
    expect(result[0]).toBeCloseTo(0.123, 5);
  });

  test("extracts vec2f components", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.1, 0.2, 0.3, 0.4);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "vec2f");

    expect(result).toHaveLength(2);
    expect(result[0]).toBeCloseTo(0.1, 5);
    expect(result[1]).toBeCloseTo(0.2, 5);
  });

  test("extracts vec3f components", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.1, 0.2, 0.3, 0.4);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "vec3f");

    expect(result).toHaveLength(3);
    expect(result[0]).toBeCloseTo(0.1, 5);
    expect(result[1]).toBeCloseTo(0.2, 5);
    expect(result[2]).toBeCloseTo(0.3, 5);
  });

  test("uses builtin position varying", async () => {
    const src = `
      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        // For 1x1 texture, position should be (0.5, 0.5) at pixel center
        return vec4f(pos.x, pos.y, 0.0, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "vec2f");

    // Position at pixel center is (0.5, 0.5)
    expect(result[0]).toBeCloseTo(0.5, 1);
    expect(result[1]).toBeCloseTo(0.5, 1);
  });

  test("derivatives are available - fwidth", async () => {
    const src = `
      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let x = pos.x;
        let derivative = fwidth(x);
        return vec4f(derivative, 0.0, 0.0, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "f32");

    // Derivative should be non-zero (exact value depends on implementation)
    expect(result[0]).toBeGreaterThanOrEqual(0.0);
    expect(result[0]).toBeLessThan(10.0); // Sanity check
  });

  test("derivatives are available - dFdx", async () => {
    const src = `
      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let derivative = dFdx(pos.x);
        return vec4f(derivative, 0.0, 0.0, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "f32");

    // Should have a derivative value
    expect(result[0]).toBeDefined();
    expect(isFinite(result[0])).toBe(true);
  });

  test("derivatives are available - dFdy", async () => {
    const src = `
      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let derivative = dFdy(pos.y);
        return vec4f(derivative, 0.0, 0.0, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "f32");

    // Should have a derivative value
    expect(result[0]).toBeDefined();
    expect(isFinite(result[0])).toBe(true);
  });

  test("performs basic math operations", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        let a = 2.0;
        let b = 3.0;
        let result = a * b + 1.0; // Should be 7.0
        return vec4f(result, 0.0, 0.0, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "f32");

    expect(result[0]).toBeCloseTo(7.0, 5);
  });

  test("supports trigonometric functions", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        let result = sin(0.0); // Should be 0.0
        return vec4f(result, 0.0, 0.0, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "f32");

    expect(result[0]).toBeCloseTo(0.0, 5);
  });

  test("handles negative values correctly", async () => {
    const src = `
      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(-0.5, -1.0, -2.5, 1.0);
      }
    `;
    const result = await testFragmentShader(import.meta.url, gpu, src, "vec3f");

    expect(result[0]).toBeCloseTo(-0.5, 5);
    expect(result[1]).toBeCloseTo(-1.0, 5);
    expect(result[2]).toBeCloseTo(-2.5, 5);
  });
});
```

**Test coverage requirements:**
- ✅ Basic constant color rendering
- ✅ Component extraction (f32, vec2f, vec3f, vec4f)
- ✅ Builtin varyings (`@builtin(position)`)
- ✅ All derivative functions (fwidth, dFdx, dFdy)
- ✅ Basic math operations
- ✅ Negative values
- ✅ Trigonometric functions

### Run Tests

```bash
cd /Users/lee/wesl/wesl-js/tools/packages/wesl-debug
pnpm test
```

All tests should pass before considering the implementation complete.

## Technical Considerations

### Derivative Behavior in 1x1 Texture

**Question:** Do derivatives work with a 1x1 texture?

**Answer:** Yes! Fragment shaders execute in 2x2 quad groups ("helper invocations") even for single pixels. The helper invocations may fall outside the 1x1 viewport, but they still execute to provide derivative values. This is required by the WebGPU spec.

### Texture Format Precision

**RGBA32Float format:**
- 32-bit float per component (full IEEE 754 single precision)
- ~7 decimal digits of precision
- Identical to storage buffer precision
- Sufficient for test tolerances (typically `epsilon = 0.0001`)

### Entry Point Detection

**Current approach:** Hard-code `"fs_main"` as fragment shader entry point.

**Future enhancement:** Auto-detect entry point by scanning for `@fragment` attribute in shader source. This would require parsing the shader AST or using regex to find the fragment function name.

**For now:** Document that fragment shader entry point must be named `fs_main`.

### Error Handling

Follow the same error handling pattern as `testComputeShader()`:
- Push error scopes before operations
- Pop and check for validation/OOM/internal errors
- Throw descriptive errors
- Clean up resources (destroy texture/device) even on error

## Implementation Checklist

- [ ] Create `SimpleFragmentShader.ts` with three functions
  - [ ] `testFragmentShader()` - Main API
  - [ ] `generateFullscreenTriangleVertex()` - Helper
  - [ ] `runSimpleRenderPipeline()` - Core execution
- [ ] Update `index.ts` exports
- [ ] Verify `thimbleberry` dependency
- [ ] Create `SimpleFragmentShader.test.ts` with 10+ tests
- [ ] Run tests and verify all pass
- [ ] Update package documentation (if applicable)

## Success Criteria

1. ✅ All standalone tests pass
2. ✅ API mirrors `testComputeShader()` for consistency
3. ✅ Derivatives (fwidth, dFdx, dFdy) work correctly
4. ✅ Component extraction works for all formats (f32, vec2f, vec3f, vec4f)
5. ✅ No regressions in existing `testComputeShader()` tests
6. ✅ Code follows existing wesl-debug patterns and style

---

## Follow-up: Lygia Integration

**After** the wesl-debug implementation is complete and tested, the lygia shader library can create tests for derivative-dependent functions.

See [test-fragshader.md](test-fragshader.md) for the lygia-side integration plan, which includes:

1. Adding `testFragmentShader()` wrapper in `lygia/test/wesl/testUtil.ts`
2. Creating `lygia/test/wesl/derivatives.test.ts` with tests for 6 functions:
   - `math/aafloor.wesl` - Anti-aliased floor
   - `math/aafract.wesl` - Anti-aliased fract
   - `math/aastep.wesl` - Anti-aliased step
   - `math/aamirror.wesl` - Anti-aliased mirror
   - `math/fcos.wesl` - Fast cosine approximation
   - `filter/sharpen/adaptive.wesl` - Adaptive sharpening filter
3. Updating skipped tests in `lygia/test/wesl/functions.test.ts`

**Note:** Lygia integration should NOT begin until wesl-debug implementation is complete and its standalone tests are passing.
