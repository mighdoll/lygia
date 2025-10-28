# Distribution Testing for Random Functions

## Overview

This document describes the approach for adding statistical distribution tests to LYGIA's random number generator functions. These tests validate that random functions produce uniform distributions with correct statistical properties.

## Motivation

Current tests validate:
- ✅ Determinism (same input → same output)
- ✅ Range bounds (output within [0,1] or [-1,1])
- ✅ Hash properties (avalanche effect, component independence)
- ❌ **Distribution uniformity** (NOT tested yet)
- ❌ **Statistical properties** (mean, variance)

Distribution tests catch:
- Systematic bias (mean shifted from expected value)
- Limited range (values clustered in subset of expected range)
- Clustering (non-uniform distribution across range)
- Implementation bugs that pass basic range checks but produce poor distributions

## New wesl-debug Capabilities

As of wesl-debug v0.6.13+, `testComputeShader()` supports configurable buffer sizes:

```typescript
const result = await testComputeShader({
  projectDir: import.meta.url,
  device,
  src: shaderSource,
  resultFormat: "f32",
  size: 4096  // 4096 bytes = 1024 f32 values
});
```

Previously, we were limited to 16 bytes (4 float values). Now we can collect hundreds or thousands of samples for statistical analysis.

## Statistical Tests

### 1. Mean Test (Detects Systematic Bias)

**Purpose:** Verify the average value is close to the expected center of the range.

**Expected Values:**
- `random()` functions [0, 1]: mean ≈ 0.5
- `srandom()` functions [-1, 1]: mean ≈ 0.0

**Threshold:** ±0.05 from expected (accounts for random variance with typical sample sizes)

**Why this matters:** A shifted mean indicates systematic bias in the hash function or incorrect normalization.

### 2. Bucket Distribution Test (Detects Clustering)

**Purpose:** Verify values spread evenly across the entire range.

**Method:**
- Divide range into 10 equal buckets
- Count samples in each bucket
- Each bucket should contain ~10% of samples

**Threshold:** Each bucket should have 8-12% of samples (10% ± 2%)

**Why this matters:** Good random functions should fill all buckets roughly equally. Clustering indicates patterns or periodicity in the generator.

### 3. Min/Max Coverage Test (Detects Limited Range)

**Purpose:** Verify the function reaches near the edges of its declared range.

**Method:**
- With sufficient samples, min should be < 0.1 (for [0,1] range)
- With sufficient samples, max should be > 0.9 (for [0,1] range)

**Why this matters:** Some poor hash functions only use part of their range despite claiming [0,1] output.

## Implementation Approach

### Option A: GPU Loop (Recommended for Initial Implementation)

Collect multiple samples in a single GPU workgroup using a loop:

```wgsl
@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < 1024u; i++) {
    let sample = random(f32(i));
    test::results[i] = sample;
  }
}
```

**Pros:**
- Works with current test infrastructure
- Simple to implement
- Returns all samples for analysis in JavaScript

**Cons:**
- Sequential execution (but still very fast)
- Requires larger buffer size parameter

## Helper Functions to Add

### 1. `testDistribution()` - Collect Samples

Add to `test/wesl/testUtil.ts`:

```typescript
// Add this import at the top of the file
import { elementStride } from "wesl-debug";

/**
 * Test distribution properties of a random function.
 * Collects multiple samples and returns them for statistical analysis.
 *
 * @param src - WESL shader source that writes samples to test::results
 * @param sampleCount - Number of samples to collect
 * @param elem - Element type (default "f32")
 * @param constants - Constants to pass via constants:: namespace (e.g., SAMPLE_COUNT)
 * @returns Array of sample values
 */
export async function testDistribution(
  src: string,
  sampleCount: number,
  elem: WgslElementType = "f32",
  constants?: Record<string, string | number>
): Promise<number[]> {
  const device = await getGPUDevice();
  const bufferSize = sampleCount * elementStride(elem);

  return testComputeShader({
    projectDir,
    device,
    src,
    resultFormat: elem,
    size: bufferSize,
    constants
  });
}
```

### 2. `expectDistribution()` - Validate Statistics

Add to `test/wesl/testUtil.ts`:

```typescript
/**
 * Validate that samples follow a uniform distribution.
 *
 * @param samples - Array of sample values
 * @param range - Expected [min, max] range
 * @param options - Test options (meanTolerance, bucketTolerance)
 */
export function expectDistribution(
  samples: number[],
  range: [number, number],
  options: {
    meanTolerance?: number;
    bucketTolerance?: number;
  } = {}
): void {
  const { meanTolerance = 0.05, bucketTolerance = 0.02 } = options;
  const [min, max] = range;
  const expectedMean = (min + max) / 2;

  // Test 1: Mean
  const actualMean = samples.reduce((sum, v) => sum + v, 0) / samples.length;
  const meanDiff = Math.abs(actualMean - expectedMean);
  if (meanDiff > meanTolerance) {
    expect.fail(
      `Mean ${actualMean.toFixed(4)} differs from expected ${expectedMean} by ${meanDiff.toFixed(4)} (threshold: ${meanTolerance})`
    );
  }

  // Test 2: Bucket distribution
  const bucketCount = 10;
  const buckets = new Array(bucketCount).fill(0);
  const bucketWidth = (max - min) / bucketCount;

  for (const value of samples) {
    const bucketIndex = Math.min(
      Math.floor((value - min) / bucketWidth),
      bucketCount - 1
    );
    buckets[bucketIndex]++;
  }

  const expectedPerBucket = samples.length / bucketCount;
  const expectedRatio = 1.0 / bucketCount; // 0.1 for 10 buckets

  for (let i = 0; i < bucketCount; i++) {
    const ratio = buckets[i] / samples.length;
    const diff = Math.abs(ratio - expectedRatio);

    if (diff > bucketTolerance) {
      const bucketRange = [
        (min + i * bucketWidth).toFixed(2),
        (min + (i + 1) * bucketWidth).toFixed(2)
      ];
      expect.fail(
        `Bucket ${i} [${bucketRange[0]}, ${bucketRange[1]}) has ${(ratio * 100).toFixed(1)}% of samples (expected ${(expectedRatio * 100)}% ± ${(bucketTolerance * 100)}%)\n` +
        `Distribution: ${buckets.map(b => ((b / samples.length) * 100).toFixed(1) + '%').join(', ')}`
      );
    }
  }
}
```

### 3. Alternative: `expectMean()` - Simple Mean Test

For cases where you only want to test the mean:

```typescript
/**
 * Validate that samples have the expected mean.
 *
 * @param samples - Array of sample values
 * @param expectedMean - Expected mean value
 * @param tolerance - Acceptable deviation (default 0.05)
 */
export function expectMean(
  samples: number[],
  expectedMean: number,
  tolerance = 0.05
): void {
  const actualMean = samples.reduce((sum, v) => sum + v, 0) / samples.length;
  const diff = Math.abs(actualMean - expectedMean);

  if (diff > tolerance) {
    expect.fail(
      `Mean ${actualMean.toFixed(4)} differs from expected ${expectedMean} by ${diff.toFixed(4)} (threshold: ${tolerance})`
    );
  }
}
```

## Test Patterns

### Pattern 1: Single-Value Random Function

Test `random(f32)` which returns a single float [0, 1]:

```typescript
test("random - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import lygia::generative::random::random;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < ${sampleCount}u; i++) {
        let sample = random(f32(i));
        test::results[i] = sample;
      }
    }
  `;

  const samples = await testDistribution(src, sampleCount);
  expectDistribution(samples, [0.0, 1.0]);
});
```

### Pattern 2: Vector Random Function (Single Component)

Test `random2(vec2f)` which returns a single float [0, 1]:

```typescript
test("random2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import lygia::generative::random::random2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < ${sampleCount}u; i++) {
        // Vary input across 2D space
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        let sample = random2(vec2f(x, y));
        test::results[i] = sample;
      }
    }
  `;

  const samples = await testDistribution(src, sampleCount);
  expectDistribution(samples, [0.0, 1.0]);
});
```

### Pattern 3: Multi-Component Random Function

Test `random22(vec2f)` which returns `vec2f` [0, 1]:

```typescript
test("random22 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import lygia::generative::random::random22;

    @compute @workgroup_size(1)
    fn main() {
      // We can only test one component at a time with flat array
      for (var i = 0u; i < ${sampleCount}u; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        let sample = random22(vec2f(x, y));
        test::results[i] = sample.x; // Test x component
      }
    }
  `;

  const samples = await testDistribution(src, sampleCount);
  expectDistribution(samples, [0.0, 1.0]);

  // Note: Could add separate test for .y component if desired
});
```

### Pattern 4: Signed Random Function

Test `srandom(f32)` which returns a single float [-1, 1]:

```typescript
test("srandom - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import lygia::generative::srandom::srandom;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < ${sampleCount}u; i++) {
        let sample = srandom(f32(i));
        test::results[i] = sample;
      }
    }
  `;

  const samples = await testDistribution(src, sampleCount);
  expectDistribution(samples, [-1.0, 1.0]); // mean should be ~0.0
});
```

## Functions to Test

### High Priority (Core Uniform Random)

These are the most commonly used and should have distribution tests:

1. **`random(f32)` → f32 [0, 1]**
   - Sample count: 1024
   - Most basic random function

2. **`random2(vec2f)` → f32 [0, 1]**
   - Sample count: 512
   - 2D hash function

3. **`random3(vec3f)` → f32 [0, 1]**
   - Sample count: 512
   - 3D hash function

4. **`srandom(f32)` → f32 [-1, 1]**
   - Sample count: 1024
   - Signed version

5. **`srandom2(vec2f)` → f32 [-1, 1]**
   - Sample count: 512
   - Signed 2D version

### Medium Priority (Vector-Valued Random)

Test one component of vector outputs:

6. **`random22(vec2f)` → vec2f [0, 1]**
   - Sample count: 512
   - Test x component

7. **`random33(vec3f)` → vec3f [0, 1]**
   - Sample count: 512
   - Test x component

8. **`srandom22(vec2f)` → vec2f [-1, 1]**
   - Sample count: 512
   - Test x component

### Lower Priority (Skip for Now)

- `random4`, `random21`, `random23`, etc. - Less commonly used
- `random41`, `random42`, `random43`, `random44` - Already have avalanche tests
- Noise functions (`cnoise`, `snoise`, etc.) - Designed for smoothness, not uniformity
- `worley` functions - Distance-based, not uniform random

## Sample Sizes

**Recommended sample counts:**

| Function Type | Sample Count | Buffer Size | Rationale |
|---------------|--------------|-------------|-----------|
| Scalar input (f32) | 1024 | 4096 bytes | Simple sequence, need more samples |
| Vector input (vec2f) | 512 | 2048 bytes | 2D space provides more variation |
| Vector input (vec3f) | 512 | 2048 bytes | 3D space provides even more variation |

**Why not more samples?**
- 512-1024 samples gives good statistical confidence for our tolerance thresholds
- Keeps test execution fast
- Buffer sizes remain reasonable (2-4 KB)
- Diminishing returns beyond this point

**Statistical confidence:**
- With 1024 samples and 10 buckets, expected ~102 per bucket
- Tolerance of ±20 allows 82-122 per bucket
- This catches significant distribution problems while allowing natural variance

## Tolerance Thresholds

### Mean Tolerance: ±0.05

**For [0, 1] range:**
- Expected mean: 0.5
- Acceptable range: [0.45, 0.55]

**For [-1, 1] range:**
- Expected mean: 0.0
- Acceptable range: [-0.05, 0.05]

**Rationale:**
- 10% deviation from expected is noticeable
- Still allows for random variance
- Would catch systematic bias

### Bucket Tolerance: ±2%

**For 10 buckets:**
- Expected per bucket: 10%
- Acceptable range: [8%, 12%]

**Rationale:**
- 20% relative deviation (2% out of 10%) is noticeable
- With 1024 samples: allows 82-122 per bucket (expected ~102)
- Catches clustering while allowing natural variance

## Example Test Output

### Successful Test:
```
✓ random - distribution (23ms)
```

### Failed Test (Systematic Bias):
```
✗ random - distribution (24ms)
  AssertionError: Mean 0.5847 differs from expected 0.5 by 0.0847 (threshold: 0.05)
```

### Failed Test (Clustering):
```
✗ random - distribution (25ms)
  AssertionError: Bucket 3 [0.30, 0.40) has 15.2% of samples (expected 10% ± 2%)
  Distribution: 9.8%, 10.1%, 9.5%, 15.2%, 8.9%, 10.3%, 8.7%, 9.2%, 10.4%, 7.9%
```

## Test Consolidation Strategy

Once distribution tests are in place, we can **consolidate and simplify** existing tests. The distribution test validates multiple properties at once, so many individual checks become redundant.

### What Distribution Tests Validate

A single distribution test validates:
1. ✅ **Range bounds** - All 1024 samples must be within [0,1] or [-1,1] (implicitly checked)
2. ✅ **Mean** - Average value is correct (no systematic bias)
3. ✅ **Uniformity** - Values spread evenly (no clustering)
4. ✅ **Coverage** - Min/max near edges (full range usage)

### Current Test Structure (BEFORE)

```typescript
test("random", async () => {
  const src = `...`;
  const result = await testCompute(src, "vec4f");

  // Test 1: Determinism
  expectCloseTo([result[0]], [result[1]]);

  // Test 2: Different inputs → different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);

  // Test 3: Range bounds ← REDUNDANT with distribution test
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);

  // Test 4: Exact value (GLSL parity)
  expectCloseTo([0.7629680633544922], [result[0]]);
});
```

### Recommended Test Structure (AFTER)

```typescript
// Test 1: Determinism + Exact Value (GLSL parity)
test("random", async () => {
  const src = `
    import lygia::generative::random::random;
    @compute @workgroup_size(1)
    fn main() {
      let r1 = random(1.0);
      let r2 = random(1.0); // Same input
      test::results[0] = vec4f(r1, r2, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Determinism: same input → same output
  expectCloseTo([result[0]], [result[1]]);

  // GLSL parity: verify exact output value
  expectCloseTo([0.7629680633544922], [result[0]]);
});

// Test 2: Distribution (validates range, mean, uniformity, coverage)
test("random - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random;
    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        test::results[i] = random(f32(i));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount
  });
  expectDistribution(samples, [0.0, 1.0]);
});
```

### What to Keep vs Remove

| Test Check | Keep? | Rationale |
|------------|-------|-----------|
| **Determinism** (same input → same output) | ✅ KEEP | Not validated by distribution test; important property |
| **Exact value** (GLSL parity regression test) | ✅ KEEP | Critical for ensuring GLSL↔WESL parity |
| **Range bounds** (value in [0,1]) | ❌ REMOVE | Redundant - distribution test checks 1024 samples |
| **Different inputs** (r1 ≠ r2 for different inputs) | 🤔 OPTIONAL | Somewhat redundant - distribution test checks 1024 different inputs |
| **Distribution** (new) | ✅ ADD | New high-value test |

### Consolidation Examples

#### Example 1: `random` Function

**BEFORE (5 checks, 2 samples):**
```typescript
test("random", async () => {
  const result = await testCompute(src, "vec4f");
  expectCloseTo([result[0]], [result[1]]);  // Determinism
  expect(result[0]).not.toBeCloseTo(result[2], 1);  // Different inputs
  expect(result[0]).toBeGreaterThanOrEqual(0.0);  // Range min
  expect(result[0]).toBeLessThanOrEqual(1.0);  // Range max
  expectCloseTo([0.7629680633544922], [result[0]]);  // Exact value
});
```

**AFTER (2 separate tests, 2 + 1024 samples):**
```typescript
// Test 1: Determinism + Exact Value
test("random", async () => {
  const result = await testCompute(src, "vec4f");
  expectCloseTo([result[0]], [result[1]]);  // Determinism
  expectCloseTo([0.7629680633544922], [result[0]]);  // Exact value (GLSL parity)
});

// Test 2: Distribution (replaces range bounds + uniformity)
test("random - distribution", async () => {
  const samples = await testDistribution(src, 1024, "f32", { SAMPLE_COUNT: 1024 });
  expectDistribution(samples, [0.0, 1.0]);
});
```

**Result:**
- Removed: 3 redundant checks (range min/max, different inputs)
- Added: 1 comprehensive distribution test
- Kept: Determinism + exact value (important properties)

#### Example 2: `random42` (Hash Properties)

**BEFORE (Multiple separate tests):**
```typescript
test("random42 - hash properties", async () => {
  // Determinism test
  const determinism = await testCompute(src1, "vec4f");
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism, 5);

  // Range test
  const result = await testCompute(src2, "vec4f");
  result.forEach(v => {
    expect(v).toBeGreaterThanOrEqual(0.0);  // ← REDUNDANT
    expect(v).toBeLessThanOrEqual(1.0);  // ← REDUNDANT
  });

  // Independence test
  const allSame = result.every(v => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Avalanche test
  const avalanche = await testCompute(src3, "vec4f");
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.03);
});
```

**AFTER (Consolidate into 2 tests):**
```typescript
// Test 1: Determinism + Avalanche
test("random42 - hash properties", async () => {
  // Determinism
  const determinism = await testCompute(src1, "vec4f");
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism, 5);

  // Avalanche effect
  const avalanche = await testCompute(src3, "vec4f");
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.03);
});

// Test 2: Distribution (replaces range + independence checks)
test("random42 - distribution (x component)", async () => {
  const samples = await testDistribution(src, 512, "f32", { SAMPLE_COUNT: 512 });
  expectDistribution(samples, [0.0, 1.0]);
});
```

**Result:**
- Removed: Range bounds check (redundant)
- Kept: Determinism, avalanche, independence tests (hash-specific properties)
- Added: Distribution test (validates range + uniformity for x component)

### Functions Without Distribution Tests

Some functions don't need distribution tests:

| Function Type | Example | Distribution Test? | Rationale |
|---------------|---------|-------------------|-----------|
| **Noise functions** | `cnoise`, `snoise` | ❌ NO | Designed for smoothness, not uniformity |
| **Worley functions** | `worley`, `worley22` | ❌ NO | Distance-based, not uniform random |
| **Wavelet functions** | `wavelet`, `waveletScaled` | ❌ NO | Structured patterns, not uniform |
| **Periodic functions** | `pnoise` | ❌ NO | Periodicity is more important than distribution |

For these functions, keep existing tests focused on their specific properties:
- Noise: determinism, continuity, periodicity
- Worley: F1 ≤ F2 property, determinism
- Wavelet: phase/scale behavior

### Consolidation Checklist

When adding distribution tests, follow this checklist for each function:

- [ ] **Add** distribution test (new)
- [ ] **Keep** determinism check (if it exists)
- [ ] **Keep** exact value / GLSL parity check (if it exists)
- [ ] **Remove** range bounds checks (toBeGreaterThanOrEqual / toBeLessThanOrEqual)
- [ ] **Consider removing** "different inputs" check (may be redundant)
- [ ] **Keep** function-specific properties (e.g., avalanche effect for hashes)

### Migration Strategy

1. **Add distribution tests first** (don't remove anything yet)
2. **Run all tests** to ensure distribution tests work correctly
3. **Remove redundant checks** one function at a time
4. **Verify tests still pass** after each consolidation

This conservative approach ensures we don't accidentally remove important coverage.

## Adding Tests Conveniently

### Approach 1: Test Helper Generator (Recommended)

Create a function that generates distribution tests:

```typescript
/**
 * Generate a distribution test for a random function.
 */
function makeDistributionTest(
  functionName: string,
  importPath: string,
  inputGenerator: (i: number) => string,
  range: [number, number],
  sampleCount: number
): () => Promise<void> {
  return async () => {
    const src = `
      import ${importPath};

      @compute @workgroup_size(1)
      fn main() {
        for (var i = 0u; i < ${sampleCount}u; i++) {
          let sample = ${functionName}(${inputGenerator.toString().replace('i', 'i')});
          test::results[i] = sample;
        }
      }
    `;

    const samples = await testDistribution(src, sampleCount);
    expectDistribution(samples, range);
  };
}

// Usage:
test("random - distribution",
  makeDistributionTest(
    "random",
    "lygia::generative::random::random",
    (i) => `f32(${i})`,
    [0.0, 1.0],
    1024
  )
);
```

### Approach 2: Batch Test Definition

Define all tests in a configuration array:

```typescript
const distributionTests = [
  { name: "random", import: "lygia::generative::random::random",
    input: (i: string) => `f32(${i})`, range: [0, 1], samples: 1024 },
  { name: "random2", import: "lygia::generative::random::random2",
    input: (i: string) => `vec2f(f32(${i} % 32u), f32(${i} / 32u))`,
    range: [0, 1], samples: 512 },
  { name: "srandom", import: "lygia::generative::srandom::srandom",
    input: (i: string) => `f32(${i})`, range: [-1, 1], samples: 1024 },
  // ... more
];

for (const config of distributionTests) {
  test(`${config.name} - distribution`, async () => {
    const src = `
      import ${config.import};
      @compute @workgroup_size(1)
      fn main() {
        for (var i = 0u; i < ${config.samples}u; i++) {
          test::results[i] = ${config.name}(${config.input('i')});
        }
      }
    `;
    const samples = await testDistribution(src, config.samples);
    expectDistribution(samples, config.range);
  });
}
```

### Approach 3: Manual but Consistent (Simplest) ⭐ RECOMMENDED

Just copy-paste the pattern for each function with minor tweaks.

**IMPORTANT:** Use the `constants::` mechanism to pass the sample count to WESL, not string interpolation!

```typescript
// Pattern for scalar functions
test("random - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        test::results[i] = random(f32(i));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount
  });
  expectDistribution(samples, [0.0, 1.0]);
});

// Pattern for vector input functions
test("random2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = random2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount
  });
  expectDistribution(samples, [0.0, 1.0]);
});

// Pattern for signed functions
test("srandom - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        test::results[i] = srandom(f32(i));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount
  });
  expectDistribution(samples, [-1.0, 1.0]);
});
```

**Recommendation:** Start with Approach 3 (manual) for the first 4-5 tests. It's simple, explicit, and follows WESL best practices.

## Implementation Sequence

### Phase 1: Add Infrastructure (Low Risk)

1. **Add helper functions to `testUtil.ts`:**
   - Import `elementStride` from wesl-debug
   - Add `testDistribution()` function
   - Add `expectDistribution()` function
   - Add `expectMean()` function (optional, simpler alternative)

### Phase 2: Add Distribution Tests (Medium Risk)

2. **Add 2-3 initial tests to `generative.test.ts`:**
   - `random - distribution`
   - `random2 - distribution`
   - `srandom - distribution`

3. **Run tests and tune thresholds if needed:**
   - If tests are too flaky, increase tolerances slightly (±0.05 mean, ±0.02 bucket)
   - If tests pass with obviously bad distributions, decrease tolerances
   - Document any threshold adjustments in this file

4. **Add remaining high-priority tests:**
   - `random3 - distribution`
   - `srandom2 - distribution`

5. **Add medium-priority tests (optional):**
   - `random22 - distribution` (test x component)
   - `random33 - distribution` (test x component)
   - `srandom22 - distribution` (test x component)

### Phase 3: Consolidate Existing Tests (Conservative)

6. **Consolidate one function at a time:**
   - Follow the consolidation checklist (see "Test Consolidation Strategy")
   - Start with `random()` as the example
   - Remove redundant range checks
   - Keep determinism + exact value tests
   - Verify tests still pass

7. **Apply to remaining functions:**
   - `random2`, `random3`, `random4`
   - `random21`, `random22`, `random23`
   - `random31`, `random32`, `random33`
   - `random41`, `random42`, `random43`, `random44`
   - `srandom`, `srandom2`, `srandom3`, `srandom4`
   - `srandom22`, `srandom33`

8. **Document learnings:**
   - Update this file with any discoveries
   - Note any functions that needed different thresholds
   - Record any interesting distribution issues found
   - Document which tests were consolidated vs removed

## Future Enhancements

### Chi-Squared Test (More Rigorous)

For more sophisticated statistical testing:

```typescript
function chiSquaredTest(buckets: number[], expectedPerBucket: number): number {
  let chiSq = 0;
  for (const observed of buckets) {
    const diff = observed - expectedPerBucket;
    chiSq += (diff * diff) / expectedPerBucket;
  }
  return chiSq;
  // Compare against chi-squared distribution table
  // df = 9 (10 buckets - 1), p=0.05 → critical value = 16.919
}
```

### Multi-Dimensional Distribution Tests

Test independence of vector components:

```typescript
test("random22 - component independence", async () => {
  // Collect vec2f samples
  // Test correlation between x and y components
  // Should be near zero for independent components
});
```

### Visual Distribution Analysis

Generate histograms for debugging:

```typescript
function generateHistogram(samples: number[], bucketCount = 10): string {
  // ASCII art histogram
  // Useful for debugging distribution issues
}
```

## References

- **wesl-debug PR**: Configurable buffer sizes (commit 742c399f)
- **Test best practices**: notes/test-review.md
- **LYGIA WESL tests**: test/wesl/generative.test.ts
- **Statistical testing**: Chi-squared goodness of fit test
- **WESL best practices**: CLAUDE.md (no string interpolation in WESL code)

## Summary

### Key Decisions

1. ✅ **Use constants:: mechanism** - Pass sample counts via `constants::SAMPLE_COUNT`, not string interpolation
2. ✅ **GPU loop approach** - Collect samples in a single workgroup loop (Option A)
3. ✅ **Manual test pattern** - Use Approach 3 (copy-paste) for initial tests
4. ✅ **Conservative consolidation** - Add distribution tests first, then remove redundant checks

### What Gets Tested

**After implementation, each random function will have:**
- ✅ Determinism test (same input → same output)
- ✅ Exact value test (GLSL↔WESL parity)
- ✅ Distribution test (mean + uniformity across 10 buckets)

**What gets removed:**
- ❌ Range bounds checks (redundant with distribution test)
- ❌ "Different inputs" checks (redundant with 1024 diverse inputs)

### Sample Sizes

| Input Type | Samples | Buffer Size | Rationale |
|-----------|---------|-------------|-----------|
| Scalar (f32) | 1024 | 4096 bytes | Simple sequence needs more samples |
| Vector (vec2f/vec3f) | 512 | 2048 bytes | 2D/3D space provides variation |

### Statistical Thresholds

| Test | Expected | Tolerance | Detects |
|------|----------|-----------|---------|
| Mean | 0.5 for [0,1]<br>0.0 for [-1,1] | ±0.05 | Systematic bias |
| Buckets | 10% per bucket | ±2% (8-12%) | Clustering patterns |

### Implementation Effort

| Phase | Effort | Risk | Task |
|-------|--------|------|------|
| 1. Infrastructure | ~30 min | Low | Add 2 helper functions |
| 2. Initial tests | ~30 min | Medium | Add 3 distribution tests |
| 3. Tune thresholds | ~15 min | Low | Adjust if too flaky |
| 4. Consolidation | ~2 hours | Low | Remove redundant checks from ~20 functions |

**Total: ~3-4 hours** for complete implementation
