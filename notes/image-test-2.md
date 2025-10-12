# Image Testing Extensions for wesl-debug

**Target Audience:** wesl-debug developers
**Purpose:** Add full-image testing and visual regression capabilities
**Status:** Design document for implementation

---

## Executive Summary

This document extends the uniform/texture support outlined in `image-test.md` with full-image testing capabilities. These additions enable visual regression testing and fast iteration for shader development.

**Key Features:**
1. **Standard test image library** - Pre-generated patterns for common shader operations
2. **Full image retrieval** - Get complete rendered images, not just pixel (0,0)
3. **PNG generation** - Save GPU renders to files for visual inspection
4. **Image comparison** - Automated visual regression testing adapted from Vitest browser mode
5. **HTML diff report** - Auto-generated visual comparison table for all failures
6. **Fast dev cycle** - Quick visual tuning with auto-save and snapshot management

**Implementation Location:** `wesl-debug` package (not LYGIA-specific)

**Key Principle:** Keep implementation generic - works with any WGSL/WESL shaders.

---

## 1. Standard Test Image Library

Beyond the basic patterns (solid, gradient, checkerboard) already in wesl-debug, add 5 standard test images commonly used in image processing.

### Proposed Test Images

1. **Radial Gradient** (256×256)
   - Circular gradient from white center to black edge
   - Use case: Blur filters, radial distortion, vignette effects

2. **Edge Pattern** (256×256)
   - Sharp vertical, horizontal, and diagonal lines
   - Use case: Edge detection filters (Sobel, Prewitt, etc.)

3. **Color Bars** (256×256)
   - Vertical bars of primary/secondary colors (red, yellow, green, cyan, blue, magenta)
   - Use case: Color space conversions, color blending operations

4. **Noise Pattern** (256×256)
   - Seeded random noise (deterministic)
   - Use case: Denoise filters, noise reduction testing

5. **Photo Sample** (512×512)
   - Real photo image (provided)
   - Use case: Realistic filter testing (sharpen, blur, tone mapping)

### API Design

```typescript
// Add to wesl-debug/src/TextureHelpers.ts

/**
 * Create a radial gradient texture (white center → black edge)
 */
export function createRadialGradientTexture(
  device: GPUDevice,
  size: number
): GPUTexture;

/**
 * Create an edge test pattern with sharp lines
 */
export function createEdgePatternTexture(
  device: GPUDevice,
  size: number
): GPUTexture;

/**
 * Create color bars (RGB primaries and secondaries)
 */
export function createColorBarsTexture(
  device: GPUDevice,
  size: number
): GPUTexture;

/**
 * Create seeded noise pattern (deterministic)
 */
export function createNoiseTexture(
  device: GPUDevice,
  size: number,
  seed?: number
): GPUTexture;

/**
 * Load photo sample from provided image file
 */
export function createPhotoSampleTexture(
  device: GPUDevice,
  imagePath: string
): GPUTexture;
```

### Implementation Notes

- All textures use `rgba8unorm` format (filterable, efficient)
- Deterministic generation (same inputs → same outputs)
- Data generated on CPU, uploaded to GPU via `writeTexture`
- Texture usage: `TEXTURE_BINDING | COPY_DST`
- Photo sample loaded from provided image file (using pngjs or similar)

---

## 2. Full Image Retrieval

Extend `testFragmentShader()` to return complete rendered images, not just pixel (0,0).

### API Design

```typescript
// Add to wesl-debug/src/TestFragmentShader.ts

export interface ImageData {
  /** Raw pixel data (width × height × 4 bytes) */
  data: Uint8Array;

  /** Image width in pixels */
  width: number;

  /** Image height in pixels */
  height: number;

  /** GPU texture format */
  format: GPUTextureFormat;
}

/**
 * Test a fragment shader and return the complete rendered image
 * @returns Full image data for visual inspection or comparison
 */
export async function testFragmentShaderImage(
  params: FragmentTestParams
): Promise<ImageData>;
```

### Implementation Strategy

**Phase 1: Basic Implementation**

```typescript
export async function testFragmentShaderImage(
  params: FragmentTestParams
): Promise<ImageData> {
  const { device, size = [1, 1], outputFormat = "rgba32float" } = params;

  // Render shader (reuse existing pipeline setup)
  const texture = await renderFragmentShader(params);

  // Read back ENTIRE texture using thimbleberry utility (like testFragmentShader does)
  const imageData = await withTextureCopy(
    device,
    texture,
    outputFormat,
    size,
    (data) => convertToUint8(data, outputFormat, size[0], size[1])
  );

  texture.destroy();

  return {
    data: imageData,
    width: size[0],
    height: size[1],
    format: outputFormat,
  };
}
```

**Phase 2: Format Conversion**

```typescript
function convertToUint8(
  data: Uint8Array,
  format: GPUTextureFormat,
  width: number,
  height: number
): Uint8Array {
  if (format === "rgba8unorm") {
    return data; // Already uint8
  }

  if (format === "rgba32float") {
    // Convert float32 → uint8
    const floatView = new Float32Array(data.buffer);
    const uint8Data = new Uint8Array(width * height * 4);

    for (let i = 0; i < floatView.length; i++) {
      uint8Data[i] = Math.round(Math.max(0, Math.min(1, floatView[i])) * 255);
    }

    return uint8Data;
  }

  throw new Error(`Unsupported format: ${format}`);
}
```

---

## 3. PNG Generation

Add utilities to save GPU renders as PNG files for visual inspection.

### Dependencies

```json
{
  "dependencies": {
    "pngjs": "^7.0.0"
  },
  "devDependencies": {
    "@types/pngjs": "^6.0.0"
  }
}
```

### API Design

```typescript
// Add to wesl-debug/src/ImageHelpers.ts

import { PNG } from "pngjs";
import * as fs from "fs";
import * as path from "path";

/**
 * Save ImageData to a PNG file
 * @param imageData - Image data from testFragmentShaderImage
 * @param filepath - Output file path (e.g., "test/output/blur.png")
 */
export async function saveImageDataToPNG(
  imageData: ImageData,
  filepath: string
): Promise<void> {
  // Ensure directory exists
  const dir = path.dirname(filepath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // Create PNG
  const png = new PNG({
    width: imageData.width,
    height: imageData.height,
    colorType: 6, // RGBA
  });

  png.data = Buffer.from(imageData.data);

  // Write to file
  return new Promise((resolve, reject) => {
    png.pack()
      .pipe(fs.createWriteStream(filepath))
      .on("finish", resolve)
      .on("error", reject);
  });
}
```

### Usage Example

```typescript
// In a test file
import { testFragmentShaderImage, saveImageDataToPNG } from "wesl-debug";

test("simple blur filter", async () => {
  const result = await testFragmentShaderImage({
    device,
    projectDir: import.meta.url,
    src: blurShaderSource,
    size: [256, 256],
  });

  // Save for visual inspection
  await saveImageDataToPNG(result, "test/output/blur-result.png");
});
```

---

## 4. Image Comparison (Adapted from Vitest Browser Mode)

Extract and adapt Vitest's screenshot comparison infrastructure for Node.js-based GPU testing.

### Architecture Overview

Vitest browser mode's `toMatchScreenshot` uses a clean architecture:

1. **Comparator** - Performs pixel-by-pixel comparison (uses `pixelmatch`)
2. **Snapshot Manager** - Handles reference/actual/diff file management
3. **Custom Matcher** - Integrates with Vitest's `expect` API

We'll adapt this for Node.js WebGPU testing by:
- Extracting the comparator and snapshot manager logic
- Replacing browser screenshot capture with GPU texture readback
- Keeping the file management and comparison logic

### Dependencies

```json
{
  "dependencies": {
    "pixelmatch": "^6.0.0"
  }
}
```

### Phase 1: Comparator Interface

```typescript
// Add to wesl-debug/src/ImageComparison.ts

import pixelmatch from "pixelmatch";
import { PNG } from "pngjs";

export interface ComparisonOptions {
  /** Color difference threshold (0-1). Default: 0.1 */
  threshold?: number;

  /** Max ratio of pixels allowed to differ (0-1). Default: 0 */
  allowedMismatchedPixelRatio?: number;

  /** Max absolute number of pixels allowed to differ. Default: 0 */
  allowedMismatchedPixels?: number;
}

export interface ComparisonResult {
  /** Whether the comparison passed */
  pass: boolean;

  /** PNG buffer of the diff image (if generated) */
  diffBuffer?: Buffer;

  /** Human-readable message */
  message: string;

  /** Number of pixels that differed */
  mismatchedPixels: number;

  /** Ratio of pixels that differed (0-1) */
  mismatchedPixelRatio: number;
}

/**
 * Compare two images using pixelmatch
 * Adapted from Vitest browser mode's comparator
 */
export async function compareImages(
  reference: Buffer,
  actual: Buffer,
  options: ComparisonOptions = {}
): Promise<ComparisonResult> {
  const {
    threshold = 0.1,
    allowedMismatchedPixelRatio = 0,
    allowedMismatchedPixels = 0,
  } = options;

  // Parse PNG buffers
  const refPng = PNG.sync.read(reference);
  const actualPng = PNG.sync.read(actual);

  // Validate dimensions
  if (refPng.width !== actualPng.width || refPng.height !== actualPng.height) {
    return {
      pass: false,
      message: `Image dimensions don't match: ${refPng.width}×${refPng.height} vs ${actualPng.width}×${actualPng.height}`,
      mismatchedPixels: refPng.width * refPng.height,
      mismatchedPixelRatio: 1.0,
    };
  }

  // Create diff buffer
  const { width, height } = refPng;
  const diffPng = new PNG({ width, height });

  // Compare using pixelmatch
  const mismatchedPixels = pixelmatch(
    refPng.data,
    actualPng.data,
    diffPng.data,
    width,
    height,
    { threshold }
  );

  const totalPixels = width * height;
  const mismatchedPixelRatio = mismatchedPixels / totalPixels;

  // Determine pass/fail
  const pass =
    mismatchedPixels <= allowedMismatchedPixels ||
    mismatchedPixelRatio <= allowedMismatchedPixelRatio;

  // Generate diff buffer if failed
  const diffBuffer = pass ? undefined : PNG.sync.write(diffPng);

  // Create message
  const message = pass
    ? `Images match (${mismatchedPixels} pixels differ, ${(mismatchedPixelRatio * 100).toFixed(2)}%)`
    : `Images don't match: ${mismatchedPixels} pixels differ (${(mismatchedPixelRatio * 100).toFixed(2)}%), threshold: ${allowedMismatchedPixelRatio * 100}%`;

  return {
    pass,
    diffBuffer,
    message,
    mismatchedPixels,
    mismatchedPixelRatio,
  };
}
```

### Phase 2: Snapshot Manager

```typescript
// Add to wesl-debug/src/SnapshotManager.ts

import * as fs from "fs";
import * as path from "path";

export interface SnapshotConfig {
  /** Directory for reference snapshots. Default: "__image_snapshots__" */
  snapshotDir?: string;

  /** Directory for diff images. Default: "__image_diffs__" */
  diffDir?: string;

  /** Directory for actual images. Default: "__image_actual__" */
  actualDir?: string;
}

/**
 * Manages reference/actual/diff snapshot files
 * Adapted from Vitest browser mode's snapshot manager
 */
export class ImageSnapshotManager {
  private config: Required<SnapshotConfig>;
  private testDir: string;

  constructor(testFilePath: string, config: SnapshotConfig = {}) {
    this.testDir = path.dirname(testFilePath);
    this.config = {
      snapshotDir: config.snapshotDir ?? "__image_snapshots__",
      diffDir: config.diffDir ?? "__image_diffs__",
      actualDir: config.actualDir ?? "__image_actual__",
    };
  }

  /**
   * Get path to reference snapshot
   */
  getReferencePath(snapshotName: string): string {
    return path.join(this.testDir, this.config.snapshotDir, `${snapshotName}.png`);
  }

  /**
   * Get path to actual snapshot
   */
  getActualPath(snapshotName: string): string {
    return path.join(this.testDir, this.config.actualDir, `${snapshotName}.png`);
  }

  /**
   * Get path to diff snapshot
   */
  getDiffPath(snapshotName: string): string {
    return path.join(this.testDir, this.config.diffDir, `${snapshotName}.png`);
  }

  /**
   * Check if snapshots should be updated
   */
  shouldUpdate(): boolean {
    return process.env.VITEST_UPDATE_SNAPSHOTS === "true" || process.argv.includes("-u");
  }

  /**
   * Load reference snapshot (returns null if doesn't exist)
   */
  async loadReference(snapshotName: string): Promise<Buffer | null> {
    const refPath = this.getReferencePath(snapshotName);
    if (!fs.existsSync(refPath)) {
      return null;
    }
    return fs.promises.readFile(refPath);
  }

  /**
   * Save reference snapshot
   */
  async saveReference(buffer: Buffer, snapshotName: string): Promise<void> {
    const refPath = this.getReferencePath(snapshotName);
    const dir = path.dirname(refPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    await fs.promises.writeFile(refPath, buffer);
  }

  /**
   * Save actual snapshot
   */
  async saveActual(buffer: Buffer, snapshotName: string): Promise<void> {
    const actualPath = this.getActualPath(snapshotName);
    const dir = path.dirname(actualPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    await fs.promises.writeFile(actualPath, buffer);
  }

  /**
   * Save diff snapshot
   */
  async saveDiff(buffer: Buffer, snapshotName: string): Promise<void> {
    const diffPath = this.getDiffPath(snapshotName);
    const dir = path.dirname(diffPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    await fs.promises.writeFile(diffPath, buffer);
  }
}
```

### Phase 3: Custom Vitest Matcher

```typescript
// Add to wesl-debug/src/ImageSnapshotMatcher.ts

import { expect } from "vitest";
import type { MatcherResult } from "@vitest/expect";
import { compareImages, type ComparisonOptions } from "./ImageComparison";
import { ImageSnapshotManager } from "./SnapshotManager";
import { saveImageDataToPNG } from "./ImageHelpers";
import type { ImageData } from "./TestFragmentShader";
import { PNG } from "pngjs";

interface ToMatchImageSnapshotOptions extends ComparisonOptions {
  /** Custom snapshot name (defaults to auto-generated) */
  name?: string;
}

/**
 * Custom Vitest matcher for image snapshots
 * Usage: await expect(imageData).toMatchImageSnapshot('snapshot-name');
 */
async function toMatchImageSnapshot(
  this: any,
  received: ImageData | Buffer,
  nameOrOptions?: string | ToMatchImageSnapshotOptions
): Promise<MatcherResult> {
  // Parse arguments
  let snapshotName: string;
  let options: ComparisonOptions;

  if (typeof nameOrOptions === "string") {
    snapshotName = nameOrOptions;
    options = {};
  } else {
    snapshotName = nameOrOptions?.name ?? this.currentTestName ?? "snapshot";
    options = nameOrOptions ?? {};
  }

  // Convert ImageData to PNG Buffer
  const actualBuffer = Buffer.isBuffer(received)
    ? received
    : await imageDataToPNGBuffer(received);

  // Setup snapshot manager
  const testPath = this.testPath ?? process.cwd();
  const manager = new ImageSnapshotManager(testPath);

  // Save actual snapshot
  await manager.saveActual(actualBuffer, snapshotName);

  // Load reference snapshot
  const referenceBuffer = await manager.loadReference(snapshotName);

  // If no reference exists, create it
  if (!referenceBuffer) {
    if (manager.shouldUpdate() || process.env.CI !== "true") {
      await manager.saveReference(actualBuffer, snapshotName);
      return {
        pass: true,
        message: () => `Created new snapshot: ${snapshotName}`,
      };
    } else {
      return {
        pass: false,
        message: () => `No reference snapshot found: ${snapshotName}. Run with --update-snapshots to create.`,
      };
    }
  }

  // Compare images
  const comparison = await compareImages(referenceBuffer, actualBuffer, options);

  // Save diff if failed
  if (!comparison.pass && comparison.diffBuffer) {
    await manager.saveDiff(comparison.diffBuffer, snapshotName);
  }

  // Update snapshot if requested
  if (!comparison.pass && manager.shouldUpdate()) {
    await manager.saveReference(actualBuffer, snapshotName);
    return {
      pass: true,
      message: () => `Updated snapshot: ${snapshotName}`,
    };
  }

  return {
    pass: comparison.pass,
    message: () => comparison.message,
    actual: `${manager.getActualPath(snapshotName)}`,
    expected: `${manager.getReferencePath(snapshotName)}`,
  };
}

async function imageDataToPNGBuffer(imageData: ImageData): Promise<Buffer> {
  const png = new PNG({
    width: imageData.width,
    height: imageData.height,
    colorType: 6, // RGBA
  });
  png.data = Buffer.from(imageData.data);
  return PNG.sync.write(png);
}

// Export matcher setup function
export function setupImageSnapshotMatcher() {
  expect.extend({ toMatchImageSnapshot });
}
```

---

## 5. Fast Development Cycle

Enable quick visual tuning by auto-saving test outputs and providing easy snapshot updates.

### Directory Structure

```
test/
  __image_snapshots__/     # Reference images (git committed)
    blur-filter.png
    edge-detection.png
  __image_diffs__/         # Diff images (gitignored)
    blur-filter.png
  __image_actual__/        # Actual outputs (gitignored)
    blur-filter.png
  output/                  # Dev mode outputs (gitignored)
    experiment-1.png
```

### .gitignore Additions

```
# Image test outputs (don't commit)
test/__image_diffs__/
test/__image_actual__/
test/output/
```

### Configuration

```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    // Image snapshot configuration
    setupFiles: ["./test/setup.ts"],
  },
});
```

```typescript
// test/setup.ts
import { setupImageSnapshotMatcher } from "wesl-debug";

// Setup custom matcher
setupImageSnapshotMatcher();
```

### Workflow

**Initial Test Creation:**
```bash
# Run tests - creates reference snapshots
pnpm test

# Review generated snapshots in __image_snapshots__/
# Commit if they look correct
git add test/__image_snapshots__/
git commit -m "Add visual regression tests"
```

**Making Changes:**
```bash
# Run tests - fails if output changed
pnpm test

# Review diffs in __image_diffs__/
# If changes are intentional, update snapshots
pnpm test -- -u

# Commit updated references
git add test/__image_snapshots__/
git commit -m "Update snapshots after filter improvement"
```

**Quick Experimentation:**
```typescript
// Save to output/ for quick visual inspection
test.only("experiment with blur radius", async () => {
  for (const radius of [1, 2, 4, 8]) {
    const result = await testFragmentShaderImage({ /* params with radius */ });
    await saveImageDataToPNG(result, `test/output/blur-radius-${radius}.png`);
  }
});
```

### HTML Diff Report

When image snapshot tests fail, automatically generate an HTML report showing all failures in a single page for easy review.

#### Report Structure

**Location:** `test/__image_diffs__/report.html` (automatically generated on test failure)

**Layout:**
- Table format with one row per failed test
- Three columns: Expected | Actual | Diff
- Metadata: test name, mismatch statistics, file paths
- Generated timestamp

#### API Design

```typescript
// Add to wesl-debug/src/DiffReport.ts

export interface ImageSnapshotFailure {
  /** Test name */
  testName: string;

  /** Snapshot name */
  snapshotName: string;

  /** Comparison result */
  comparison: ComparisonResult;

  /** File paths */
  paths: {
    reference: string;
    actual: string;
    diff: string;
  };
}

export interface DiffReportConfig {
  /** Auto-open report in browser. Default: false */
  autoOpen?: boolean;

  /** Report output path. Default: "test/__image_diffs__/report.html" */
  outputPath?: string;
}

/**
 * Generate HTML diff report for all failed image snapshots
 * Called automatically at end of test run if failures exist
 */
export async function generateDiffReport(
  failures: ImageSnapshotFailure[],
  config?: DiffReportConfig
): Promise<void>;
```

#### Implementation

```typescript
// wesl-debug/src/DiffReport.ts
import * as fs from "fs";
import * as path from "path";
import { exec } from "child_process";

export async function generateDiffReport(
  failures: ImageSnapshotFailure[],
  config: DiffReportConfig = {}
): Promise<void> {
  const { autoOpen = false, outputPath = "test/__image_diffs__/report.html" } = config;

  if (failures.length === 0) {
    return; // No failures, no report needed
  }

  // Generate HTML content
  const html = createReportHTML(failures);

  // Ensure directory exists
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // Write report
  await fs.promises.writeFile(outputPath, html, "utf-8");

  console.log(`\n📊 Image diff report generated: ${outputPath}`);

  // Auto-open in browser
  if (autoOpen) {
    const openCommand = process.platform === "darwin" ? "open" :
                       process.platform === "win32" ? "start" : "xdg-open";
    exec(`${openCommand} "${outputPath}"`);
  }
}

function createReportHTML(failures: ImageSnapshotFailure[]): string {
  const timestamp = new Date().toLocaleString();
  const totalFailures = failures.length;

  const rows = failures.map(failure => {
    const { testName, snapshotName, comparison, paths } = failure;
    const { mismatchedPixels, mismatchedPixelRatio } = comparison;

    // Convert absolute paths to relative for HTML links
    const relRef = path.relative(path.dirname(paths.diff), paths.reference);
    const relActual = path.relative(path.dirname(paths.diff), paths.actual);
    const relDiff = path.basename(paths.diff);

    return `
      <tr>
        <td class="test-name">
          <strong>${escapeHtml(testName)}</strong><br>
          <code>${escapeHtml(snapshotName)}</code>
        </td>
        <td class="image-cell">
          <a href="${relRef}" target="_blank">
            <img src="${relRef}" alt="Expected" />
          </a>
          <div class="label">Expected</div>
        </td>
        <td class="image-cell">
          <a href="${relActual}" target="_blank">
            <img src="${relActual}" alt="Actual" />
          </a>
          <div class="label">Actual</div>
        </td>
        <td class="image-cell">
          <a href="${relDiff}" target="_blank">
            <img src="${relDiff}" alt="Diff" />
          </a>
          <div class="label">Diff</div>
        </td>
        <td class="stats">
          <div><strong>${mismatchedPixels}</strong> pixels</div>
          <div><strong>${(mismatchedPixelRatio * 100).toFixed(2)}%</strong></div>
        </td>
      </tr>
    `;
  }).join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Snapshot Failures - ${totalFailures} failed</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 20px;
      background: #f5f5f5;
    }
    header {
      background: white;
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    h1 {
      color: #d73027;
      margin-bottom: 10px;
    }
    .meta {
      color: #666;
      font-size: 14px;
    }
    .update-hint {
      background: #fff3cd;
      border: 1px solid #ffc107;
      border-radius: 4px;
      padding: 12px;
      margin: 15px 0;
    }
    .update-hint code {
      background: #f8f9fa;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: 'Monaco', 'Courier New', monospace;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    th, td {
      padding: 15px;
      text-align: left;
      border-bottom: 1px solid #e0e0e0;
    }
    th {
      background: #f8f9fa;
      font-weight: 600;
      color: #333;
      text-align: center;
    }
    tr:last-child td {
      border-bottom: none;
    }
    tr:hover {
      background: #fafafa;
    }
    .test-name {
      min-width: 200px;
      vertical-align: top;
    }
    .test-name strong {
      color: #333;
    }
    .test-name code {
      color: #666;
      font-size: 12px;
      background: #f5f5f5;
      padding: 2px 6px;
      border-radius: 3px;
    }
    .image-cell {
      text-align: center;
      vertical-align: top;
      width: 25%;
    }
    .image-cell a {
      display: block;
      text-decoration: none;
    }
    .image-cell img {
      max-width: 100%;
      height: auto;
      max-height: 300px;
      border: 1px solid #ddd;
      border-radius: 4px;
      cursor: pointer;
      transition: transform 0.2s;
    }
    .image-cell img:hover {
      transform: scale(1.02);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .label {
      margin-top: 8px;
      font-size: 12px;
      color: #666;
      font-weight: 500;
    }
    .stats {
      text-align: center;
      color: #d73027;
      font-size: 14px;
      vertical-align: top;
    }
    .stats div {
      margin: 5px 0;
    }
    footer {
      margin-top: 20px;
      text-align: center;
      color: #999;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <header>
    <h1>🔴 Image Snapshot Failures</h1>
    <div class="meta">
      <strong>${totalFailures}</strong> test${totalFailures === 1 ? '' : 's'} failed •
      Generated: ${escapeHtml(timestamp)}
    </div>
    <div class="update-hint">
      💡 <strong>To update snapshots:</strong> Run <code>pnpm test -- -u</code> or <code>vitest -u</code>
    </div>
  </header>

  <table>
    <thead>
      <tr>
        <th>Test</th>
        <th>Expected</th>
        <th>Actual</th>
        <th>Diff</th>
        <th>Mismatch</th>
      </tr>
    </thead>
    <tbody>
      ${rows}
    </tbody>
  </table>

  <footer>
    Click images to view full size •
    Diff images highlight mismatched pixels in yellow/red
  </footer>
</body>
</html>`;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
```

#### Integration with Vitest Reporter

```typescript
// wesl-debug/src/ImageSnapshotReporter.ts

import type { Reporter, Task } from "vitest";
import { generateDiffReport, type ImageSnapshotFailure } from "./DiffReport";

/**
 * Custom Vitest reporter that collects image snapshot failures
 * and generates HTML report at end of test run
 */
export class ImageSnapshotReporter implements Reporter {
  private failures: ImageSnapshotFailure[] = [];

  onTaskUpdate(tasks: Task[]) {
    // Collect image snapshot failures from test context
    for (const task of tasks) {
      if (task.result?.state === "fail" && task.meta?.imageSnapshotFailure) {
        this.failures.push(task.meta.imageSnapshotFailure);
      }
    }
  }

  async onFinished() {
    if (this.failures.length > 0) {
      await generateDiffReport(this.failures, {
        autoOpen: process.env.IMAGE_DIFF_AUTO_OPEN === "true",
      });
    }
  }
}
```

#### Configuration

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config";
import { ImageSnapshotReporter } from "wesl-debug";

export default defineConfig({
  test: {
    setupFiles: ["./test/setup.ts"],
    reporters: [
      "default",
      new ImageSnapshotReporter(), // Add custom reporter
    ],
  },
});
```

#### Updated Workflow

```bash
# Run tests
pnpm test

# If image tests fail:
# ✅ Terminal shows failure summary
# ✅ Diff images saved to __image_diffs__/
# ✅ HTML report generated: test/__image_diffs__/report.html

# Open report (if not auto-opened)
open test/__image_diffs__/report.html

# Review all failures in table:
# - See expected/actual/diff side by side
# - Click images to view full size
# - Check mismatch statistics

# Decision:
# 1. Fix bugs → re-run tests
# 2. Update snapshots → pnpm test -- -u
```

#### Example Report Screenshot

When tests fail, the generated HTML shows:

```
+------------------------------------------------------------------+
| 🔴 Image Snapshot Failures                                      |
| 3 tests failed • Generated: 2025-01-14 10:30:45                 |
| 💡 To update snapshots: Run pnpm test -- -u                     |
+------------------------------------------------------------------+
| Test         | Expected    | Actual      | Diff        | Stats   |
|--------------|-------------|-------------|-------------|---------|
| box-blur     | [ref img]   | [act img]   | [diff img]  | 237 px  |
| blur-filter  |             |             |             | 1.45%   |
|--------------|-------------|-------------|-------------|---------|
| edge-sobel   | [ref img]   | [act img]   | [diff img]  | 89 px   |
| edge-det     |             |             |             | 0.54%   |
|--------------|-------------|-------------|-------------|---------|
| grayscale    | [ref img]   | [act img]   | [diff img]  | 12 px   |
|              |             |             |             | 0.07%   |
+------------------------------------------------------------------+
```

Each image is clickable to view full size.

---

## 6. Implementation Tests (wesl-debug)

Test the image comparison infrastructure using plain WGSL shaders (no LYGIA).

### Test 1: Simple Blur Filter

```typescript
// wesl-debug/src/test/ImageSnapshot.test.ts
import { expect, test } from "vitest";
import { getGPUDevice } from "../GetGPUDevice";
import { testFragmentShaderImage } from "../TestFragmentShader";
import { createCheckerboardTexture, createSampler } from "../TextureHelpers";
import { setupImageSnapshotMatcher } from "../ImageSnapshotMatcher";

setupImageSnapshotMatcher();

test("simple box blur", async () => {
  const device = await getGPUDevice();
  const inputTex = createCheckerboardTexture(device, 128, 128, 16);
  const sampler = createSampler(device);

  const src = `
    @group(0) @binding(0) var input_tex: texture_2d<f32>;
    @group(0) @binding(1) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 128.0;
      let pixel_size = 1.0 / 128.0;

      // 3x3 box blur
      var color = vec4f(0.0);
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let offset = vec2f(f32(x), f32(y)) * pixel_size;
          color += textureSample(input_tex, input_samp, uv + offset);
        }
      }

      return color / 9.0;
    }
  `;

  const result = await testFragmentShaderImage({
    projectDir: import.meta.url,
    device,
    src,
    size: [128, 128],
    inputTextures: [{ texture: inputTex, sampler }],
  });

  await expect(result).toMatchImageSnapshot("box-blur");
});
```

### Test 2: Edge Detection

```typescript
test("simple edge detection", async () => {
  const device = await getGPUDevice();
  const inputTex = createEdgePatternTexture(device, 128);
  const sampler = createSampler(device);

  const src = `
    @group(0) @binding(0) var input_tex: texture_2d<f32>;
    @group(0) @binding(1) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 128.0;
      let pixel_size = 1.0 / 128.0;

      // Simple Sobel operator
      let tl = textureSample(input_tex, input_samp, uv + vec2f(-pixel_size, -pixel_size)).rgb;
      let tr = textureSample(input_tex, input_samp, uv + vec2f( pixel_size, -pixel_size)).rgb;
      let bl = textureSample(input_tex, input_samp, uv + vec2f(-pixel_size,  pixel_size)).rgb;
      let br = textureSample(input_tex, input_samp, uv + vec2f( pixel_size,  pixel_size)).rgb;

      let gx = -tl + tr - bl + br;
      let gy = -tl - tr + bl + br;
      let mag = length(vec2f(dot(gx, vec3f(0.299, 0.587, 0.114)),
                             dot(gy, vec3f(0.299, 0.587, 0.114))));

      return vec4f(vec3f(mag), 1.0);
    }
  `;

  const result = await testFragmentShaderImage({
    projectDir: import.meta.url,
    device,
    src,
    size: [128, 128],
    inputTextures: [{ texture: inputTex, sampler }],
  });

  await expect(result).toMatchImageSnapshot("edge-detection", {
    allowedMismatchedPixelRatio: 0.01, // Allow 1% difference
  });
});
```

### Test 3: Color Transform

```typescript
test("grayscale conversion", async () => {
  const device = await getGPUDevice();
  const inputTex = createColorBarsTexture(device, 128);
  const sampler = createSampler(device);

  const src = `
    @group(0) @binding(0) var input_tex: texture_2d<f32>;
    @group(0) @binding(1) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / 128.0;
      let color = textureSample(input_tex, input_samp, uv);

      // Perceptual grayscale
      let gray = dot(color.rgb, vec3f(0.299, 0.587, 0.114));

      return vec4f(vec3f(gray), 1.0);
    }
  `;

  const result = await testFragmentShaderImage({
    projectDir: import.meta.url,
    device,
    src,
    size: [128, 128],
    inputTextures: [{ texture: inputTex, sampler }],
  });

  await expect(result).toMatchImageSnapshot("grayscale");
});
```

---

## 7. LYGIA Usage Example

Once implemented in wesl-debug, LYGIA can use these features for comprehensive filter testing.

```typescript
// lygia/test/wesl/filter-blur.test.ts
import { expect, test } from "vitest";
import { getGPUDevice, testFragmentShaderImage, createRadialGradientTexture, createSampler } from "wesl-debug";
import { setupImageSnapshotMatcher } from "wesl-debug";

setupImageSnapshotMatcher();

test("gaussianBlur produces expected visual result", async () => {
  const device = await getGPUDevice();
  const inputTex = createRadialGradientTexture(device, 256);
  const sampler = createSampler(device);

  const src = `
    import lygia::filter::gaussianBlur;

    @group(0) @binding(0) var<uniform> u: test::Uniforms;
    @group(0) @binding(1) var input_tex: texture_2d<f32>;
    @group(0) @binding(2) var input_samp: sampler;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let uv = pos.xy / u.resolution;
      let pixel_size = 1.0 / u.resolution;

      return gaussianBlur(input_tex, input_samp, uv, pixel_size, 13);
    }
  `;

  const result = await testFragmentShaderImage({
    projectDir: import.meta.url,
    device,
    src,
    size: [256, 256],
    uniforms: { time: 0.0, mouse: [0.0, 0.0] },
    inputTextures: [{ texture: inputTex, sampler }],
  });

  // Visual regression test - ensures filter behavior is stable
  await expect(result).toMatchImageSnapshot("gaussian-blur-radial", {
    threshold: 0.1,
    allowedMismatchedPixelRatio: 0.01,
  });
});
```

---

## 8. Dependencies

Add to `wesl-debug/package.json`:

```json
{
  "dependencies": {
    "pngjs": "^7.0.0",
    "pixelmatch": "^6.0.0"
  },
  "devDependencies": {
    "@types/pngjs": "^6.0.0"
  }
}
```

---

## 9. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Add standard test image generators (radial, edges, color bars, noise)
- [ ] Add photo sample loader (uses provided image file)
- [ ] Implement `testFragmentShaderImage()` using thimbleberry's `withTextureCopy`
- [ ] Add format conversion utilities (rgba32float → rgba8unorm)
- [ ] Implement PNG saving with `pngjs` (`saveImageDataToPNG`)

### Phase 2: Image Comparison
- [ ] Extract/adapt comparator from Vitest (`compareImages()`)
- [ ] Implement `ImageSnapshotManager`
- [ ] Create custom Vitest matcher `toMatchImageSnapshot`
- [ ] Add configuration support
- [ ] Implement HTML diff report generator (`generateDiffReport()`)
- [ ] Create custom Vitest reporter (`ImageSnapshotReporter`)
- [ ] Add auto-open configuration option

### Phase 3: Tests & Documentation
- [ ] Write wesl-debug tests using plain WGSL
- [ ] Document API in README
- [ ] Add usage examples
- [ ] Update TypeScript exports

---

## 10. License Considerations

**Vitest License:** MIT License
**Action Required:** Check Vitest's MIT license terms and add appropriate attribution when extracting code.

**Files to attribute:**
- `packages/browser/src/node/commands/screenshotMatcher/comparators/pixelmatch.ts`
- `packages/browser/src/node/commands/screenshotMatcher/index.ts`

**Suggested attribution in source files:**
```typescript
// Adapted from Vitest (MIT License)
// Original: https://github.com/vitest-dev/vitest/blob/main/packages/browser/...
// Copyright (c) 2021-Present, Anthony Fu and Vitest contributors
```

---

## Summary

This document outlines extensions to wesl-debug that enable:

1. **Full-image testing** - Get complete rendered results, not just single pixels
2. **Visual regression testing** - Automated snapshot comparison using battle-tested Vitest infrastructure
3. **Fast iteration** - Auto-save outputs, easy snapshot updates, visual inspection workflow
4. **Generic implementation** - Works with any WGSL/WESL shaders, not LYGIA-specific

**Result:** Developers can quickly iterate on shader visual effects with confidence that changes don't break existing behavior.

**Next Steps:** Implement Phase 1 (core infrastructure) in wesl-debug, validate with plain WGSL tests.
