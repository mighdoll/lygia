# LYGIA Development Scripts

This directory contains utility scripts for LYGIA development and WESL conversion.

## Scripts Overview

- **split-test-file.py** - Split large test files into organized categories
- **check-unconverted.sh** - Find GLSL files without WESL conversions
- **find-untested-wesl.sh** - Find WESL files without test coverage
- **find-untested-functions.py** - Find WESL functions without test coverage (function-level granularity)
- **extract-tested-functions.sh** - List all tested WESL functions

---

# split-test-file.py

## Overview

Python script for splitting large test files into smaller, well-organized category-based files.

## Usage

```bash
python3 scripts/split-test-file.py <input-file> <config-json>
```

### Example

```bash
# Split color.test.ts using the provided config
python3 scripts/split-test-file.py \
  test/wesl/color.test.ts \
  scripts/examples/split-color-config.json

# Split Functions.test.ts
python3 scripts/split-test-file.py \
  test/wesl/Functions.test.ts \
  scripts/examples/split-functions-config.json
```

## Configuration Format

The config file is JSON with the following structure:

```json
{
  "header": "import { expect, test } from \"vitest\";\nimport { expectCloseTo, testCompute } from \"./testUtil.ts\";\n\n",
  "categories": {
    "output-filename.test.ts": {
      "tests": ["exactTestName1", "exactTestName2"],
      "patterns": ["substring1", "substring2"]
    }
  }
}
```

### Config Fields

- **`header`** (optional): Text to prepend to each output file (typically imports)
- **`categories`**: Object mapping output filenames to categorization rules
  - **`tests`**: Array of exact test names to include
  - **`patterns`**: Array of substrings - any test containing these will be included

Tests are matched in order:
1. Exact match in `tests` array
2. Pattern match in `patterns` array
3. If no match, test is reported as uncategorized

## Example Configs

### Color Tests Split

See [`examples/split-color-config.json`](examples/split-color-config.json)

Splits color.test.ts into:
- `color-space.test.ts` - Color space conversions (patterns: "2rgb", "rgb2", "2xyz", etc.)
- `color-blend.test.ts` - Blend modes (pattern: "blend")
- `color-composite.test.ts` - Composite operations (patterns: "composite", "layer")
- `color-adjust.test.ts` - Adjustments (explicit list: desaturate, contrast, levels, etc.)
- `color-util.test.ts` - Utilities (explicit list: luminance, luma, mixOklab, etc.)

### Functions Tests Split

See [`examples/split-functions-config.json`](examples/split-functions-config.json)

Splits Functions.test.ts into:
- `math.test.ts` - Math functions (explicit list: fmod, cubicMix, smootherstep, etc.)
- `noise.test.ts` - Noise functions (patterns: "noise", "wavelet", "random", "worley")
- `lighting.test.ts` - Lighting (patterns: "fresnel", "specular", "raymarch")
- `space.test.ts` - Spatial transforms (pattern: "fisheye")

## How It Works

1. Reads the input test file
2. Splits on `test(` and `test.skip(` declarations
3. Extracts test names from each block
4. Categorizes tests based on config rules
5. Writes separate files for each category
6. Reports uncategorized tests (if any)

## Benefits

- **Consistency**: Ensures all split files have the same header/imports
- **Flexibility**: Mix exact matches and pattern matching
- **Reusability**: Save configs for future re-splits
- **Validation**: Reports uncategorized tests so nothing is lost

---

# check-unconverted.sh

Find GLSL files that don't have corresponding WESL conversions.

## Usage

```bash
scripts/check-unconverted.sh              # Show all unconverted files
scripts/check-unconverted.sh --skip-barrels  # Exclude barrel files (import-only)
```

## Options

- `--skip-barrels` or `-s`: Exclude barrel files (files that only contain imports, no implementations)

## What are Barrel Files?

Barrel files are import-only files that re-export functions from other modules. Examples:
- `math.glsl` - Re-exports all math functions
- `color/blend.glsl` - Re-exports all blend modes
- `animation/easing.glsl` - Re-exports all easing functions

These don't need WESL conversions since they contain no actual implementations.

## Example Output

```
./sdf/opExtrude.glsl
./sdf/opRevolution.glsl
./lighting/shadow.glsl
```

---

# find-untested-wesl.sh

Find WESL files that don't have test coverage.

## Usage

```bash
scripts/find-untested-wesl.sh             # List all untested WESL files
scripts/find-untested-wesl.sh --count     # Count by category
```

## Options

- `--count` or `-c`: Show count of untested files by category instead of listing individual files

## Example Output

### List mode (default)
```
color/space/rgb2heat
color/blend/add
math/pow2
```

### Count mode
```
  45 sdf
  23 color
  18 lighting
  12 generative
   8 math
```

---

# find-untested-functions.py

Find LYGIA WESL functions that are not covered by tests at function-level granularity.

This script provides more detailed coverage information than `find-untested-wesl.sh` by:
- Scanning all `.wesl` files and extracting individual function names (via `fn` declarations)
- Scanning all `.test.ts` files and extracting imported `lygia::` modules
- Reporting exactly which functions within each file lack test coverage

## Usage

```bash
python3 scripts/find-untested-functions.py           # List all untested functions
python3 scripts/find-untested-functions.py --count   # Count by category
python3 scripts/find-untested-functions.py --summary # Show summary only
python3 scripts/find-untested-functions.py --files   # List .wesl files with no tests
```

## Options

- (default): Show detailed list of all untested functions with file paths and import statements
- `--count`: Show count of untested functions by category
- `--summary`: Show only total counts and coverage percentage
- `--files`: List only .wesl files with NO test coverage (files where all functions are untested)

## Example Output

### Default mode (detailed list)
```
Untested functions (409 total):

color/ (22 untested):
------------------------------------------------------------
  color/brightnessContrast.wesl
    fn brightnessContrast()
    import lygia::color::brightnessContrast::brightnessContrast
  color/distance.wesl
    fn colorDistance()
    import lygia::color::distance::colorDistance
  color/distance.wesl
    fn colorDistance4()
    import lygia::color::distance::colorDistance4

math/ (105 untested):
------------------------------------------------------------
  math/mix.wesl
    fn mix4()
    import lygia::math::mix::mix4
```

### Count mode
```
Untested functions by category:
Category             Count
===================================
color                   22
color/blend             53
color/space             58
math                   105
space                   66
===================================
Total                  409

Coverage: 433/842 (51.4%)
```

### Summary mode
```
Summary:
  Total functions: 842
  Tested: 433 (51.4%)
  Untested: 409 (48.6%)
```

### Files mode
```
Files with no test coverage (2 files):

color/dither/bayer.wesl
math/aafract.wesl

Total: 2 files with no tests
Coverage: 433/842 (51.4%)
```

**Note**: This shows only files where ALL functions are untested. Files with partial coverage (some functions tested, some not) are excluded.

## Difference from find-untested-wesl.sh

| Feature | find-untested-wesl.sh | find-untested-functions.py |
|---------|----------------------|---------------------------|
| Granularity | File-level | Function-level |
| Details | Shows untested files | Shows untested functions within files |
| Use case | Find files needing any tests | Find specific functions without tests |
| Output | File paths | Function names + import statements |
| --files mode | N/A | Shows only completely untested files |

### Example

For a file with multiple overloads like `color/blend/add.wesl`:
```wesl
fn blendAdd(base: f32, blend: f32) -> f32 { ... }
fn blendAdd3(base: vec3f, blend: vec3f) -> vec3f { ... }
fn blendAdd3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f { ... }
```

**Scenario 1**: Only `blendAdd` and `blendAdd3` are tested
- `find-untested-wesl.sh`: Shows nothing (file has some tests)
- `find-untested-functions.py` (default): Shows "Missing: blendAdd3Opacity"
- `find-untested-functions.py --files`: Shows nothing (file has some tests)

**Scenario 2**: No functions are tested at all
- `find-untested-wesl.sh`: Shows "color/blend/add"
- `find-untested-functions.py` (default): Shows all 3 untested functions
- `find-untested-functions.py --files`: Shows "color/blend/add.wesl"

## Use Cases

- **Comprehensive test coverage** - Ensure all function overloads are tested
- **Test planning** - Generate list of functions needing tests
- **CI/CD integration** - Track coverage metrics over time
- **Documentation** - Know exactly which functions have test validation

---

# extract-tested-functions.sh

Extract and display all tested WESL functions, organized by test file.

## Usage

```bash
scripts/extract-tested-functions.sh
```

## Example Output

```
=== TESTED WESL FUNCTIONS ===

# color-space.test.ts
color/space/rgb2heat
color/space/rgb2xyz
color/space/hsl2rgb

# noise.test.ts
generative/cnoise
generative/snoise
generative/worley
```

## Use Cases

- Verify test coverage for specific modules
- Cross-reference with `find-untested-wesl.sh` results
- Generate documentation of tested functionality
