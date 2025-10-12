# Barrel Files (Import-Only Files)

This document lists all GLSL files that are "barrel files" - files containing only `#include` statements with no actual function implementations. These files don't need WESL equivalents because WESL users can import specific functions directly.

**Total barrel files identified: 21**

---

## Root Level Barrels - 2 files

- `math.glsl` - Aggregates all math functions
- `sdf.glsl` - Aggregates all SDF functions

---

## Animation Barrels - 12 files

All animation files are barrel files that aggregate easing functions:

### Animation Main
- `animation/easing.glsl` - Top-level aggregator

### Animation/Easing - 11 files
- `animation/easing/back.glsl`
- `animation/easing/bounce.glsl`
- `animation/easing/circular.glsl`
- `animation/easing/cubic.glsl`
- `animation/easing/elastic.glsl`
- `animation/easing/exponential.glsl`
- `animation/easing/linear.glsl`
- `animation/easing/quadratic.glsl`
- `animation/easing/quartic.glsl`
- `animation/easing/quintic.glsl`
- `animation/easing/sine.glsl`

**Note**: All easing functions have individual implementations (e.g., `cubicIn.glsl`, `cubicOut.glsl`, `cubicInOut.glsl`) which DO need conversion.

---

## Color Barrels - 3 files

- `color/blend.glsl` - Aggregates blend modes
- `color/layer.glsl` - Aggregates layer blend modes
- `color/composite.glsl` - Aggregates composite operations

---

## Geometry Barrels - 2 files

- `geometry/aabb.glsl` - Aggregates AABB functions
- `geometry/triangle.glsl` - Aggregates triangle functions

---

## Sample Barrels - 1 file

- `sample.glsl` - Aggregates sampling functions

---

## Color/Palette Barrels - 1 file

- `color/palette/wada.glsl` - Aggregates Wada palette functions

---

## Usage

To get accurate conversion statistics excluding barrel files:

```bash
# Count unconverted files excluding barrels
./check-unconverted.sh | grep -v -F -f <(cat <<'EOF'
math.glsl
sdf.glsl
animation/easing.glsl
animation/easing/back.glsl
animation/easing/bounce.glsl
animation/easing/circular.glsl
animation/easing/cubic.glsl
animation/easing/elastic.glsl
animation/easing/exponential.glsl
animation/easing/linear.glsl
animation/easing/quadratic.glsl
animation/easing/quartic.glsl
animation/easing/quintic.glsl
animation/easing/sine.glsl
color/blend.glsl
color/layer.glsl
color/composite.glsl
geometry/aabb.glsl
geometry/triangle.glsl
sample.glsl
color/palette/wada.glsl
EOF
) | wc -l
```

Or use the updated check-unconverted.sh with `--skip-barrels` flag.

---

## Why Skip Barrel Files?

In GLSL, barrel files are useful because `#include` statements are processed by the preprocessor. However, in WESL:

1. **Direct imports are preferred**: Users can write `import lygia::math::saturate` instead of importing a barrel file
2. **No preprocessor**: WESL uses module imports, not `#include`
3. **Tree-shaking friendly**: Importing specific functions is better for optimization
4. **Less maintenance**: No need to maintain aggregator files

Therefore, barrel files can be safely skipped during conversion.
