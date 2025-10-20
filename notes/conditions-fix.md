# WESL Conditions Fix - Dual-Use Problem

## Problem
Several WESL files incorrectly use the same identifier as both:
1. A boolean condition in `@if(FOO)`
2. A constant value (vec2f, mat4x4f, f32, etc.)

This doesn't work in current WESL where conditions are strictly boolean.

## Solution
Use the `constants::` namespace from the linker API to separate boolean conditions from constant values.

### Pattern:
```wesl
// Add imports at top of file
@if(CENTER_2D)
import constants::CENTER_2D;

// Update header comment
options:
    - CENTER_2D: boolean condition to enable custom 2D center (requires constants::CENTER_2D vec2f value from linker)

// Use in code
@if(CENTER_2D)
fn scale2(st: vec2f, s: vec2f) -> vec2f {
    return (st - constants::CENTER_2D) * s + constants::CENTER_2D;
}
```

## Files Fixed
- [x] space/scale.wesl (CENTER_2D, CENTER_3D)
- [x] space/rotate.wesl (CENTER_2D, CENTER_3D, CENTER_4D)

## Files Remaining

### Category 1: CENTER_* constants (4 files)
- [ ] space/rotateX.wesl
  - CENTER_3D (vec3f)
  - CENTER_4D (vec4f)
- [ ] space/rotateY.wesl
  - CENTER_3D (vec3f)
  - CENTER_4D (vec4f)
- [ ] space/rotateZ.wesl
  - CENTER_3D (vec3f)
  - CENTER_4D (vec4f)
- [ ] sdf/rectSDF.wesl
  - CENTER_2D (vec2f)

### Category 2: CAMERA_* constants (4 files)
- [ ] space/screen2viewPosition.wesl
  - CAMERA_PROJECTION_MATRIX (mat4x4f)
  - INVERSE_CAMERA_PROJECTION_MATRIX (mat4x4f)
- [ ] space/view2screenPosition.wesl
  - CAMERA_PROJECTION_MATRIX (mat4x4f)
- [ ] space/depth2viewZ.wesl
  - CAMERA_NEAR_CLIP (f32)
  - CAMERA_FAR_CLIP (f32)
- [ ] space/viewZ2depth.wesl
  - CAMERA_NEAR_CLIP (f32)
  - CAMERA_FAR_CLIP (f32)

### Category 3: Other (1 file)
- [ ] space/bracketing.wesl
  - BRACKETING_ANGLE_DELTA (f32)

## Testing

### Current Status
Testing with `constants::` is currently **blocked** because wesl-debug doesn't yet support the `constants` parameter in its test APIs.

See **notes/test-constants.md** for a proposal to add this support.

### Testing Strategy (once wesl-debug supports constants)
Each fixed file should have a test that:
1. Tests default behavior (condition = false)
2. Tests with constants:: values (condition = true, constants provided via linker)

Example tests are prepared in test/wesl/space.test.ts but commented out until wesl-debug is updated.
