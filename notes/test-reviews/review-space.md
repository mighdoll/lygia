# space.test.ts Review (59 tests)

- [x] cart2polar2 - ✅ GOOD: Tests Cartesian to polar conversion with (1,1) → (π/4, √2), verifies angle and radius calculation
- [x] polar2cart - ✅ GOOD: Tests roundtrip conversion, polar(π/4, √2) → (1,1), validates inverse operation
- [x] center - ✅ GOOD: Tests centering operation (0.5 → 0.0), validates shift to [-0.5, 0.5] range
- [x] center2 - ✅ GOOD: Tests vec2 centering with (0.5, 1.0) → (0.0, 1.0), validates per-component centering
- [x] rotateX3 - ✅ GOOD: Tests 90° rotation around X axis, (1,1,0) → (1,0,-1), validates Y→Z transformation
- [x] rotateY3 - ✅ GOOD: Tests 90° rotation around Y axis, (1,1,0) → (0,1,-1), validates X→Z transformation
- [x] rotateZ3 - ✅ GOOD: Tests 90° rotation around Z axis, (1,0,1) → (0,-1,1), validates X→Y transformation
- [x] uncenter - ✅ GOOD: Tests inverse of center operation (0.0 → 0.5), validates shift back to [0,1] range
- [x] uncenter2 - ✅ GOOD: Tests vec2 uncenter with (-1.0, 1.0) → (0.0, 1.0), validates per-component shift
- [x] flipY2 - ✅ GOOD: Tests Y-axis flip for coordinate systems, (0.5, 0.25) → (0.5, 0.75), validates 1-y operation
- [x] aspect - ✅ GOOD: Tests aspect ratio correction for 1920x1080, verifies x scaling by 16/9 ratio
- [x] equirect2xyz - ✅ GOOD: Tests equirectangular to 3D direction at center (0.5,0.5) → (-1,0,0), validates theta/phi math
- [x] xyz2equirect - ✅ GOOD: Tests 3D direction (+X) to equirect UV, expects (0.5, 0.5), validates atan2/acos conversion
- [x] sqTile - ✅ GOOD: Tests square tiling with (2.5, 3.7), returns fract and floor components [0.5,0.7,2,3]
- [x] checkerTile2 - ✅ GOOD: Tests checkerboard pattern with 3 tile positions, validates even/odd tile coloring (0,1,1,0)
- [x] brickTile2 - ✅ GOOD: Tests brick offset pattern (2.5, 3.7) with row-based horizontal shift, validates [0.0,0.7,3,3]
- [x] triTile - ⚠️ SMOKE-TEST: Only verifies function returns vec4f without crashing, doesn't validate triangle tessellation correctness
- [x] hexTile - ⚠️ SMOKE-TEST: Only verifies function returns vec4f without crashing, doesn't validate hexagonal tiling pattern
- [x] mirrorTile2 - ⚠️ SMOKE-TEST: Only verifies function returns vec4f without crashing, doesn't validate mirroring behavior
- [x] windmillTile2 - ⚠️ SMOKE-TEST: Only verifies function returns vec4f without crashing, doesn't validate windmill rotation pattern
- [x] linearizeDepth - ✅ GOOD: Tests perspective depth linearization with near=0.1, far=100, depth=0.5 → 0.1998, validates formula
- [x] depth2viewZ perspective - ✅ GOOD: Tests depth to view-space Z with near/far planes, validates perspective division formula → -1.98
- [x] viewZ2depth perspective - ✅ GOOD: Tests roundtrip with depth2viewZ, viewZ=-1.98 → depth=0.5, validates inverse operation
- [x] kaleidoscope - ⚠️ SMOKE-TEST: Only verifies function returns vec2f without crashing, doesn't validate kaleidoscope symmetry pattern
- [x] bracketing - ⚠️ SMOKE-TEST: Only verifies function returns BracketingResult struct, doesn't validate axis bracketing behavior
- [x] tbn - ✅ GOOD: Tests tangent-bitangent-normal matrix construction with identity basis vectors, validates mat3 creation
- [x] perspective - ✅ GOOD: Tests perspective projection matrix with 90° FOV and 16:9 aspect, validates [0][0]=9/16=0.5625
- [x] orthographic - ✅ GOOD: Tests orthographic projection matrix construction, validates [0][0]=2/(r-l)=1.0
- [x] decimateNormal - ⚠️ WEAK: Tests normal quantization but only validates output is unit length, doesn't verify quantization steps
- [x] eulerView - ⚠️ SMOKE-TEST: Creates view matrix and applies to vector but only returns dummy value (1,1,1,1), doesn't validate view transform
- [x] lookAt - ⚠️ TRIVIAL: Only creates matrix and returns dummy value (1,1,1,1), doesn't validate look-at transformation
- [x] view2screenPosition - ✅ GOOD: Tests view to screen projection with (0,0,-1) → (0.5,0.5), validates perspective transformation to NDC
- [x] screen2viewPosition - ✅ GOOD: Tests inverse screen to view with depth, validates unprojection formula → (0,0,-1,1.002)
- [x] lookAtView - ⚠️ TRIVIAL: Creates matrix but only returns dummy value (1,1,1,1), doesn't validate camera view transformation
- [x] lookAtViewRoll - ⚠️ TRIVIAL: Creates matrix with roll parameter but only returns dummy value (1,1,1,1), doesn't validate roll rotation
- [x] lookAtViewFromDirection - ⚠️ TRIVIAL: Creates matrix from direction but only returns dummy value (1,1,1,1), doesn't validate directional view
- [x] fisheye2xyz - ✅ GOOD: Tests fisheye projection at (0.75,0.5), validates output is normalized direction vector (length=1)
- [x] fisheye2xyz - division by zero at center - ✅ GOOD: Tests edge case at center (0.5,0.5) where R=0, expects (0,1,0) fallback
- [x] nearest - ⚠️ WEAK: Tests nearest-neighbor sampling but only validates output is close to input, doesn't verify pixel snapping
- [x] ratio - ⚠️ SMOKE-TEST: Only verifies function returns vec2f, doesn't validate aspect ratio adjustment behavior
- [x] rotate - ✅ GOOD: Tests 2D rotation by 90° around (0.5,0.5), point (1.0,0.5) → (0.5,1.0), validates rotation matrix
- [x] rotate_c - ✅ GOOD: Tests 2D rotation with custom center (0,0), point (1,0) rotated 90° → (0,1), validates center parameter
- [x] rotate3 - ⚠️ WEAK: Tests 3D rotation around Z axis but only validates output length=1, doesn't verify rotated coordinates
- [x] scale2 - ✅ GOOD: Tests non-uniform 2D scaling (0.75,0.25) by (2.0,0.5) around center → (1.0,0.375), validates scale formula
- [x] scale2_f - ✅ GOOD: Tests uniform 2D scaling (0.75,0.25) by 2.0 around center → (1.0,0.0), validates scalar scale
- [x] scale3 - ✅ GOOD: Tests 3D scaling with non-uniform factors around (0.5,0.5,0.5) → (1.0,0.375,0.5,0), validates vec3 scaling
- [x] sprite - ✅ GOOD: Tests sprite sheet UV mapping for cell 5 in 4x4 grid, validates output is in [0,1] range
- [x] translate - ✅ GOOD: Tests 3D translation matrix with (10,20,30), validates translation components in last column
- [x] unratio - ✅ GOOD: Tests inverse of ratio adjustment with 1920x1080 aspect, (0.5,0.5) → (0.5,0.5), validates roundtrip

## Summary Statistics

**Total Tests:** 59
**Good Tests (✅):** 36
**Trivial/Smoke Tests (⚠️):** 10
**Weak Tests (⚠️):** 3

**Categories of Issues:**
- **Trivial (dummy returns):** lookAt, eulerView, lookAtView, lookAtViewRoll, lookAtViewFromDirection (5 tests)
- **Smoke tests (no validation):** triTile, hexTile, mirrorTile2, windmillTile2, kaleidoscope, bracketing, ratio (7 tests)
- **Weak validation:** decimateNormal, nearest, rotate3 (3 tests)

## Tests Needing Improvement

1. **triTile, hexTile, mirrorTile2, windmillTile2** - Need specific assertions about tiling patterns, not just "doesn't crash"
2. **lookAt, lookAtView, lookAtViewRoll, lookAtViewFromDirection** - Should verify actual matrix transformations, not return dummy (1,1,1,1)
3. **eulerView** - Should validate Euler angle rotations produce correct view matrix
4. **kaleidoscope** - Should verify symmetry/reflection properties with known input/output pairs
5. **bracketing** - Should validate axis bracketing math with specific expected values
6. **decimateNormal** - Should verify quantization to specific precision steps, not just unit length
7. **nearest** - Should verify pixel center snapping behavior with specific texture dimensions
8. **ratio** - Should validate aspect ratio adjustment with specific before/after coordinates
9. **rotate3** - Should verify rotated coordinates match expected values, not just length preservation
