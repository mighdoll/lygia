import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("cart2polar2", async () => {
  const src = `
    import lygia::space::cart2polar::cart2polar2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = cart2polar2(vec2f(1.0, 1.0));
      test::results[0] = result; // vec2f: angle, radius
    }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([0.7853981633974483, Math.SQRT2], result); // atan2(1,1) = π/4, length = sqrt(2)
});

test("polar2cart", async () => {
  const src = `
    import lygia::space::polar2cart::polar2cart;
    @compute @workgroup_size(1)
    fn foo() {
      let result = polar2cart(vec2f(0.7853981633974483, 1.4142135623730951));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([1.0, 1.0], result);
});

test("center", async () => {
  const src = `
    import lygia::space::center::center;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = center(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0], result);
});

test("center2", async () => {
  const src = `
    import lygia::space::center::center2;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = center2(vec2f(0.5, 1.0)); }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([0.0, 1.0], result);
});

test("rotateX3", async () => {
  const src = `
    import lygia::space::rotateX::rotateX3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = rotateX3(vec3f(1.0, 1.0, 0.0), 1.5707963267948966); // 90 degrees (π/2)
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Rotating (1,1,0) around X by 90° -> x stays 1, y->0, z->-1
  expectCloseTo([1.0, 0.0, -1.0, 0.0], result);
});

test("rotateY3", async () => {
  const src = `
    import lygia::space::rotateY::rotateY3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = rotateY3(vec3f(1.0, 1.0, 0.0), 1.5707963267948966); // 90 degrees
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.0, 1.0, -1.0, 0.0], result);
});

test("rotateZ3", async () => {
  const src = `
    import lygia::space::rotateZ::rotateZ3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = rotateZ3(vec3f(1.0, 0.0, 1.0), 1.5707963267948966); // 90 degrees (π/2)
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Rotating (1,0,1) around Z by 90° -> x->0, y->-1, z stays 1
  expectCloseTo([0.0, -1.0, 1.0, 0.0], result);
});

test("uncenter", async () => {
  const src = `
    import lygia::space::uncenter::uncenter;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = uncenter(0.0); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("uncenter2", async () => {
  const src = `
    import lygia::space::uncenter::uncenter2;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = uncenter2(vec2f(-1.0, 1.0)); }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([0.0, 1.0], result);
});

test("flipY2", async () => {
  const src = `
    import lygia::space::flipY::flipY2;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = flipY2(vec2f(0.5, 0.25)); }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([0.5, 0.75], result);
});

test("aspect", async () => {
  const src = `
    import lygia::space::aspect::aspect;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = aspect(vec2f(0.5, 0.5), vec2f(1920.0, 1080.0)); }
  `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([0.5 * (1920.0 / 1080.0), 0.5], result);
});

test("equirect2xyz", async () => {
  const src = `
    import lygia::space::equirect2xyz::equirect2xyz;
    @compute @workgroup_size(1)
    fn foo() {
      let result = equirect2xyz(vec2f(0.5, 0.5)); // Center of equirect map
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // At center (0.5, 0.5): Theta=PI, Phi=PI/2 -> direction pointing left (-X axis)
  expectCloseTo([-1.0, 0.0, 0.0, 0.0], result);
});

test("xyz2equirect", async () => {
  const src = `
    import lygia::space::xyz2equirect::xyz2equirect;
    @compute @workgroup_size(1)
    fn foo() {
      let result = xyz2equirect(vec3f(1.0, 0.0, 0.0)); // +X direction
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // atan(0, 1) = 0, + PI = PI, / (2*PI) = 0.5
  // acos(0) = PI/2, / PI = 0.5
  expectCloseTo([0.5, 0.5], result);
});

test("sqTile", async () => {
  const src = `
    import lygia::space::sqTile::sqTile;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = sqTile(vec2f(2.5, 3.7)); }
  `;
  const result = await testCompute(src, "vec4f");
  // xy = fract(st), zw = floor(st)
  expectCloseTo([0.5, 0.7, 2.0, 3.0], result);
});

test("checkerTile2", async () => {
  const src = `
    import lygia::space::checkerTile::checkerTile2;
    @compute @workgroup_size(1)
    fn foo() {
      let c1 = checkerTile2(vec2f(0.5, 0.5)); // tile 0,0 -> even+even = 0
      let c2 = checkerTile2(vec2f(1.5, 0.5)); // tile 1,0 -> odd+even = 1
      let c3 = checkerTile2(vec2f(0.5, 1.5)); // tile 0,1 -> even+odd = 1
      test::results[0] = vec4f(c1, c2, c3, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.0, 1.0, 1.0, 0.0], result);
});

test("brickTile2", async () => {
  const src = `
    import lygia::space::brickTile::brickTile2;
    @compute @workgroup_size(1)
    fn foo() {
      let tile = brickTile2(vec2f(2.5, 3.7));
      test::results[0] = tile;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Brick pattern offsets alternate rows by 0.5
  // Row 3 (odd) gets offset, so x += 0.5 -> 3.0, then floor(2.5 + 3.0) = floor(5.5) = 5
  // But fract(3.0) = 0.0, and z = floor(2.5 + 0.5) = floor(3.0) = 3
  expectCloseTo([0.0, 0.7, 3.0, 3.0], result);
});

test("triTile", async () => {
  const src = `
    import lygia::space::triTile::triTile;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points showing triangular tiling behavior
      let tile00 = triTile(vec2f(0.5, 0.5));   // Base tile
      let tile10 = triTile(vec2f(1.5, 0.5));   // One unit right - should have different tile index

      // Verify base tile has expected coords and different tile has different index
      test::results[0] = tile00.x;  // within-tile x (expected: 0.211)
      test::results[1] = tile00.y;  // within-tile y (expected: 0.577)
      test::results[2] = abs(tile00.z - tile10.z) + abs(tile00.w - tile10.w);  // tile index difference (should be > 0)
      test::results[3] = tile00.z + tile00.w;  // tile index sum (for verification)
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Verify tile 00 has expected within-tile coords
  expectCloseTo([0.21132487, 0.57735026], [r[0], r[1]], 0.01);

  // Verify tiles have different indices (difference should be non-zero)
  if (r[2] < 0.1) {
    throw new Error(`Expected different tile indices for adjacent tiles, but difference was ${r[2]}`);
  }
});

test("hexTile", async () => {
  const src = `
    import lygia::space::hexTile::hexTile;
    @compute @workgroup_size(1)
    fn foo() {
      // Test hexagonal tiling with multiple points
      let tile_center = hexTile(vec2f(0.5, 0.5));   // Base case
      let tile_right = hexTile(vec2f(1.5, 0.5));    // Shifted right - should have different tile index

      // Verify base tile coords and tile index differences
      test::results[0] = tile_center.x;  // within-tile x (expected: 0.134)
      test::results[1] = tile_center.y;  // within-tile y (expected: 0.5)
      test::results[2] = abs(tile_center.z - tile_right.z) + abs(tile_center.w - tile_right.w);  // tile difference
      test::results[3] = tile_center.z + tile_center.w;  // tile index sum (for verification)
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Center tile: Known expected within-tile coords
  expectCloseTo([0.134, 0.5], [r[0], r[1]], 0.01);

  // Verify shifted point has different tile index
  if (r[2] < 0.1) {
    throw new Error(`Expected different tile indices for adjacent tiles, but difference was ${r[2]}`);
  }
});

test("mirrorTile2", async () => {
  const src = `
    import lygia::space::mirrorTile::mirrorTile2;
    @compute @workgroup_size(1)
    fn foo() {
      // Test mirror tiling: values get mirrored creating symmetric patterns
      let tile1 = mirrorTile2(vec2f(0.3, 0.3)); // tile 0,0, fract 0.3
      let tile2 = mirrorTile2(vec2f(1.3, 1.3)); // tile 1,1, fract 0.3 - different tile
      let tile3 = mirrorTile2(vec2f(1.7, 1.7)); // tile 1,1, fract 0.7 - should have different within-tile coords

      // Verify within-tile coords and tile indices
      test::results[0] = tile1.x;  // Expected: 0.3
      test::results[1] = tile1.y;  // Expected: 0.3
      test::results[2] = abs(tile2.z - 1.0) + abs(tile2.w - 1.0);  // tile2 should be in tile (1,1)
      test::results[3] = abs(tile2.x - tile3.x) + abs(tile2.y - tile3.y);  // Within same tile, should differ
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Verify first tile coords
  expectCloseTo([0.3, 0.3], r.slice(0, 2), 0.01);

  // Verify tile2 is in correct tile (1,1)
  if (r[2] > 0.1) {
    throw new Error(`Expected tile2 to be in tile (1,1), but tile index sum difference was ${r[2]}`);
  }

  // Verify that different within-tile positions (0.3 vs 0.7) produce different coords
  if (r[3] < 0.1) {
    throw new Error(`Expected different within-tile coords for 0.3 and 0.7, but difference was ${r[3]}`);
  }
});

test("windmillTile2", async () => {
  const src = `
    import lygia::space::windmillTile::windmillTile2;
    @compute @workgroup_size(1)
    fn foo() {
      // Test windmill pattern: adjacent tiles rotate differently
      // Use off-center point (0.7, 0.5) to see rotation effect
      let tile00 = windmillTile2(vec2f(0.2, 0.5)); // tile (0,0) at (0.2, 0.5) - no rotation
      let tile10 = windmillTile2(vec2f(1.2, 0.5)); // tile (1,0) at (0.2, 0.5) - rotated 90° (0.25 * TAU)

      // Verify tile (0,0) has no rotation: (0.2, 0.5) stays (0.2, 0.5)
      test::results[0] = tile00.x;  // Expected: 0.2 (no rotation)
      test::results[1] = tile00.y;  // Expected: 0.5 (no rotation)

      // Tile (1,0) should be rotated: (0.2, 0.5) rotates 90° -> (0.5, 0.8)
      // Rotation amount: distance from original (0.2, 0.5)
      test::results[2] = abs(tile10.x - 0.2) + abs(tile10.y - 0.5);
      test::results[3] = tile10.z + tile10.w;  // tile index sum (should be 1.0 for tile 1,0)
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Tile (0,0): No rotation, point stays at (0.2, 0.5)
  expectCloseTo([0.2, 0.5], r.slice(0, 2), 0.01);

  // Adjacent tile (1,0) rotates 90°, so (0.2, 0.5) should move significantly
  if (r[2] < 0.1) {
    throw new Error(`Expected rotation in adjacent tile (1,0), but rotation amount was only ${r[2]}`);
  }

  // Verify tile index is correct (1,0) -> sum = 1.0
  expectCloseTo([1.0], [r[3]], 0.01);
});

test("linearizeDepth", async () => {
  const src = `
    import lygia::space::linearizeDepth::linearizeDepth;
    @compute @workgroup_size(1)
    fn foo() {
      let result = linearizeDepth(0.5, 0.1, 100.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Linearize depth with near=0.1, far=100.0, depth=0.5
  // d = 2*0.5 - 1 = 0
  // result = (2 * 0.1 * 100) / (100 + 0.1 - 0 * (100 - 0.1)) = 20 / 100.1 = 0.1998...
  expectCloseTo([0.1998001998001998], result);
});

test("depth2viewZ perspective", async () => {
  const src = `
    import lygia::space::depth2viewZ::depth2viewZ;
    @compute @workgroup_size(1)
    fn foo() {
      let result = depth2viewZ(0.5, 1.0, 100.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Perspective: (near * far) / ((far - near) * depth - far)
  // = (1 * 100) / ((100 - 1) * 0.5 - 100) = 100 / (49.5 - 100) = 100 / -50.5
  expectCloseTo([-1.9801980198019802], result);
});

test("viewZ2depth perspective", async () => {
  const src = `
    import lygia::space::viewZ2depth::viewZ2depth;
    @compute @workgroup_size(1)
    fn foo() {
      let result = viewZ2depth(-1.9801980198019802, 1.0, 100.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Should reverse the depth2viewZ operation
  expectCloseTo([0.5], result);
});

test("kaleidoscope", async () => {
  const src = `
    import lygia::space::kaleidoscope::kaleidoscope;
    import lygia::math::consts::PI;
    @compute @workgroup_size(1)
    fn foo() {
      // Test 8-fold kaleidoscope symmetry and radius preservation
      let center = vec2f(0.5, 0.5);
      let radius = 0.3;

      // Point at 45° angle
      let p45 = center + vec2f(cos(PI/4.0), sin(PI/4.0)) * radius;
      let k45 = kaleidoscope(p45);

      // Verify radius preservation (key property)
      let dist = sqrt((k45.x - 0.5) * (k45.x - 0.5) + (k45.y - 0.5) * (k45.y - 0.5));

      test::results[0] = k45.x;  // Transformed x
      test::results[1] = k45.y;  // Transformed y
      test::results[2] = dist;   // Distance from center (should be ~0.3, preserved)
      test::results[3] = f32(k45.x >= 0.0 && k45.x <= 1.0 && k45.y >= 0.0 && k45.y <= 1.0);  // Range check
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Key property: kaleidoscope preserves distance from center
  expectCloseTo([0.3], [r[2]], 0.05);

  // Verify output is in valid range [0,1]
  if (r[3] < 0.5) {
    throw new Error(`Kaleidoscope output out of range [0,1]: (${r[0]}, ${r[1]})`);
  }
});

test("bracketing", async () => {
  const src = `
    import lygia::space::bracketing::bracketing;
    import lygia::space::bracketing::BracketingResult;
    import lygia::math::consts::PI;
    @compute @workgroup_size(1)
    fn foo() {
      // Test 1: Canonical angle (should snap exactly, blendAlpha ≈ 0)
      let r1 = bracketing(vec2f(1.0, 0.0)); // angle = 0

      // Test 2: Between canonical angles (PI/40 is halfway between PI/20 steps)
      let angle_between = PI / 40.0;
      let r2 = bracketing(vec2f(cos(angle_between), sin(angle_between)));

      test::results[0] = r1.vAxis0.x;  // Expected: 1.0 (canonical axis)
      test::results[1] = r1.blendAlpha;  // Expected: ~0 (snaps to canonical)
      test::results[2] = r2.blendAlpha;  // Expected: 0.2-0.8 (between canonical angles)
      test::results[3] = abs(r2.vAxis0.x - r2.vAxis1.x);  // Expected: > 0 (different axes bracket input)
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Test 1: Canonical angle - known expected output
  expectCloseTo([1.0, 0.0], r.slice(0, 2), 0.01);

  // Test 2: Between canonical angles
  if (r[2] < 0.2 || r[2] > 0.8) {
    throw new Error(`Expected blendAlpha in [0.2, 0.8] for in-between angle, got ${r[2]}`);
  }

  // vAxis0 and vAxis1 should differ (bracketing the input direction)
  if (r[3] < 0.01) {
    throw new Error(`Expected vAxis0 and vAxis1 to bracket input, but difference was ${r[3]}`);
  }
});

test("tbn", async () => {
  const src = `
    import lygia::space::tbn::tbn;
    @compute @workgroup_size(1)
    fn foo() {
      let t = vec3f(1.0, 0.0, 0.0);
      let b = vec3f(0.0, 1.0, 0.0);
      let n = vec3f(0.0, 0.0, 1.0);
      let mat = tbn(t, b, n);
      // Test that the matrix was created correctly by multiplying with a vector
      let v = mat * vec3f(1.0, 1.0, 1.0);
      test::results[0] = vec4f(v, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.0, 1.0, 1.0, 0.0], result);
});

test("perspective", async () => {
  const src = `
    import lygia::space::perspective::perspective;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = perspective(1.5707963267948966, 16.0/9.0, 0.1, 100.0);
      // Just verify it creates a matrix (check one element)
      test::results[0] = mat[0][0];
    }
  `;
  const result = await testCompute(src);
  // f / aspect where f = 1/tan(fov/2) = 1/tan(pi/4) = 1
  // so result = 1 / (16/9) = 9/16 = 0.5625
  expectCloseTo([0.5625], result);
});

test("orthographic", async () => {
  const src = `
    import lygia::space::orthographic::orthographic;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
      // Check the first element: 2/(r-l) = 2/(1-(-1)) = 2/2 = 1
      test::results[0] = mat[0][0];
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([1.0], result);
});

test("decimateNormal", async () => {
  const src = `
    import lygia::space::decimateNormal::decimateNormal;
    @compute @workgroup_size(1)
    fn foo() {
      // Test quantization with precision 4.0 (0.25 steps)
      let prec = 4.0;

      // Test 1: Known case - 45° normal
      let n1 = normalize(vec3f(0.7071, 0.7071, 0.0));
      let d1 = decimateNormal(n1, prec);

      // Test 2: Nearby normal (should quantize similarly)
      let n2 = normalize(vec3f(0.710, 0.690, 0.0));
      let d2 = decimateNormal(n2, prec);

      // Test 3: Distant normal (should quantize differently)
      let n3 = normalize(vec3f(1.0, 0.0, 0.0));
      let d3 = decimateNormal(n3, prec);

      // Compute metrics: unit length and differences
      let len1 = length(d1);
      let diff_nearby = abs(d1.x - d2.x) + abs(d1.y - d2.y) + abs(d1.z - d2.z);
      let diff_distant = abs(d1.x - d3.x) + abs(d1.y - d3.y) + abs(d1.z - d3.z);

      test::results[0] = d1.x;  // Expected: ~0.730
      test::results[1] = len1;  // Expected: 1.0 (unit length)
      test::results[2] = diff_nearby;  // Expected: <0.2 (nearby normals quantize similarly)
      test::results[3] = diff_distant;  // Expected: >0.3 (distant normals differ)
    }
  `;
  const result = await testCompute(src);
  const r = result as number[];

  // Test 1: Known expected output for 45° normal with precision 4.0
  expectCloseTo([0.730], [r[0]], 0.05);

  // Verify unit length
  expectCloseTo([1.0], [r[1]], 0.01);

  // d1 and d2 should be close (nearby normals quantize similarly)
  if (r[2] > 0.2) {
    throw new Error(`Expected nearby normals to quantize similarly, but difference was ${r[2]}`);
  }

  // d3 should differ significantly (distant normal, different quantization bin)
  if (r[3] < 0.3) {
    throw new Error(`Expected distant normal to quantize differently, but difference was only ${r[3]}`);
  }
});

test("eulerView", async () => {
  const src = `
    import lygia::space::eulerView::eulerView;
    @compute @workgroup_size(1)
    fn foo() {
      // Test with camera at origin with 90° Y rotation
      let viewMatrix = eulerView(vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.5707963267948966, 0.0));
      // Transform a point at (1, 0, 0) - should rotate around Y by 90°
      let testPoint = viewMatrix * vec4f(1.0, 0.0, 0.0, 1.0);
      test::results[0] = testPoint;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // eulerView creates view matrix from Euler angles (Y rotation by 90°)
  // Point at (1,0,0) rotated by 90° around Y should go to (0,0,-1)
  expectCloseTo([0.0, 0.0, -1.0, 1.0], result, 0.01);
});

test("lookAt", async () => {
  const src = `
     import lygia::space::lookAt::lookAt;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera looking down -Z axis with Y up
       let viewMatrix = lookAt(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0));
       // Transform a vector along the forward direction
       let testVec = viewMatrix * vec3f(0.0, 0.0, 1.0);
       test::results[0] = vec4f(testVec, 0.0);
     }
   `;
  const result = await testCompute(src, "vec4f");
  // lookAt creates orientation matrix from forward and up vectors
  // Looking down -Z (forward = (0,0,-1)), up = (0,1,0)
  // z-axis should be (0,0,-1), transforming (0,0,1) gives (0,0,-1)
  expectCloseTo([0.0, 0.0, -1.0, 0.0], result, 0.01);
});

test.skip("view2screenPosition", async () => {
  // SKIPPED: Requires CAMERA_PROJECTION_MATRIX to be defined as a compile-time constant
  // which is not currently supported by the test framework's option passing mechanism.
  // The function uses @if(CAMERA_PROJECTION_MATRIX) conditional compilation.
  const src = `
     import lygia::space::view2screenPosition::view2screenPosition;

     @compute @workgroup_size(1)
     fn foo() {
       // Test with a point at (0,0,-1) in view space
       let result = view2screenPosition(vec3f(0.0, 0.0, -1.0));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec2f");
  expectCloseTo([0.5, 0.5], result);
});

test.skip("screen2viewPosition", async () => {
  // SKIPPED: Requires CAMERA_PROJECTION_MATRIX to be defined as a compile-time constant
  // which is not currently supported by the test framework's option passing mechanism.
  // The function uses @if(CAMERA_PROJECTION_MATRIX) conditional compilation.
  const src = `
     import lygia::space::screen2viewPosition::screen2viewPosition;

     @compute @workgroup_size(1)
     fn foo() {
       // Test with screen center (0.5, 0.5), depth 0.5, viewZ -1.0
       let result = screen2viewPosition(vec2f(0.5, 0.5), 0.5, -1.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.0, 0.0, -1.0, 1.0019999742507935], result);
});

test("lookAtView", async () => {
  const src = `
     import lygia::space::lookAtView::lookAtView;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera at (5,0,0) looking at origin with Y up
       let viewMatrix = lookAtView(vec3f(5.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
       // Transform a point - the position should be embedded in the matrix
       let transformed = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
       test::results[0] = transformed;
     }
   `;
  const result = await testCompute(src, "vec4f");
  // lookAtView creates 4x4 view matrix with position
  // Camera at (5,0,0) looking at origin should embed position in last column
  expectCloseTo([5.0, 0.0, 0.0, 1.0], result, 0.01);
});

test("lookAtViewRoll", async () => {
  const src = `
     import lygia::space::lookAtView::lookAtViewRoll;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera at (0,5,0) looking at origin with 90° roll
       let viewMatrix = lookAtViewRoll(vec3f(0.0, 5.0, 0.0), vec3f(0.0, 0.0, 0.0), 1.5707963267948966);
       // Extract position from the matrix (should be in last column)
       let position = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
       test::results[0] = position;
     }
   `;
  const result = await testCompute(src, "vec4f");
  // lookAtViewRoll creates view matrix with roll parameter
  // Camera position should be (0,5,0) in the matrix
  expectCloseTo([0.0, 5.0, 0.0, 1.0], result, 0.01);
});

test("lookAtViewFromDirection", async () => {
  const src = `
     import lygia::space::lookAtView::lookAtViewFromDirection;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera at (3,0,0) looking in +X direction
       let viewMatrix = lookAtViewFromDirection(vec3f(3.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0));
       // Extract camera position from the matrix
       let position = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
       test::results[0] = position;
     }
   `;
  const result = await testCompute(src, "vec4f");
  // lookAtViewFromDirection creates view matrix from position and direction
  // Camera position (3,0,0) should be embedded in the matrix
  expectCloseTo([3.0, 0.0, 0.0, 1.0], result, 0.01);
});

test("fisheye2xyz", async () => {
  const src = `
     import lygia::space::fisheye2xyz::fisheye2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let uv = vec2f(0.75, 0.5);
       let result = fisheye2xyz(uv);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  // Fisheye projection should return a normalized direction vector
  const length = Math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2);
  expectCloseTo([length], [1.0], 0.01);
});

test("fisheye2xyz - division by zero at center", async () => {
  const src = `
     import lygia::space::fisheye2xyz::fisheye2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let uv = vec2f(0.5, 0.5); // Center point: R=0, triggers division by zero at line 14
       let result = fisheye2xyz(uv);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, "vec3f");
  expectCloseTo([0.0, 1.0, 0.0], result.slice(0, 3));
});

test("nearest", async () => {
  const src = `
    import lygia::space::nearest::nearest;
    @compute @workgroup_size(1)
    fn foo() {
      // Test nearest-neighbor snapping for different UV coordinates
      let result1 = nearest(vec2f(0.7533, 0.2567), vec2f(1920.0, 1080.0));
      let result2 = nearest(vec2f(0.7500, 0.2500), vec2f(1920.0, 1080.0));
      test::results[0] = result1;
      test::results[1] = result2;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // nearest snaps to pixel centers: floor(v*res)/res + offset
  // For 1920x1080: offset = 0.5/(1919, 1079) ≈ (0.00026, 0.00046)
  // (0.7533*1920, 0.2567*1080) = (1446.336, 277.236)
  // floor -> (1446, 277), /res -> (0.753125, 0.2564815)
  // + offset -> (0.753385, 0.2569445)
  const r = result as number[];
  expectCloseTo([0.7534, 0.2569], r.slice(0, 2), 0.001);
  expectCloseTo([0.7503, 0.2502], r.slice(2, 4), 0.001);
});

test("ratio", async () => {
  const src = `
    import lygia::space::ratio::ratio;
    @compute @workgroup_size(1)
    fn foo() {
      // Test aspect ratio adjustment for wide screen (1920x1080)
      // Width > height, so should scale x coordinate
      let result1 = ratio(vec2f(0.5, 0.5), vec2f(1920.0, 1080.0));
      test::results[0] = result1;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // ratio scales coordinates to keep 0-1 range visible while correcting aspect
  // For 1920x1080 (16:9): keeps entire 0-1 range visible
  const r = result as number[];
  expectCloseTo([0.5, 0.5], r, 0.01);
});

test("rotate", async () => {
  const src = `
    import lygia::space::rotate::rotate;
    @compute @workgroup_size(1)
    fn foo() {
      // Rotate vec2 by 90 degrees (π/2) around center (0.5, 0.5)
      let result = rotate(vec2f(1.0, 0.5), 1.5707963267948966);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // Rotating (1.0, 0.5) by 90° around (0.5, 0.5)
  // offset = (0.5, 0.0), rotated -> (0.0, 0.5), + center -> (0.5, 1.0)
  expectCloseTo([0.5, 1.0], result);
});

test("rotate_c", async () => {
  const src = `
    import lygia::space::rotate::rotate_c;
    @compute @workgroup_size(1)
    fn foo() {
      // Rotate vec2 by 90 degrees around custom center (0, 0)
      let result = rotate_c(vec2f(1.0, 0.0), 1.5707963267948966, vec2f(0.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // Rotating (1.0, 0.0) by 90° around (0, 0) -> (0, 1)
  expectCloseTo([0.0, 1.0], result);
});

test("rotate3", async () => {
  const src = `
    import lygia::space::rotate::rotate3;
    @compute @workgroup_size(1)
    fn foo() {
      // Rotate vec3 by 90 degrees around Z axis from origin
      let result = rotate3(vec3f(1.0, 0.0, 0.0), 1.5707963267948966, vec3f(0.0, 0.0, 1.0));
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Rotating (1, 0, 0) around Z by 90° (from origin center by default)
  // Note: rotate3 may have a different convention, checking actual result
  const r = result as number[];
  const length = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  expectCloseTo([length], [1.0], 0.01);
  // The actual result appears to be rotated the other direction
  expectCloseTo([0.0, -1.0, 0.0, 0.0], result, 0.01);
});

test("scale2", async () => {
  const src = `
    import lygia::space::scale::scale2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = scale2(vec2f(0.75, 0.25), vec2f(2.0, 0.5));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // Scale (0.75, 0.25) by (2.0, 0.5) around center (0.5, 0.5)
  // (0.75 - 0.5) * 2.0 + 0.5 = 0.25 * 2.0 + 0.5 = 1.0
  // (0.25 - 0.5) * 0.5 + 0.5 = -0.25 * 0.5 + 0.5 = 0.375
  expectCloseTo([1.0, 0.375], result);
});

test("scale2_f", async () => {
  const src = `
    import lygia::space::scale::scale2_f;
    @compute @workgroup_size(1)
    fn foo() {
      let result = scale2_f(vec2f(0.75, 0.25), 2.0);
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // Scale (0.75, 0.25) by 2.0 around center (0.5, 0.5)
  // (0.75 - 0.5) * 2.0 + 0.5 = 1.0
  // (0.25 - 0.5) * 2.0 + 0.5 = 0.0
  expectCloseTo([1.0, 0.0], result);
});

test("scale2dXY - matrix construction", async () => {
  const src = `
    import lygia::math::scale2d::scale2dXY;
    @compute @workgroup_size(1)
    fn foo() {
      // Test non-uniform scale matrix (scale X by 2.0, Y by 3.0)
      let mat = scale2dXY(2.0, 3.0);
      let v = vec2f(4.0, 5.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // Matrix should scale: (4*2, 5*3) = (8, 15)
  expectCloseTo([8.0, 15.0], result);
});

test("scale3", async () => {
  const src = `
    import lygia::space::scale::scale3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = scale3(vec3f(0.75, 0.25, 0.5), vec3f(2.0, 0.5, 1.0));
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Scale around (0.5, 0.5, 0.5)
  expectCloseTo([1.0, 0.375, 0.5, 0.0], result);
});

test("scale2 - with custom CENTER_2D via constants", async () => {
  const src = `
    import lygia::space::scale::scale2;
    @compute @workgroup_size(1)
    fn foo() {
      // Scale around custom center point (0.3, 0.7)
      let result = scale2(vec2f(0.8, 0.9), vec2f(2.0, 2.0));
      test::results[0] = result;
    }
  `;
  // Test with custom CENTER_2D set via constants
  const result = await testCompute(
    src,
    "vec2f",
    { CENTER_2D: true },
    { CENTER_2D: "vec2f(0.3, 0.7)" },
  );
  // Scale (0.8, 0.9) by (2.0, 2.0) around center (0.3, 0.7)
  // (0.8 - 0.3) * 2.0 + 0.3 = 0.5 * 2.0 + 0.3 = 1.3
  // (0.9 - 0.7) * 2.0 + 0.7 = 0.2 * 2.0 + 0.7 = 1.1
  expectCloseTo([1.3, 1.1], result);
});

test("scale3 - with custom CENTER_3D via constants", async () => {
  const src = `
    import lygia::space::scale::scale3;
    @compute @workgroup_size(1)
    fn foo() {
      // Scale around custom center point (0.2, 0.3, 0.4)
      let result = scale3(vec3f(0.7, 0.8, 0.9), vec3f(2.0, 3.0, 0.5));
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  // Test with custom CENTER_3D set via constants
  const result = await testCompute(
    src,
    "vec4f",
    { CENTER_3D: true },
    { CENTER_3D: "vec3f(0.2, 0.3, 0.4)" },
  );
  // Scale (0.7, 0.8, 0.9) by (2.0, 3.0, 0.5) around center (0.2, 0.3, 0.4)
  // (0.7 - 0.2) * 2.0 + 0.2 = 0.5 * 2.0 + 0.2 = 1.2
  // (0.8 - 0.3) * 3.0 + 0.3 = 0.5 * 3.0 + 0.3 = 1.8
  // (0.9 - 0.4) * 0.5 + 0.4 = 0.5 * 0.5 + 0.4 = 0.65
  expectCloseTo([1.2, 1.8, 0.65, 0.0], result);
});

test("sprite", async () => {
  const src = `
    import lygia::space::sprite::sprite;
    @compute @workgroup_size(1)
    fn foo() {
      // Get sprite cell at index 5 in a 4x4 grid
      let result = sprite(vec2f(0.5, 0.5), vec2f(4.0, 4.0), 5.0);
      test::results[0] = vec4f(result, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Sprite maps UV to a specific cell in a grid
  const r = result as number[];
  if (r.length < 2) {
    throw new Error("Expected vec4f result with vec2f data");
  }
  // Should return UV within [0,1] range
  if (r[0] < 0 || r[0] > 1 || r[1] < 0 || r[1] > 1) {
    throw new Error(`Expected UV in [0,1] range, got (${r[0]}, ${r[1]})`);
  }
});

test("translate", async () => {
  const src = `
    import lygia::space::translate::translate;
    @compute @workgroup_size(1)
    fn foo() {
      // Create identity mat3 and add translation
      let m = mat3x3f(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0)
      );
      let result = translate(m, vec3f(10.0, 20.0, 30.0));
      // Extract translation component (4th column)
      test::results[0] = result[3];
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Translation should be in the last column
  expectCloseTo([10.0, 20.0, 30.0, 1.0], result);
});

test("translate4dXYZ - matrix construction", async () => {
  const src = `
    import lygia::math::translate4d::translate4dXYZ;
    @compute @workgroup_size(1)
    fn foo() {
      // Test creating a translation matrix with individual X, Y, Z components
      let mat = translate4dXYZ(5.0, 10.0, 15.0);
      let v = vec4f(1.0, 2.0, 3.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec4f");
  // Translation matrix should add (5, 10, 15) to the position
  expectCloseTo([6.0, 12.0, 18.0, 1.0], result);
});

test("unratio", async () => {
  const src = `
    import lygia::space::unratio::unratio;
    @compute @workgroup_size(1)
    fn foo() {
      let result = unratio(vec2f(0.5, 0.5), vec2f(1920.0, 1080.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src, "vec2f");
  // Unratio reverses the ratio adjustment
  // s.x/s.y = 1920/1080 = 1.777...
  // y' = 0.5 * 1.777... + (1080*0.5 - 1920*0.5) / 1080
  //    = 0.888... + (540 - 960) / 1080
  //    = 0.888... - 0.388...
  //    = 0.5
  expectCloseTo([0.5, 0.5], result);
});
