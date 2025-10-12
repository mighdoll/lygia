import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("diffuseOrenNayar", async () => {
  const src = `
    import lygia::lighting::diffuse::orenNayar::diffuseOrenNayar;

    @compute @workgroup_size(1)
    fn foo() {
      // Test 1: Verify that roughness=0 approximates Lambert diffuse (NoL)
      // Setup: Light at 45° to surface normal
      let L = normalize(vec3f(1.0, 1.0, 1.0));
      let N = vec3f(0.0, 0.0, 1.0);
      let V = vec3f(0.0, 0.0, 1.0);  // View perpendicular to surface
      let NoV = dot(N, V);  // = 1.0
      let NoL = dot(N, L);  // = 1.0/sqrt(3) ≈ 0.5773

      // At roughness=0, Oren-Nayar should approximate Lambert (return NoL)
      let smoothResult = diffuseOrenNayar(L, N, V, NoV, NoL, 0.0);

      // Test 2: Verify roughness effect increases diffuse
      // At roughness=1.0, the A coefficient becomes larger
      let roughResult = diffuseOrenNayar(L, N, V, NoV, NoL, 1.0);

      // Test 3: Verify retroreflection effect
      // When V and L align (maximum retroreflection), roughness should increase brightness
      // Set V = L to get maximum alignment
      let V2 = L;
      let NoV2 = dot(N, V2);
      let retroResult = diffuseOrenNayar(L, N, V2, NoV2, NoL, 1.0);

      // Pack results: [smoothResult, roughResult, retroResult, NoL]
      test::results[0] = vec4f(smoothResult, roughResult, retroResult, NoL);
    }
  `;
  const result = await testCompute(src, "vec4f");

  // Test 1: Smooth (roughness=0) should approximate NoL
  // At roughness=0: A ≈ 1.0 + 0*(terms) ≈ 1.0, B ≈ 0
  // Result ≈ NoL * (1.0 + 0) = NoL
  expectCloseTo([result[0]], [result[3]], 0.01);

  // Test 2: Roughness should increase diffuse contribution
  // At roughness=1.0, A becomes larger (≈ 1.0 + 1.0*(terms) > 1.0)
  expect(result[1]).toBeGreaterThan(result[0]);

  // Test 3: Retroreflection should show maximum brightness
  // When V=L, LoV=1.0, s is maximized, retroreflection is strongest
  expect(result[2]).toBeGreaterThan(result[1]);

  // Test 4: All values should be non-negative
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[2]).toBeGreaterThanOrEqual(0.0);
});
