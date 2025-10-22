import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("diffuseOrenNayar", async () => {
  const src = `
    import lygia::lighting::diffuse::orenNayar::diffuseOrenNayar;

    @compute @workgroup_size(1)
    fn foo() {
      // Test 1: Roughness=0 should equal Lambert diffuse (NoL)
      // L at 45° to surface normal
      let L = normalize(vec3f(1.0, 1.0, 1.0));
      let N = vec3f(0.0, 0.0, 1.0);
      let V = vec3f(0.0, 0.0, 1.0);
      let NoV = dot(N, V);  // = 1.0
      let NoL = dot(N, L);  // = 1/sqrt(3) ≈ 0.5773502691896258

      // At roughness=0: sigma2=0, A=1.0, B=0.0
      // Result = NoL * (1.0 + 0) = NoL ≈ 0.5773502691896258
      let smoothResult = diffuseOrenNayar(L, N, V, NoV, NoL, 0.0);

      // Test 2: Roughness=1.0 with perpendicular view
      // sigma2 = 1.0
      // A = 1.0 + 1.0 * (1.0/(1.0+0.13) + 0.5/(1.0+0.33))
      //   = 1.0 + 1.0 * (0.884955752 + 0.375939849)
      //   = 1.0 + 1.260895601 = 2.260895601
      // LoV = dot(L,V) = NoL = 1/sqrt(3)
      // s = LoV - NoL*NoV = 1/sqrt(3) - 1/sqrt(3)*1.0 = 0
      // t = mix(1.0, max(NoL,NoV), step(0.0,s)) = mix(1.0, 1.0, 0.0) = 1.0
      // B = 0.45 * 1.0 / (1.0 + 0.09) = 0.412844037
      // Result = NoL * (A + B * 0 / 1.0) = NoL * 2.260895601 ≈ 1.3053568
      let roughResult = diffuseOrenNayar(L, N, V, NoV, NoL, 1.0);

      // Test 3: Retroreflection (V = L, roughness=1.0)
      // NoV = NoL = 1/sqrt(3)
      // LoV = dot(L,L) = 1.0
      // s = 1.0 - (1/sqrt(3))*(1/sqrt(3)) = 1.0 - 1/3 = 2/3 ≈ 0.6666667
      // t = mix(1.0, max(NoL,NoV), step(0.0,s))
      //   = mix(1.0, 1/sqrt(3), 1.0) = 1/sqrt(3) ≈ 0.5773502691896258
      // A = 2.260895601 (same as test 2)
      // B = 0.412844037 (same as test 2)
      // Result = NoL * (A + B * s / t)
      //        = (1/sqrt(3)) * (2.260895601 + 0.412844037 * (2/3) / (1/sqrt(3)))
      //        = 0.5773502691896258 * (2.260895601 + 0.412844037 * 1.1547005383792517)
      //        = 0.5773502691896258 * (2.260895601 + 0.4767312946587544)
      //        = 0.5773502691896258 * 2.7376268956587544
      //        ≈ 1.5806274
      let V2 = L;
      let NoV2 = dot(N, V2);
      let retroResult = diffuseOrenNayar(L, N, V2, NoV2, NoL, 1.0);

      test::results[0] = vec4f(smoothResult, roughResult, retroResult, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Expected values calculated manually from the Oren-Nayar formula:
  const NoL = 1.0 / Math.sqrt(3); // ≈ 0.5773502691896258

  // Test 1: roughness=0 → result = NoL
  expectCloseTo([NoL], [result[0]], 1e-5);

  // Test 2: roughness=1.0, perpendicular view
  // A = 1.0 + 1.0 * (1.0/(1.0+0.13) + 0.5/(1.0+0.33)) ≈ 2.260895601
  // s = 0, so Result = NoL * A
  const A = 1.0 + 1.0 * (1.0 / (1.0 + 0.13) + 0.5 / (1.0 + 0.33));
  const expectedRoughResult = NoL * A;
  expectCloseTo([expectedRoughResult], [result[1]], 1e-5);

  // Test 3: retroreflection (V=L, roughness=1.0)
  // s = 1.0 - 1/3 = 2/3, t = 1/sqrt(3)
  // B = 0.45 * 1.0 / (1.0 + 0.09)
  // Result = NoL * (A + B * s / t)
  const B = (0.45 * 1.0) / (1.0 + 0.09);
  const s = 1.0 - NoL * NoL;
  const t = NoL;
  const expectedRetroResult = NoL * (A + (B * s) / t);
  expectCloseTo([expectedRetroResult], [result[2]], 1e-5);

  // Verify relationships still hold as sanity check
  expect(result[1]).toBeGreaterThan(result[0]); // Rough > smooth
  expect(result[2]).toBeGreaterThan(result[1]); // Retro > rough
});
