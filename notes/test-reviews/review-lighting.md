# WESL Lighting Tests Review

## Review Summary

Reviewed 16 tests across 3 lighting test files according to the criteria in test-review.md.

---

### lighting-common.test.ts (9 tests)

- [x] GGX - ⚠️ RANGE-ONLY: Only checks output >= 0, doesn't verify GGX distribution formula behavior with expected value
- [x] GGXPrecise - ⚠️ RANGE-ONLY: Only checks output >= 0, doesn't verify precise GGX calculation or compare with standard GGX
- [x] importanceSamplingGGX - ⚠️ RANGE-ONLY: Only checks length is in [0.5, 1.5], doesn't verify importance sampling properties or direction distribution
- [x] schlick vec3f - ⚠️ RANGE-ONLY: Only checks output is in [0,1] for each component, doesn't verify Schlick approximation formula (f0 + (f90-f0)*(1-VoH)^5)
- [x] schlickVec3 - ⚠️ RANGE-ONLY: Only checks output is in [0,1] for each component, doesn't verify Schlick approximation with vec3f f90 parameter
- [x] schlickF32 - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify Schlick approximation formula behavior
- [x] smithGGXCorrelated - ⚠️ RANGE-ONLY: Only checks output >= 0, doesn't verify Smith GGX visibility term calculation
- [x] smithGGXCorrelated_Fast - ⚠️ RANGE-ONLY: Only checks output >= 0, doesn't verify fast approximation or compare with standard version

**Notes:**
- All 9 tests only validate range constraints (>= 0 or in [0,1])
- None verify the actual mathematical formulas being implemented
- GGX tests could verify the distribution peaks when NoH = 1.0 and decreases with angle
- Schlick tests could verify that at VoH = 0 (grazing angle), result approaches f90, and at VoH = 1 (normal incidence), result approaches f0
- Smith GGX tests could verify visibility decreases with increasing roughness
- importanceSamplingGGX could verify the result is a normalized vector or has specific distribution properties

---

### lighting-diffuse.test.ts (1 test)

- [x] diffuseOrenNayar - ⚠️ RANGE-ONLY: Only checks output >= 0, doesn't verify Oren-Nayar roughness behavior (should approach Lambert when roughness=0, show retroreflection with roughness>0)

**Notes:**
- Could add test cases comparing roughness=0 (should match Lambert ~dot(N,L)) vs roughness=1.0 (showing different behavior)
- Could test retroreflection property where diffuse increases when L, V, and N are coplanar

---

### lighting-misc.test.ts (6 tests)

- [x] fresnel vec3f - ✅ GOOD: Tests Fresnel at normal incidence (NoV=1.0), verifies result ~= f0 (0.04), which is physically correct behavior
- [x] fresnelF32 - ✅ GOOD: Tests Fresnel f32 overload at normal incidence (NoV=1.0), verifies result ~= f0 (0.04)
- [x] fresnelFromVectors - ✅ GOOD: Tests Fresnel from explicit vectors (perpendicular view), verifies result ~= f0, tests convenience overload
- [x] fresnelRoughness - ⚠️ RANGE-ONLY: Only checks output is in [0,1], doesn't verify roughness effect (higher roughness should reduce Fresnel contrast)
- [x] specularCookTorrance - ⚠️ RANGE-ONLY: Only checks output >= 0, doesn't verify Cook-Torrance BRDF formula (D*F*G / (4*NoV*NoL))
- [x] toShininess - ⚠️ RANGE-ONLY: Only checks output is in [0, 500], doesn't verify conversion formula from PBR roughness to Blinn-Phong shininess

**Notes:**
- First 3 fresnel tests are GOOD - they test specific physical property (Fresnel at normal incidence equals f0)
- fresnelRoughness could test that increasing roughness reduces the Fresnel effect
- specularCookTorrance could verify specific BRDF output value or compare rough vs smooth surface
- toShininess could verify inverse relationship with roughness (higher roughness → lower shininess)

---

## Overall Assessment

**Total Tests:** 16
- **Good Tests:** 3 (18.8%)
- **Trivial/Range-Only Tests:** 13 (81.2%)

**Key Issues:**
1. Most tests only validate output is non-negative or in a valid range
2. PBR lighting functions have well-defined mathematical properties that aren't being tested
3. Missing comparisons between related functions (GGX vs GGXPrecise, smithGGX vs smithGGX_Fast)
4. Missing edge case tests (roughness=0, roughness=1, grazing angles, normal incidence)

**Recommendations:**
1. GGX/Smith tests should verify distribution/visibility values at known angles
2. Schlick tests should verify behavior at normal incidence (VoH=1) and grazing angles (VoH=0)
3. Fresnel tests: The existing 3 are excellent examples - could add grazing angle test (NoV→0 should approach 1.0)
4. Add comparative tests (e.g., "GGX with roughness=0 should have sharp peak at NoH=1")
5. Oren-Nayar should verify it approaches Lambert diffuse when roughness=0
