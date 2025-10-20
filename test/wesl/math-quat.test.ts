import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("quatAdd", async () => {
  const src = `
    import lygia::math::quat::add::quatAdd;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatAdd(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(0.5, 0.5, 0.5, 0.5)); }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([1.5, 2.5, 3.5, 4.5], result);
});

test("quatSub", async () => {
  const src = `
    import lygia::math::quat::sub::quatSub;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatSub(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(0.5, 0.5, 0.5, 0.5)); }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.5, 1.5, 2.5, 3.5], result);
});

test("quatMul", async () => {
  const src = `
    import lygia::math::quat::mul::quatMul;
    @compute @workgroup_size(1)
    fn foo() {
      let q1 = normalize(vec4f(1.0, 0.0, 0.0, 1.0));
      let q2 = normalize(vec4f(0.0, 1.0, 0.0, 1.0));
      test::results[0] = quatMul(q1, q2);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.5, 0.5, 0.5, 0.5], result);
});

test("quatConj", async () => {
  const src = `
    import lygia::math::quat::conj::quatConj;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatConj(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([-1.0, -2.0, -3.0, 4.0], result);
});

test("quatNorm", async () => {
  const src = `
    import lygia::math::quat::norm::quatNorm;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatNorm(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.1826, 0.3651, 0.5477, 0.7303], result);
});

test("quatLength", async () => {
  const src = `
    import lygia::math::quat::length::quatLength;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatLength(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await testCompute(src);
  expectCloseTo([5.4772258], result);
});

test("quatLengthSq", async () => {
  const src = `
    import lygia::math::quat::lengthSq::quatLengthSq;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatLengthSq(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await testCompute(src);
  expectCloseTo([30.0], result);
});

test("quatIdentity", async () => {
  const src = `
    import lygia::math::quat::identity::QUAT_IDENTITY;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = QUAT_IDENTITY; }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.0, 0.0, 0.0, 1.0], result);
});

test("quatLerp", async () => {
  const src = `
    import lygia::math::quat::lerp::quatLerp;
    @compute @workgroup_size(1)
    fn foo() {
      let q1 = normalize(vec4f(1.0, 0.0, 0.0, 1.0));
      let q2 = normalize(vec4f(0.0, 1.0, 0.0, 1.0));
      test::results[0] = quatLerp(q1, q2, 0.5);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.4082, 0.4082, 0.0, 0.8165], result);
});

test("quat2mat3", async () => {
  const src = `
    import lygia::math::consts::INV_SQRT2;
    import lygia::math::quat::quat2mat3::quat2mat3;
    @compute @workgroup_size(1)
    fn foo() {
      let q = normalize(vec4f(0.0, 0.0, INV_SQRT2, INV_SQRT2));
      let m = quat2mat3(q);
      test::results[0] = vec4f(m[0][0], m[1][1], m[2][2], 0.0);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.0, 0.0, 1.0, 0.0], result);
});

test("quat2mat4", async () => {
  const src = `
    import lygia::math::consts::INV_SQRT2;
    import lygia::math::quat::quat2mat4::quat2mat4;
    @compute @workgroup_size(1)
    fn foo() {
      let q = normalize(vec4f(0.0, 0.0, INV_SQRT2, INV_SQRT2));
      let m = quat2mat4(q);
      test::results[0] = vec4f(m[0][0], m[1][1], m[2][2], m[3][3]);
    }
  `;
  const result = await testCompute(src, "vec4f");
  expectCloseTo([0.0, 0.0, 1.0, 1.0], result);
});
