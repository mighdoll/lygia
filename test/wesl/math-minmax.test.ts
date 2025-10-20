import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

// Min/max functions

test("mmax2", async () => {
  const src = `
    import lygia::math::mmax::mmax2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmax2(vec2f(3.0, 7.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([7.0], result);
});

test("mmax3", async () => {
  const src = `
    import lygia::math::mmax::mmax3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmax3(vec3f(3.0, 7.0, 5.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([7.0], result);
});

test("mmin2", async () => {
  const src = `
    import lygia::math::mmin::mmin2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmin2(vec2f(3.0, 7.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([3.0], result);
});

test("mmin3", async () => {
  const src = `
    import lygia::math::mmin::mmin3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = mmin3(vec3f(3.0, 7.0, 5.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  expectCloseTo([3.0], result);
});
