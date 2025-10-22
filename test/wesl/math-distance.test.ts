import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

// Distance and length functions

test("lengthSq2", async () => {
  const src = `
    import lygia::math::lengthSq::lengthSq2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = lengthSq2(vec2f(3.0, 4.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // 3� + 4� = 25
  expectCloseTo([25.0], result);
});

test("lengthSq3", async () => {
  const src = `
    import lygia::math::lengthSq::lengthSq3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = lengthSq3(vec3f(1.0, 2.0, 2.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // 1� + 2� + 2� = 9
  expectCloseTo([9.0], result);
});

test("distEuclidean2", async () => {
  const src = `
    import lygia::math::dist::distEuclidean2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = distEuclidean2(vec2f(0.0, 0.0), vec2f(3.0, 4.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Distance is 5.0
  expectCloseTo([5.0], result);
});

test("distManhattan2", async () => {
  const src = `
    import lygia::math::dist::distManhattan2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = distManhattan2(vec2f(0.0, 0.0), vec2f(3.0, 4.0));
      test::results[0] = result;
    }
  `;
  const result = await testCompute(src);
  // Manhattan distance is 3 + 4 = 7
  expectCloseTo([7.0], result);
});
