import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

// Animation easing functions
test("backIn", async () => {
  const src = `
    import lygia::animation::easing::backIn::backIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = backIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([-0.375], result);
});

test("backOut", async () => {
  const src = `
    import lygia::animation::easing::backOut::backOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = backOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([1.375], result);
});

test("backInOut", async () => {
  const src = `
    import lygia::animation::easing::backInOut::backInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = backInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("bounceIn", async () => {
  const src = `
    import lygia::animation::easing::bounceIn::bounceIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = bounceIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.28125], result);
});

test("bounceOut", async () => {
  const src = `
    import lygia::animation::easing::bounceOut::bounceOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = bounceOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.71875], result);
});

test("bounceInOut", async () => {
  const src = `
    import lygia::animation::easing::bounceInOut::bounceInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = bounceInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("circularIn", async () => {
  const src = `
    import lygia::animation::easing::circularIn::circularIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = circularIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.134], result);
});

test("circularOut", async () => {
  const src = `
    import lygia::animation::easing::circularOut::circularOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = circularOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.866], result);
});

test("circularInOut", async () => {
  const src = `
    import lygia::animation::easing::circularInOut::circularInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = circularInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("cubicIn", async () => {
  const src = `
    import lygia::animation::easing::cubicIn::cubicIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = cubicIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.125], result);
});

test("cubicOut", async () => {
  const src = `
    import lygia::animation::easing::cubicOut::cubicOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = cubicOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.875], result);
});

test("cubicInOut", async () => {
  const src = `
    import lygia::animation::easing::cubicInOut::cubicInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = cubicInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("elasticIn", async () => {
  const src = `
    import lygia::animation::easing::elasticIn::elasticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = elasticIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([-0.022097], result);
});

test("elasticOut", async () => {
  const src = `
    import lygia::animation::easing::elasticOut::elasticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = elasticOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([1.022097], result);
});

test("elasticInOut", async () => {
  const src = `
    import lygia::animation::easing::elasticInOut::elasticInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = elasticInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("exponentialIn", async () => {
  const src = `
    import lygia::animation::easing::exponentialIn::exponentialIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = exponentialIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.03125], result);
});

test("exponentialOut", async () => {
  const src = `
    import lygia::animation::easing::exponentialOut::exponentialOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = exponentialOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.96875], result);
});

test("exponentialInOut", async () => {
  const src = `
    import lygia::animation::easing::exponentialInOut::exponentialInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = exponentialInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("linearIn", async () => {
  const src = `
    import lygia::animation::easing::linearIn::linearIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = linearIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("linearOut", async () => {
  const src = `
    import lygia::animation::easing::linearOut::linearOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = linearOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("linearInOut", async () => {
  const src = `
    import lygia::animation::easing::linearInOut::linearInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = linearInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("quadraticIn", async () => {
  const src = `
    import lygia::animation::easing::quadraticIn::quadraticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quadraticIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.25], result);
});

test("quadraticOut", async () => {
  const src = `
    import lygia::animation::easing::quadraticOut::quadraticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quadraticOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.75], result);
});

test("quadraticInOut", async () => {
  const src = `
    import lygia::animation::easing::quadraticInOut::quadraticInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quadraticInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("quarticIn", async () => {
  const src = `
    import lygia::animation::easing::quarticIn::quarticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quarticIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.0625], result);
});

test("quarticOut", async () => {
  const src = `
    import lygia::animation::easing::quarticOut::quarticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quarticOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.9375], result);
});

test("quarticInOut", async () => {
  const src = `
    import lygia::animation::easing::quarticInOut::quarticInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quarticInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});

test("quinticIn", async () => {
  const src = `
    import lygia::animation::easing::quinticIn::quinticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quinticIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.03125], result);
});

test("quinticOut", async () => {
  const src = `
    import lygia::animation::easing::quinticOut::quinticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quinticOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([1.03125], result);
});

test("quinticInOut", async () => {
  const src = `
    import lygia::animation::easing::quinticInOut::quinticInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quinticInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([1.5], result);
});

test("sineIn", async () => {
  const src = `
    import lygia::animation::easing::sineIn::sineIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = sineIn(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.2929], result);
});

test("sineOut", async () => {
  const src = `
    import lygia::animation::easing::sineOut::sineOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = sineOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([Math.SQRT1_2], result);
});

test("sineInOut", async () => {
  const src = `
    import lygia::animation::easing::sineInOut::sineInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = sineInOut(0.5); }
  `;
  const result = await testCompute(src);
  expectCloseTo([0.5], result);
});
