import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("gamma2linear", async () => {
  const src = `
     import lygia::color::space::gamma2linear::gamma2linear3;

     @compute @workgroup_size(1)
     fn foo() {
       let gamma = vec3f(0.5, 0.5, 0.5);
       let result = gamma2linear3(gamma);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // pow(0.5, 2.2) ≈ 0.2181 (standard gamma 2.2)
  expectCloseTo([0.2176, 0.2176, 0.2176], result);
});

test("linear2gamma", async () => {
  const src = `
     import lygia::color::space::linear2gamma::linear2gamma3;

     @compute @workgroup_size(1)
     fn foo() {
       let linear = vec3f(0.25, 0.25, 0.25);
       let result = linear2gamma3(linear);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // pow(0.25, 1/2.2) ≈ 0.5277 (standard gamma 2.2)
  expectCloseTo([0.5325, 0.5325, 0.5325], result);
});

test("rgb2srgb", async () => {
  const src = `
     import lygia::color::space::rgb2srgb::rgb2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.5, 0.3, 0.1); // Linear RGB
       let result = rgb2srgb(rgb);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Linear RGB -> sRGB (gamma correction)
  expectCloseTo([0.7354, 0.5838, 0.3492], result);
});

test("srgb2rgb", async () => {
  const src = `
     import lygia::color::space::srgb2rgb::srgb2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(0.735, 0.584, 0.349);
       let result = srgb2rgb(srgb);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // sRGB -> Linear RGB
  expectCloseTo([0.4995, 0.3002, 0.0999], result);
});

test("rgb2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2srgb::rgb2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.5, 0.3, 0.1, 0.4); // Linear RGB with alpha
       let result = rgb2srgb4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo([0.7354, 0.5838, 0.3492, 0.4], result);
});

test("srgb2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::srgb2rgb::srgb2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(0.735, 0.584, 0.349, 0.3);
       let result = srgb2rgb4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  expectCloseTo(
    [
      0.4994581639766693, 0.30018994212150574, 0.09988708049058914,
      0.30000001192092896,
    ],
    result,
  );
});

test("gamma2linear - f32 overload", async () => {
  const src = `
     import lygia::color::space::gamma2linear::gamma2linear;

     @compute @workgroup_size(1)
     fn foo() {
       let gamma = 0.5;
       let result = gamma2linear(gamma);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // pow(0.5, 2.2) ≈ 0.2181 (standard gamma 2.2)
  expectCloseTo([0.2176], result);
});

test("gamma2linear4 - vec4 with alpha preservation", async () => {
  const src = `
     import lygia::color::space::gamma2linear::gamma2linear4;

     @compute @workgroup_size(1)
     fn foo() {
       let gamma = vec4f(0.5, 0.5, 0.5, 0.7);
       let result = gamma2linear4(gamma);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // pow(0.5, 2.2) ≈ 0.2181 for RGB, alpha unchanged
  expectCloseTo([0.2176, 0.2176, 0.2176, 0.7], result);
});

test("linear2gamma - f32 overload", async () => {
  const src = `
     import lygia::color::space::linear2gamma::linear2gamma;

     @compute @workgroup_size(1)
     fn foo() {
       let linear = 0.25;
       let result = linear2gamma(linear);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // pow(0.25, 1/2.2) ≈ 0.5277 (standard gamma 2.2)
  expectCloseTo([0.5325], result);
});

test("linear2gamma4 - vec4 with alpha preservation", async () => {
  const src = `
     import lygia::color::space::linear2gamma::linear2gamma4;

     @compute @workgroup_size(1)
     fn foo() {
       let linear = vec4f(0.25, 0.25, 0.25, 0.4);
       let result = linear2gamma4(linear);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // pow(0.25, 1/2.2) ≈ 0.5277 for RGB, alpha unchanged
  expectCloseTo([0.5325, 0.5325, 0.5325, 0.4], result);
});

test("rgb2srgb_mono - f32 function", async () => {
  const src = `
     import lygia::color::space::rgb2srgb::rgb2srgb_mono;

     @compute @workgroup_size(1)
     fn foo() {
       // Test both branches of the function
       let low = rgb2srgb_mono(0.002); // < 0.0031308 branch
       let high = rgb2srgb_mono(0.5);  // >= 0.0031308 branch
       test::results[0] = vec4f(low, high, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // low: 12.92 * 0.002 = 0.02584
  // high: 1.055 * pow(0.5, 0.41667) - 0.055 ≈ 0.735
  expectCloseTo([0.025840001180768013, 0.7353569269180298, 0.0, 0.0], result);
});

test("srgb2rgb_mono - f32 function", async () => {
  const src = `
     import lygia::color::space::srgb2rgb::srgb2rgb_mono;

     @compute @workgroup_size(1)
     fn foo() {
       // Test both branches
       let low = srgb2rgb_mono(0.03);  // < 0.04045 branch
       let high = srgb2rgb_mono(0.735); // >= 0.04045 branch
       test::results[0] = vec4f(low, high, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // low: 0.03 * 0.0773993808 ≈ 0.00232
  // high: pow((0.735 + 0.055) * 0.9478673, 2.4) ≈ 0.5
  expectCloseTo([0.0023219813592731953, 0.4994581639766693, 0.0, 0.0], result);
});
