import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("desaturate", async () => {
  const src = `
     import lygia::color::desaturate::desaturate;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.5, 0.0); // Orange color
       let result = desaturate(color, 0.5); // 50% desaturation
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Orange (1, 0.5, 0) with 50% desaturation should move toward gray (0.64, 0.64, 0.64)
  // Gray value is luminance: 1*0.3 + 0.5*0.59 + 0*0.11 = 0.3 + 0.295 = 0.595
  // 50% blend: (1+0.595)/2 = 0.7975, (0.5+0.595)/2 = 0.5475, (0+0.595)/2 = 0.2975
  expectCloseTo([0.7975, 0.5475, 0.2975], result);
});

test("desaturate4", async () => {
  const src = `
     import lygia::color::desaturate::desaturate4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.5, 0.0, 0.8); // Orange color with alpha
       let result = desaturate4(color, 0.5); // 50% desaturation
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // RGB should be same as desaturate test, alpha should remain 0.8
  expectCloseTo([0.7975, 0.5475, 0.2975, 0.8], result);
});

test("brightnessMatrix", async () => {
  const src = `
     import lygia::color::brightnessMatrix::brightnessMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let matrix = brightnessMatrix(0.2); // 20% brightness increase
       // Test that the matrix translates colors correctly
       // Matrix should have brightness offset in the last column
       test::results[0] = vec4f(matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // The translation part should be (0.2, 0.2, 0.2, 1.0)
  expectCloseTo([0.2, 0.2, 0.2, 1.0], result);
});

// Batch 3: color utility functions
test("contrast", async () => {
  const src = `
     import lygia::color::contrast::contrast;

     @compute @workgroup_size(1)
     fn foo() {
       let result = contrast(0.7, 1.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // (0.7 - 0.5) * 1.5 + 0.5 = 0.2 * 1.5 + 0.5 = 0.8
  expectCloseTo([0.8], result);
});

test("contrast3", async () => {
  const src = `
     import lygia::color::contrast::contrast3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.8, 0.6, 0.4);
       let result = contrast3(color, 2.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Each component: (v - 0.5) * 2.0 + 0.5
  // r: (0.8 - 0.5) * 2 + 0.5 = 1.1
  // g: (0.6 - 0.5) * 2 + 0.5 = 0.7
  // b: (0.4 - 0.5) * 2 + 0.5 = 0.3
  expectCloseTo([1.1, 0.7, 0.3], result);
});

test("contrastMatrix", async () => {
  const src = `
     import lygia::color::contrastMatrix::contrastMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let matrix = contrastMatrix(1.5);
       // Test diagonal and translation values
       test::results[0] = vec4f(matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][0]);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Diagonal should be 1.5, translation should be (1-1.5)*0.5 = -0.25
  expectCloseTo([1.5, 1.5, 1.5, -0.25], result);
});

test("levelsInputRange3", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.3, 0.5, 0.7);
       let result = levelsInputRange3(color, vec3f(0.2), vec3f(0.8));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // (v - iMin) / (iMax - iMin) clamped to [0, 1]
  // (0.3 - 0.2) / (0.8 - 0.2) = 0.1 / 0.6 = 0.1667
  // (0.5 - 0.2) / 0.6 = 0.5
  // (0.7 - 0.2) / 0.6 = 0.8333
  expectCloseTo([0.1667, 0.5, 0.8333], result);
});

test("levelsGamma3", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.25, 0.5, 0.75);
       let result = levelsGamma3(color, vec3f(2.0));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // pow(v, 1/gamma) = pow(v, 0.5) = sqrt(v)
  expectCloseTo([0.5, Math.SQRT1_2, 0.866], result);
});

test("levels3Float", async () => {
  const src = `
     import lygia::color::levels::levels3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.3, 0.5, 0.7);
       // Remap input [0.2, 0.8] to output [0.1, 0.9] with gamma 2.0
       let result = levels3Float(color, 0.2, 2.0, 0.8, 0.1, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Step 1: inputRange: (v - 0.2) / (0.8 - 0.2) = (v - 0.2) / 0.6
  //   r: (0.3 - 0.2) / 0.6 = 0.1667
  //   g: (0.5 - 0.2) / 0.6 = 0.5
  //   b: (0.7 - 0.2) / 0.6 = 0.8333
  // Step 2: gamma: pow(v, 1/2.0) = sqrt(v)
  //   r: sqrt(0.1667) = 0.4082
  //   g: sqrt(0.5) = INV_SQRT2 (≈ 0.7071)
  //   b: sqrt(0.8333) = 0.9129
  // Step 3: outputRange: mix(0.1, 0.9, v) = 0.1 + v * 0.8
  //   r: 0.1 + 0.4082 * 0.8 = 0.4266
  //   g: 0.1 + INV_SQRT2 * 0.8 = 0.6657
  //   b: 0.1 + 0.9129 * 0.8 = 0.8303
  expectCloseTo([0.4266, 0.6657, 0.8303], result);
});

test("brightnessContrast", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.7;
       let brightness = 0.1;
       let contrast = 1.5;
       let result = brightnessContrast(value, brightness, contrast);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // (0.7 - 0.5) * 1.5 + 0.5 + 0.1 = 0.2 * 1.5 + 0.6 = 0.9
  expectCloseTo([0.9], result);
});

test("contrast4", async () => {
  const src = `
     import lygia::color::contrast::contrast4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.8, 0.6, 0.4, 0.9);
       let result = contrast4(color, 1.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Each RGB component: (v - 0.5) * 1.5 + 0.5
  // r: (0.8 - 0.5) * 1.5 + 0.5 = 0.95
  // g: (0.6 - 0.5) * 1.5 + 0.5 = 0.65
  // b: (0.4 - 0.5) * 1.5 + 0.5 = 0.35
  // a: preserved at 0.9
  expectCloseTo([0.95, 0.65, 0.35, 0.9], result);
});

test("exposure", async () => {
  const src = `
     import lygia::color::exposure::exposure;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.25;
       let amount = 2.0; // +2 stops = 4x brighter
       let result = exposure(value, amount);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // 0.25 * 2^2 = 0.25 * 4 = 1.0
  expectCloseTo([1.0], result);
});

test("levels3", async () => {
  const src = `
     import lygia::color::levels::levels3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.4, 0.6, 0.8);
       // Input range [0.2, 0.9], gamma 2.0, output range [0.1, 0.8]
       let result = levels3(color, vec3f(0.2), vec3f(2.0), vec3f(0.9), vec3f(0.1), vec3f(0.8));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Step 1: inputRange: (v - 0.2) / (0.9 - 0.2)
  //   r: (0.4 - 0.2) / 0.7 = 0.2857
  //   g: (0.6 - 0.2) / 0.7 = 0.5714
  //   b: (0.8 - 0.2) / 0.7 = 0.8571
  // Step 2: gamma: pow(v, 0.5) = sqrt(v)
  //   r: sqrt(0.2857) = 0.5345
  //   g: sqrt(0.5714) = 0.7560
  //   b: sqrt(0.8571) = 0.9258
  // Step 3: outputRange: mix(0.1, 0.8, v) = 0.1 + v * 0.7
  //   r: 0.1 + 0.5345 * 0.7 = 0.4742
  //   g: 0.1 + 0.7560 * 0.7 = 0.6292
  //   b: 0.1 + 0.9258 * 0.7 = 0.7481
  expectCloseTo([0.4742, 0.6292, 0.7481], result);
});

test("levels4", async () => {
  const src = `
     import lygia::color::levels::levels4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.4, 0.6, 0.8, 0.75);
       // Same settings as levels3 test
       let result = levels4(color, vec3f(0.2), vec3f(2.0), vec3f(0.9), vec3f(0.1), vec3f(0.8));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // RGB should match levels3 test, alpha preserved
  expectCloseTo([0.4742, 0.6292, 0.7481, 0.75], result);
});

test("levels4Float", async () => {
  const src = `
     import lygia::color::levels::levels4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.5, 0.7, 0.3, 0.85);
       // Input [0.3, 0.8], gamma 1.5, output [0.2, 0.9]
       let result = levels4Float(color, 0.3, 1.5, 0.8, 0.2, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Step 1: inputRange: (v - 0.3) / (0.8 - 0.3) = (v - 0.3) / 0.5
  //   r: (0.5 - 0.3) / 0.5 = 0.4
  //   g: (0.7 - 0.3) / 0.5 = 0.8
  //   b: (0.3 - 0.3) / 0.5 = 0.0
  // Step 2: gamma: pow(v, 1/1.5) = pow(v, 0.6667)
  //   r: pow(0.4, 0.6667) = 0.5429
  //   g: pow(0.8, 0.6667) = 0.8618
  //   b: pow(0.0, 0.6667) = 0.0
  // Step 3: outputRange: mix(0.2, 0.9, v) = 0.2 + v * 0.7
  //   r: 0.2 + 0.5429 * 0.7 = 0.5800
  //   g: 0.2 + 0.8618 * 0.7 = 0.8032
  //   b: 0.2 + 0.0 * 0.7 = 0.2
  //   a: preserved at 0.85
  expectCloseTo([0.5800, 0.8032, 0.2, 0.85], result);
});

// Gamma function tests
test("levelsGamma3Float", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.16, 0.36, 0.64);
       let result = levelsGamma3Float(color, 2.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // pow(v, 1/2.0) = sqrt(v)
  // sqrt(0.16) = 0.4, sqrt(0.36) = 0.6, sqrt(0.64) = 0.8
  expectCloseTo([0.4, 0.6, 0.8], result);
});

test("levelsGamma4", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.25, 0.5, 0.75, 0.9);
       let result = levelsGamma4(color, vec3f(2.0, 1.5, 3.0));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // pow(v, 1/gamma)
  // r: pow(0.25, 0.5) = 0.5
  // g: pow(0.5, 1/1.5) = pow(0.5, 0.6667) = 0.6300
  // b: pow(0.75, 1/3.0) = 0.9086
  // a: preserved at 0.9
  expectCloseTo([0.5, 0.63, 0.9086, 0.9], result);
});

test("levelsGamma4Float", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.09, 0.25, 0.49, 0.85);
       let result = levelsGamma4Float(color, 2.0);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // pow(v, 1/2.0) = sqrt(v)
  // sqrt(0.09) = 0.3, sqrt(0.25) = 0.5, sqrt(0.49) = 0.7
  // a: preserved at 0.85
  expectCloseTo([0.3, 0.5, 0.7, 0.85], result);
});

// Input Range function tests
test("levelsInputRange3Float", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.2, 0.5, 0.8);
       let result = levelsInputRange3Float(color, 0.1, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // (v - iMin) / (iMax - iMin) clamped to [0, 1]
  // (0.2 - 0.1) / (0.9 - 0.1) = 0.1 / 0.8 = 0.125
  // (0.5 - 0.1) / 0.8 = 0.5
  // (0.8 - 0.1) / 0.8 = 0.875
  expectCloseTo([0.125, 0.5, 0.875], result);
});

test("levelsInputRange4", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.3, 0.6, 0.9, 0.75);
       let result = levelsInputRange4(color, vec3f(0.2, 0.4, 0.5), vec3f(0.8, 0.9, 1.0));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Per-channel input range mapping:
  // r: (0.3 - 0.2) / (0.8 - 0.2) = 0.1 / 0.6 = 0.1667
  // g: (0.6 - 0.4) / (0.9 - 0.4) = 0.2 / 0.5 = 0.4
  // b: (0.9 - 0.5) / (1.0 - 0.5) = 0.4 / 0.5 = 0.8
  // a: preserved at 0.75
  expectCloseTo([0.1667, 0.4, 0.8, 0.75], result);
});

test("levelsInputRange4Float", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.15, 0.45, 0.75, 0.95);
       let result = levelsInputRange4Float(color, 0.1, 0.8);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // (v - 0.1) / (0.8 - 0.1) = (v - 0.1) / 0.7
  // r: (0.15 - 0.1) / 0.7 = 0.0714
  // g: (0.45 - 0.1) / 0.7 = 0.5
  // b: (0.75 - 0.1) / 0.7 = 0.9286
  // a: preserved at 0.95
  expectCloseTo([0.0714, 0.5, 0.9286, 0.95], result);
});

// Output Range function tests
test("levelsOutputRange3Float", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.0, 0.5, 1.0);
       let result = levelsOutputRange3Float(color, 0.2, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // mix(0.2, 0.9, v) = 0.2 + v * (0.9 - 0.2) = 0.2 + v * 0.7
  // r: 0.2 + 0.0 * 0.7 = 0.2
  // g: 0.2 + 0.5 * 0.7 = 0.55
  // b: 0.2 + 1.0 * 0.7 = 0.9
  expectCloseTo([0.2, 0.55, 0.9], result);
});

test("levelsOutputRange4", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.25, 0.5, 0.75, 0.8);
       let result = levelsOutputRange4(color, vec3f(0.1, 0.2, 0.3), vec3f(0.8, 0.9, 1.0));
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Per-channel output range mapping: mix(oMin, oMax, v)
  // r: mix(0.1, 0.8, 0.25) = 0.1 + 0.25 * 0.7 = 0.275
  // g: mix(0.2, 0.9, 0.5) = 0.2 + 0.5 * 0.7 = 0.55
  // b: mix(0.3, 1.0, 0.75) = 0.3 + 0.75 * 0.7 = 0.825
  // a: preserved at 0.8
  expectCloseTo([0.275, 0.55, 0.825, 0.8], result);
});

test("levelsOutputRange4Float", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.2, 0.6, 0.8, 0.7);
       let result = levelsOutputRange4Float(color, 0.3, 0.95);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // mix(0.3, 0.95, v) = 0.3 + v * (0.95 - 0.3) = 0.3 + v * 0.65
  // r: 0.3 + 0.2 * 0.65 = 0.43
  // g: 0.3 + 0.6 * 0.65 = 0.69
  // b: 0.3 + 0.8 * 0.65 = 0.82
  // a: preserved at 0.7
  expectCloseTo([0.43, 0.69, 0.82, 0.7], result);
});

// Batch 4: composite functions
test("tonemapReinhard3", async () => {
  const src = `
     import lygia::color::tonemap::reinhard::tonemapReinhard3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       test::results[0] = tonemapReinhard3(hdr);
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // v / (1 + luminance(v))
  // luminance ≈ 2.0*0.2125 + 1.5*0.7154 + 1.0*0.0721 = 1.5706
  // result = hdr / (1 + 1.5706) = hdr / 2.5706
  expectCloseTo([0.7782, 0.5836, 0.3891], result.slice(0, 3));
});

test("tonemapUnreal3", async () => {
  const src = `
     import lygia::color::tonemap::unreal::tonemapUnreal3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(1.0, 0.5, 0.25);
       let result = tonemapUnreal3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // x / (x + 0.155) * 1.019
  // Each component separately: 1.0/(1.155)*1.019=0.8823, 0.5/(0.655)*1.019=0.7779, 0.25/(0.405)*1.019=0.6290
  expectCloseTo([0.8823, 0.7779, 0.6290], result.slice(0, 3));
});

test("tonemapLinear3 - identity baseline", async () => {
  const src = `
     import lygia::color::tonemap::linear::tonemapLinear3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: HDR values remain unchanged (identity function)
       let hdr = vec3f(1.5, 2.0, 0.5);
       let result1 = tonemapLinear3(hdr);

       // Test 2: Values >1.0 are NOT clamped (unlike other tonemappers)
       let bright = vec3f(5.0, 10.0, 100.0);
       let result2 = tonemapLinear3(bright);

       // Test 3: Negative values also pass through (no clamping)
       let negative = vec3f(-0.5, 0.0, 1.0);
       let result3 = tonemapLinear3(negative);

       test::results[0] = vec4f(result1.r, result2.r, result2.g, result3.r);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Verify passthrough behavior:
  // result1.r = 1.5 (unchanged)
  // result2.r = 5.0 (not clamped to 1.0)
  // result2.g = 10.0 (not clamped)
  // result3.r = -0.5 (negative preserved)
  expectCloseTo([1.5, 5.0, 10.0, -0.5], result);
});

// Batch 6: color space
test("hueShift", async () => {
  const src = `
     import lygia::color::hueShift::hueShift;
     import lygia::math::consts::TAU;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let shifted = hueShift(rgb, TAU * 0.3333); // Shift by 120° (TAU/3 radians)
       test::results[0] = shifted;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Red shifted by 120° should become green
  // HSV color space conversion introduces small floating-point errors (~0.0002)
  expectCloseTo([0.0, 1.0, 0.0], result, 0.001);
});

test("vibrance", async () => {
  const src = `
     import lygia::color::vibrance::vibrance3;

     @compute @workgroup_size(1)
     fn foo() {
       // Orange color with low saturation (muted)
       let rgb = vec3f(0.6, 0.5, 0.4);
       // Increase vibrance by 0.5 (should increase saturation of muted colors)
       let result = vibrance3(rgb, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Vibrance formula: mix(vec3(luma), color, 1.0 + (v * 1.0 - sign(v) * sat))
  // max_color = 0.6, min_color = 0.4, sat = 0.2
  // luma ≈ 0.6*0.2126 + 0.5*0.7152 + 0.4*0.0722 = 0.5141
  // mix factor = 1.0 + (0.5 * 1.0 - sign(0.5) * 0.2) = 1.0 + 0.5 - 0.2 = 1.3
  // mix(0.5141, color, 1.3) means interpolate/extrapolate
  // r: 0.5141 + (0.6 - 0.5141) * 1.3 = 0.5141 + 0.1117 = 0.6258
  // g: 0.5141 + (0.5 - 0.5141) * 1.3 = 0.5141 - 0.0183 = 0.4958
  // b: 0.5141 + (0.4 - 0.5141) * 1.3 = 0.5141 - 0.1483 = 0.3658
  expectCloseTo([0.6258, 0.4958, 0.3658], result);
});

test("vibrance - selective saturation boost", async () => {
  const src = `
     import lygia::color::vibrance::vibrance3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: Muted color (low saturation) - should change significantly
       let muted = vec3f(0.6, 0.5, 0.4);  // sat = 0.2
       let muted_boosted = vibrance3(muted, 0.5);

       // Test 2: Saturated color (high saturation) - should change less
       let saturated = vec3f(1.0, 0.1, 0.0);  // sat = 0.9
       let saturated_boosted = vibrance3(saturated, 0.5);

       // Test 3: Negative vibrance should desaturate
       let color = vec3f(0.8, 0.4, 0.2);
       let desaturated = vibrance3(color, -0.5);

       // Calculate saturation change for each
       let muted_sat_change = (muted_boosted.r - muted_boosted.b) / (muted.r - muted.b);
       let saturated_sat_change = (saturated_boosted.r - saturated_boosted.g) / (saturated.r - saturated.g);

       test::results[0] = vec4f(muted_sat_change, saturated_sat_change, desaturated.r, desaturated.g);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Vibrance should increase muted saturation more than saturated colors
  // muted_sat_change should be > saturated_sat_change
  expect(result[0]).toBeGreaterThan(result[1]);

  // Muted color should have increased saturation (change > 1.0)
  expect(result[0]).toBeGreaterThan(1.0);

  // Negative vibrance should move colors toward gray (desaturated)
  // For vec3f(0.8, 0.4, 0.2), luma ≈ 0.525
  // Vibrance -0.5 with sat ≈ 0.6 gives mix factor 1.0 + (-0.5 - (-1) * 0.6) = 1.0 - 0.5 + 0.6 = 1.1
  // But mix factor is clamped/saturated, so result moves toward luma
  expectCloseTo([0.833, 0.393], [result[2], result[3]]);
});

test("ditherBayer", async () => {
  const src = `
     import lygia::color::dither::bayer::{ditherBayer, ditherBayer3Precision};

     @compute @workgroup_size(1)
     fn foo() {
       // Test Bayer dithering with quantization to 16 levels
       // Use a value between quantization levels to see dithering effect
       let color = vec3f(0.53, 0.53, 0.53); // Between 8/16 (0.5) and 9/16 (0.5625)
       let xy = vec2f(2.0, 3.0);

       // Get the Bayer threshold value for this pixel position
       let bayerValue = ditherBayer(xy);

       // Get the dithered color (quantized to 16 levels)
       let dithered = ditherBayer3Precision(color, xy, 16);

       test::results[0] = vec4f(dithered, bayerValue);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // xy=(2,3): x % 8 = 2, y % 8 = 3, index = 2 + 3*8 = 26
  // Bayer matrix[26] = 52.0/64.0 = 0.8125
  const expectedBayerValue = 0.8125;

  // For color 0.53 quantized to 16 levels:
  // decimated = floor(0.53 * 16) / 16 = floor(8.48) / 16 = 8/16 = 0.5
  // diff = (0.53 - 0.5) * 16 = 0.48
  // step(0.8125, 0.48) = 0.0 (since 0.48 < 0.8125)
  // result = decimate3(0.53 + 0.0/16, vec3(16)) = decimate3(0.53, vec3(16))
  //        = floor(0.53 * 16) / 16 = floor(8.48) / 16 = 8/16 = 0.5
  expectCloseTo([0.5, 0.5, 0.5, expectedBayerValue], result);
});

test("ditherBlueNoise - spatial distribution", async () => {
  const src = `
     import lygia::color::dither::blueNoise::ditherBlueNoise;

     @compute @workgroup_size(1)
     fn foo() {
       // Test spatial distribution characteristics of blue noise

       // Test 1: Adjacent pixels should have different noise values
       let noise_0_0 = ditherBlueNoise(vec2f(0.0, 0.0));
       let noise_1_0 = ditherBlueNoise(vec2f(1.0, 0.0));
       let noise_0_1 = ditherBlueNoise(vec2f(0.0, 1.0));
       let noise_1_1 = ditherBlueNoise(vec2f(1.0, 1.0));

       // Calculate variance of this 2x2 block (should be high for good distribution)
       let mean = (noise_0_0 + noise_1_0 + noise_0_1 + noise_1_1) * 0.25;
       let variance = (
         pow(noise_0_0 - mean, 2.0) +
         pow(noise_1_0 - mean, 2.0) +
         pow(noise_0_1 - mean, 2.0) +
         pow(noise_1_1 - mean, 2.0)
       ) * 0.25;

       // Test 2: Deterministic - same coordinates produce same values
       let repeat1 = ditherBlueNoise(vec2f(5.0, 7.0));
       let repeat2 = ditherBlueNoise(vec2f(5.0, 7.0));

       test::results[0] = vec4f(variance, repeat1, repeat2, noise_0_0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Variance should be reasonably high (> 0.01) for well-distributed noise
  expect(result[0]).toBeGreaterThan(0.01);

  // Deterministic: same input should produce same output
  expectCloseTo([result[1]], [result[2]]);

  // All values should be in [0, 1] range
  expect(result[3]).toBeGreaterThanOrEqual(0.0);
  expect(result[3]).toBeLessThanOrEqual(1.0);
});

test("ditherBlueNoise3", async () => {
  const src = `
     import lygia::color::dither::blueNoise::ditherBlueNoise3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test blue noise dithering with a value between quantization levels
       // At 256 levels (default), 0.503 is between level 128 (0.5) and 129 (0.50390625)
       let color = vec3f(0.503, 0.503, 0.503);
       let xy = vec2f(2.0, 3.0);
       let dithered = ditherBlueNoise3(color, xy);

       // Store: (dithered.r, dithered.g, dithered.b, original_value)
       test::results[0] = vec4f(dithered, color.r);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Blue noise dithering should quantize to one of two adjacent levels
  // Either 128/256=0.5 or 129/256≈0.50391
  const quantLevel128 = 128.0 / 256.0;
  const quantLevel129 = 129.0 / 256.0;

  // Result should be quantized (not the input 0.503)
  expect(result[0]).not.toBeCloseTo(0.503, 3);

  // All three channels should be quantized to one of the two levels
  expect([quantLevel128, quantLevel129]).toContain(result[0]);
  expect([quantLevel128, quantLevel129]).toContain(result[1]);
  expect([quantLevel128, quantLevel129]).toContain(result[2]);

  // Verify original value was 0.503
  expectCloseTo([0.503], [result[3]]);
});

test("ditherBlueNoise3Precision", async () => {
  const src = `
     import lygia::color::dither::blueNoise::ditherBlueNoise3Precision;

     @compute @workgroup_size(1)
     fn foo() {
       // Test with custom precision (16 levels)
       // Color 0.53 at 16 levels should dither between 8/16 (0.5) and 9/16 (0.5625)
       let color = vec3f(0.53, 0.53, 0.53);
       let xy = vec2f(2.0, 3.0);
       let dithered = ditherBlueNoise3Precision(color, xy, 16);

       test::results[0] = vec4f(dithered, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Should quantize to 16 levels: either 8/16 or 9/16
  const quantLevel8 = 8.0 / 16.0; // 0.5
  const quantLevel9 = 9.0 / 16.0; // 0.5625

  // All channels should be one of these two values
  expect([quantLevel8, quantLevel9]).toContain(result[0]);
  expect([quantLevel8, quantLevel9]).toContain(result[1]);
  expect([quantLevel8, quantLevel9]).toContain(result[2]);

  // Verify result is saturated to [0, 1]
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.0);
});

test("ditherVlachos3", async () => {
  const src = `
     import lygia::color::dither::vlachos::ditherVlachos3;
     import lygia::math::decimate::decimate;

     @compute @workgroup_size(1)
     fn foo() {
       // Test Vlachos dithering quantization behavior
       // Use 0.5 which should quantize to exactly 128/256 after adding noise
       let color = vec3f(0.5, 0.5, 0.5);
       let xy = vec2f(2.0, 3.0);
       let dithered = ditherVlachos3(color, xy);

       // Compare with undithered quantization (should differ slightly due to noise)
       let undithered = decimate(0.5, 256.0);

       test::results[0] = vec4f(dithered.r, dithered.g, dithered.b, undithered);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Vlachos adds noise in range [-1/255, 1/255] then quantizes to 256 levels
  // For input 0.5, result should be close to 0.5 (within one quantization step)
  expectCloseTo(
    [0.5, 0.5, 0.5],
    [result[0], result[1], result[2]],
    1.0 / 256.0,
  );

  // All three channels should be quantized (multiples of 1/256)
  const tolerance = 0.0001;
  expect(
    Math.abs(result[0] * 256.0 - Math.round(result[0] * 256.0)),
  ).toBeLessThan(tolerance);
  expect(
    Math.abs(result[1] * 256.0 - Math.round(result[1] * 256.0)),
  ).toBeLessThan(tolerance);
  expect(
    Math.abs(result[2] * 256.0 - Math.round(result[2] * 256.0)),
  ).toBeLessThan(tolerance);

  // Undithered should be exactly 0.5 (no noise added)
  expectCloseTo([0.5], [result[3]]);
});

test("ditherVlachos4", async () => {
  const src = `
     import lygia::color::dither::vlachos::ditherVlachos4;

     @compute @workgroup_size(1)
     fn foo() {
       // Test vec4 variant (preserves alpha)
       let color = vec4f(0.7, 0.5, 0.3, 0.85);
       let xy = vec2f(4.0, 6.0);
       let dithered = ditherVlachos4(color, xy);

       test::results[0] = dithered;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // RGB channels should be dithered and quantized
  const tolerance = 0.0001;
  expect(
    Math.abs(result[0] * 256.0 - Math.round(result[0] * 256.0)),
  ).toBeLessThan(tolerance);
  expect(
    Math.abs(result[1] * 256.0 - Math.round(result[1] * 256.0)),
  ).toBeLessThan(tolerance);
  expect(
    Math.abs(result[2] * 256.0 - Math.round(result[2] * 256.0)),
  ).toBeLessThan(tolerance);

  // Alpha should be preserved exactly
  expectCloseTo([0.85], [result[3]]);

  // RGB should be in valid range and close to original
  expectCloseTo(
    [0.7, 0.5, 0.3],
    [result[0], result[1], result[2]],
    1.0 / 256.0,
  );
});
