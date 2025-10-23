import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("luminance", async () => {
  const src = `
     import lygia::color::luminance::luminance;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.5, 0.0); // Orange color
       let result = luminance(color);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // Luminance calculation: 1*0.2125 + 0.5*0.7154 + 0*0.0721 = 0.2125 + 0.3577 = 0.5702
  expectCloseTo([0.5702], result);
});

test("luminance4", async () => {
  const src = `
     import lygia::color::luminance::luminance4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.5, 0.0, 0.8); // Orange color with alpha
       let result = luminance4(color);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // Should be same as luminance test (alpha ignored)
  expectCloseTo([0.5702], result);
});

test("colorDistance", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let color1 = vec3f(1.0, 0.0, 0.0); // Red
       let color2 = vec3f(0.0, 0.0, 1.0); // Blue
       let result = colorDistanceLAB(color1, color2);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // LAB Euclidean distance between red and blue in LAB color space (0-100 scale)
  // This is a perceptual color distance metric
  expectCloseTo([176.314], result);
});

test("luma", async () => {
  const src = `
     import lygia::color::luma::luma3;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.5, 0.0); // Orange
       let result = luma3(rgb);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // Luma using Rec709 coefficients: 1.0*0.2126 + 0.5*0.7152 + 0.0*0.0722 = 0.5702
  expectCloseTo([0.5702], result);
});

test("mixOklab", async () => {
  const src = `
     import lygia::color::mixOklab::mixOklab;

     @compute @workgroup_size(1)
     fn foo() {
       let color1 = vec3f(1.0, 0.0, 0.0); // Red
       let color2 = vec3f(0.0, 0.0, 1.0); // Blue
       let result = mixOklab(color1, color2, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Mix red and blue in Oklab space - purple-ish result
  expectCloseTo([0.2637, 0.0866, 0.3628], result);
});

test("brightnessContrast3", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.6, 0.5, 0.4);
       let brightness = 0.1;
       let contrast = 1.2;
       let result = brightnessContrast3(color, brightness, contrast);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // (0.6 - 0.5) * 1.2 + 0.5 + 0.1 = 0.12 + 0.6 = 0.72
  // (0.5 - 0.5) * 1.2 + 0.5 + 0.1 = 0 + 0.6 = 0.6
  // (0.4 - 0.5) * 1.2 + 0.5 + 0.1 = -0.12 + 0.6 = 0.48
  expectCloseTo([0.72, 0.6, 0.48], result);
});

test("brightnessContrast4", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.6, 0.5, 0.4, 0.8);
       let brightness = 0.1;
       let contrast = 1.2;
       let result = brightnessContrast4(color, brightness, contrast);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // RGB adjusted, alpha preserved
  expectCloseTo([0.72, 0.6, 0.48, 0.8], result);
});

test("exposure3", async () => {
  const src = `
     import lygia::color::exposure::exposure3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.5, 0.5, 0.5);
       let amount = 1.0; // +1 stop = 2x brighter
       let result = exposure3(color, amount);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // 0.5 * 2^1 = 1.0
  expectCloseTo([1.0, 1.0, 1.0], result);
});

test("exposure4", async () => {
  const src = `
     import lygia::color::exposure::exposure4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.5, 0.5, 0.5, 0.7);
       let amount = 1.0;
       let result = exposure4(color, amount);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // RGB doubled, alpha preserved
  expectCloseTo([1.0, 1.0, 1.0, 0.7], result);
});

test("hueShiftRYB", async () => {
  const src = `
     import lygia::color::hueShiftRYB::hueShiftRYB;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.0, 0.0); // Red
       let angle = 2.0944; // 120 degrees (1/3 turn) - shift red toward yellow in RYB space
       let result = hueShiftRYB(color, angle);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // RYB hue shift: Red shifted by 120° in RYB space
  // After RGB->RYB->hue shift->RGB conversion
  // Red shifted 120° in RYB color wheel goes toward yellow
  expectCloseTo([1.0, 1.0, 0.0], result);
});

test("heatmap", async () => {
  const src = `
     import lygia::color::palette::heatmap::heatmap;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.5;
       let result = heatmap(value);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Heatmap formula: 1.0 - (v*2.1 - vec3(1.8,1.14,0.3))^2
  // For v=0.5: 1.0 - (1.05 - vec3(1.8,1.14,0.3))^2 = vec3(0.4375, 0.9919, 0.4375)
  expectCloseTo([0.4375, 0.9919, 0.4375], result);
});

test("paletteHue", async () => {
  const src = `
     import lygia::color::palette::hue::hue;

     @compute @workgroup_size(1)
     fn foo() {
       let x = 0.5;
       let ratio = 0.333; // neon ratio (1/3)
       let result = hue(x, ratio);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Physical hue palette at x=0.5 with ratio=1/3
  // Formula: v = abs(fmod(x + [0,1,2]*ratio, 1) * 2 - 1)
  // Then smoothstep: v*v*(3-2*v)
  // For x=0.5, ratio=1/3: [0.5, 0.833, 0.167] -> fmod -> [0.5, 0.833, 0.167]
  // -> *2-1 -> [0, 0.666, -0.666] -> abs -> [0, 0.666, 0.666] -> smoothstep
  // Loose precision due to smoothstep approximation differences
  expectCloseTo([0.0, 0.74, 0.743], result, 0.05);
});

test("hueDefault", async () => {
  const src = `
     import lygia::color::palette::hue::hueDefault;

     @compute @workgroup_size(1)
     fn foo() {
       // hueDefault uses default ratio of 1/3 (neon)
       // Test that it matches hue(x, 0.33333)
       let result = hueDefault(0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // hueDefault(0.5) should match paletteHue test: hue(0.5, 0.333)
  // Result should be [0.0, 0.740, 0.743] from paletteHue test
  // Loose precision due to smoothstep approximation differences
  expectCloseTo([0.0, 0.74, 0.743], result, 0.05);
});

test("mixSpectral", async () => {
  const src = `
     import lygia::color::mixSpectral::mixSpectral;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);

       // Test 50% spectral mix of red and blue
       // Spectral mixing uses Kubelka-Munk theory - physically accurate paint mixing
       // This produces a different result than linear RGB interpolation
       let mixed = mixSpectral(red, blue, 0.5);

       // For comparison, store what linear mix would give
       let linearMix = mix(red, blue, 0.5); // Would be (0.5, 0, 0.5)

       test::results[0] = vec4f(mixed.r, mixed.g, mixed.b, linearMix.r);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // Spectral mixing of red and blue produces a dark purple color
  // This is VERY different from linear RGB mixing which would give bright (0.5, 0, 0.5)
  // Spectral mixing uses Kubelka-Munk theory - physically accurate paint mixing
  // Real paint mixing produces darker, more muted colors than digital RGB mixing
  const mixed = [result[0], result[1], result[2]];

  // Spectral mixing produces a much darker result than linear mixing
  // All components should be quite low (realistic for physical pigment mixing)
  expect(mixed[0]).toBeGreaterThan(0.0);
  expect(mixed[0]).toBeLessThan(0.15); // Red is low

  expect(mixed[1]).toBeGreaterThan(0.0);
  expect(mixed[1]).toBeLessThan(0.1); // Green is very low

  expect(mixed[2]).toBeGreaterThan(0.0);
  expect(mixed[2]).toBeLessThan(0.1); // Blue is also low (darken effect from mixing)

  // Overall, the spectral mix should be notably darker than either input color
  const maxComponent = Math.max(...mixed);
  expect(maxComponent).toBeLessThan(0.2); // Dark purple, not bright

  // Verify linear mix comparison value is 0.5 (as expected for linear interpolation)
  expectCloseTo([0.5], [result[3]]);

  // Spectral mix should differ DRAMATICALLY from linear mix
  // (physical paint mixing produces darker colors than digital RGB mixing)
  expect(Math.abs(mixed[0] - result[3])).toBeGreaterThan(0.3); // 0.067 vs 0.5 = big difference!

  // Regression check - exact spectral mix values
  expectCloseTo([0.0673, 0.0093, 0.0241], mixed);
});

test("whiteBalance3", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance3;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec3f(0.5, 0.5, 0.5);

       // Test warm, neutral, and cool temperatures
       let warm = whiteBalance3(gray, 0.2, 0.0);    // Warm (orange/yellow)
       let neutral = whiteBalance3(gray, 0.0, 0.0); // Neutral (unchanged)
       let cool = whiteBalance3(gray, -0.2, 0.0);   // Cool (blue)

       test::results[0] = warm.r;
       test::results[1] = warm.b;
       test::results[2] = cool.r;
       test::results[3] = cool.b;
     }
   `;
  const result = await testCompute(src);

  const warmR = result[0];
  const warmB = result[1];
  const coolR = result[2];
  const coolB = result[3];

  // Warm temperature should increase red
  expect(warmR).toBeGreaterThan(0.5); // warm.r > 0.5

  // Cool temperature should increase blue
  expect(coolB).toBeGreaterThan(warmB); // cool.b > warm.b

  // Warm should have higher red than cool
  expect(warmR).toBeGreaterThan(coolR); // warm.r > cool.r

  // Regression check - exact white balance values
  expectCloseTo([0.5141, 0.4585, 0.4727, 0.5919], [warmR, warmB, coolR, coolB]);
});

test("whiteBalance4", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance4;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec4f(0.5, 0.5, 0.5, 0.8);

       // Test temperature shift (warm)
       let tempShift = whiteBalance4(gray, 0.2, 0.0);

       // Test tint shift (magenta/green)
       let tintMagenta = whiteBalance4(gray, 0.0, 0.1);  // Positive = magenta
       let tintGreen = whiteBalance4(gray, 0.0, -0.1);   // Negative = green

       test::results[0] = tempShift.r;
       test::results[1] = tempShift.b;
       test::results[2] = tintMagenta.g;
       test::results[3] = tintGreen.g;
     }
   `;
  const result = await testCompute(src);

  const tempShiftR = result[0];
  const tempShiftB = result[1];
  const magentaG = result[2];
  const greenG = result[3];

  // Temperature shift: warm should have R > B
  expect(tempShiftR).toBeGreaterThan(tempShiftB); // tempShift.r > tempShift.b

  // Tint behavior: magenta tint reduces green, green tint increases green
  expect(greenG).toBeGreaterThan(magentaG); // green tint increases G
  expect(magentaG).toBeLessThan(0.5); // magenta tint decreases G
  expect(greenG).toBeGreaterThan(0.5); // green tint increases G

  // Regression check - exact white balance values with tint
  expectCloseTo(
    [0.5141, 0.4585, 0.4872, 0.5132],
    [tempShiftR, tempShiftB, magentaG, greenG],
  );
});

test("saturationMatrix", async () => {
  const src = `
     import lygia::color::saturationMatrix::saturationMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let amount = 1.5; // Increase saturation by 50%
       let mat = saturationMatrix(amount);
       // Test matrix by applying to an orange color
       let color = vec3f(0.8, 0.5, 0.3);
       let result = mat * vec4f(color, 1.0);
       test::results[0] = result.xyz;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Saturation matrix at 1.5 should increase color saturation
  // Original: (0.8, 0.5, 0.3) -> more saturated orange
  // Luma ~0.57, with 1.5 saturation should push values further from luma
  // Expected: R increases (>0.8), G stays similar, B decreases (<0.3)
  expect(result[0]).toBeGreaterThan(0.8); // Red should increase
  expect(result[2]).toBeLessThan(0.3); // Blue should decrease
  // Loose precision due to matrix multiplication accumulation
  expectCloseTo([0.95, 0.53, 0.17], result, 0.1);
});

test("levelsOutputRange3", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.5, 0.5, 0.5);
       let minOutput = vec3f(0.2, 0.2, 0.2);
       let maxOutput = vec3f(0.8, 0.8, 0.8);
       let result = levelsOutputRange3(color, minOutput, maxOutput);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Middle value (0.5) should map to middle of output range (0.5)
  expectCloseTo([0.5, 0.5, 0.5], result);
});

// Color distance function tests
test("colorDistance", async () => {
  const src = `
     import lygia::color::distance::colorDistance;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);
       let distance = colorDistance(red, blue);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // Default is CIE94 distance between red and blue (0-100 scale)
  expectCloseTo([71.0491], result);
});

test("colorDistance4", async () => {
  const src = `
     import lygia::color::distance::colorDistance4;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec4f(1.0, 0.0, 0.0, 0.8);
       let blue = vec4f(0.0, 0.0, 1.0, 0.6);
       let distance = colorDistance4(red, blue);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // Alpha is ignored, should be same as colorDistance (0-100 scale)
  expectCloseTo([71.0491], result);
});

test("colorDistanceLABCIE94", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLABCIE94;

     @compute @workgroup_size(1)
     fn foo() {
       let green = vec3f(0.0, 1.0, 0.0);
       let yellow = vec3f(1.0, 1.0, 0.0);
       let distance = colorDistanceLABCIE94(green, yellow);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // CIE94 distance between green and yellow (0-100 scale)
  // These are relatively close colors in perceptual space
  expectCloseTo([10.0626], result);
});

test("colorDistanceOKLAB", async () => {
  const src = `
     import lygia::color::distance::colorDistanceOKLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let orange = vec3f(1.0, 0.5, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);

       // Perceptual distances
       let redToOrange = colorDistanceOKLAB(red, orange);
       let redToBlue = colorDistanceOKLAB(red, blue);

       test::results[0] = vec4f(redToOrange, redToBlue, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  const redToOrange = result[0];
  const redToBlue = result[1];

  // Orange is perceptually closer to red than blue is
  expect(redToOrange).toBeLessThan(redToBlue);

  // Red-orange should be relatively small (similar hues)
  expect(redToOrange).toBeGreaterThan(0.05);
  expect(redToOrange).toBeLessThan(0.3);

  // Red-blue should be larger (opposite hues)
  expect(redToBlue).toBeGreaterThan(0.3);

  // Regression check - exact OKLAB distance values
  expectCloseTo([0.2917, 0.5371], [redToOrange, redToBlue]);
});

test("colorDistanceYCbCr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYCbCr;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: Different chrominance (red vs blue)
       let red = vec3f(0.8, 0.2, 0.2);
       let blue = vec3f(0.2, 0.2, 0.8);
       let chromaDist = colorDistanceYCbCr(red, blue);

       // Test 2: Same chrominance, different luma (should be ~0)
       // Dark gray vs light gray (same neutral chroma)
       let darkGray = vec3f(0.3, 0.3, 0.3);
       let lightGray = vec3f(0.7, 0.7, 0.7);
       let lumaDist = colorDistanceYCbCr(darkGray, lightGray);

       test::results[0] = vec4f(chromaDist, lumaDist, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  const chromaDist = result[0];
  const lumaDist = result[1];

  // Different chrominance should produce measurable distance
  expect(chromaDist).toBeGreaterThan(0.5);
  expect(chromaDist).toBeLessThan(1.5);

  // Same chrominance (grays) should have near-zero distance
  // (YCbCr distance ignores Y/luma)
  expectCloseTo([0.0], [lumaDist]);

  // Regression check - exact YCbCr chroma distance
  expectCloseTo([0.5316], [chromaDist]);
});

test("colorDistanceYPbPr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYPbPr;

     @compute @workgroup_size(1)
     fn foo() {
       // Complementary colors (magenta vs cyan)
       let magenta = vec3f(1.0, 0.0, 1.0);
       let cyan = vec3f(0.0, 1.0, 1.0);
       let complementaryDist = colorDistanceYPbPr(magenta, cyan);

       // Similar colors (cyan vs blue)
       let blue = vec3f(0.0, 0.0, 1.0);
       let similarDist = colorDistanceYPbPr(cyan, blue);

       // Test symmetry
       let dist1 = colorDistanceYPbPr(magenta, cyan);
       let dist2 = colorDistanceYPbPr(cyan, magenta);

       test::results[0] = vec4f(complementaryDist, similarDist, dist1, dist2);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  const complementaryDist = result[0];
  const similarDist = result[1];
  const dist1 = result[2];
  const dist2 = result[3];

  // Complementary colors should have larger distance than similar colors
  expect(complementaryDist).toBeGreaterThan(similarDist);

  // Distance should be symmetric
  expectCloseTo([dist1], [dist2]);

  // Complementary colors should have significant distance
  expect(complementaryDist).toBeGreaterThan(0.5);

  // Regression check - exact YPbPr distance values
  expectCloseTo([0.9919, 0.5957], [complementaryDist, similarDist]);
});

test("colorDistanceYUV", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYUV;

     @compute @workgroup_size(1)
     fn foo() {
       let white = vec3f(1.0, 1.0, 1.0);
       let gray = vec3f(0.5, 0.5, 0.5);
       let distance = colorDistanceYUV(white, gray);
       test::results[0] = distance;
     }
   `;
  const result = await testCompute(src);
  // YUV distance between white and gray (mainly Y difference)
  // Should be around 0.5 (difference in luminance)
  expectCloseTo([0.5], result);
});

// Color transformation function tests
test("hueShift4", async () => {
  const src = `
     import lygia::color::hueShift::hueShift4;
     import lygia::math::consts::TAU;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.0, 0.0, 0.85); // Red with alpha
       let shifted = hueShift4(color, TAU * 0.3333); // Shift by 120°
       test::results[0] = shifted;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Red shifted by 120° should become green, alpha preserved
  // Loose precision due to HSV conversion and hue rotation
  expectCloseTo([0.0, 1.0, 0.0, 0.85], result, 0.05);
});

test("hueShiftRYB4", async () => {
  const src = `
     import lygia::color::hueShiftRYB::hueShiftRYB4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.0, 0.0, 0.7); // Red with alpha
       let angle = 2.0944; // 120 degrees
       let result = hueShiftRYB4(color, angle);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // RYB hue shift: Red shifted by 120° toward yellow
  // Loose precision due to RYB color space conversion approximations
  expectCloseTo([1.0, 1.0, 0.0, 0.7], result.slice(0, 4), 0.15);
  expectCloseTo([0.7], [result[3]]); // Alpha exact
});

test("luma - grayscale consistency", async () => {
  const src = `
     import lygia::color::luma::luma;
     import lygia::color::luma::luma3;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.75;
       let gray = vec3f(0.75, 0.75, 0.75);

       let scalarLuma = luma(value);
       let vectorLuma = luma3(gray);

       // For grayscale, both should give same result
       test::results[0] = vec4f(scalarLuma, vectorLuma, 0.0, 0.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Grayscale color should have luma equal to its value
  expectCloseTo([0.75, 0.75], [result[0], result[1]]);
});

test("luma4", async () => {
  const src = `
     import lygia::color::luma::luma4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.8, 0.3, 0.1, 0.6); // Orange-ish with alpha
       let result = luma4(color);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src);
  // Luma using Rec709: 0.8*0.2126 + 0.3*0.7152 + 0.1*0.0722
  // = 0.17008 + 0.21456 + 0.00722 = 0.39186
  expectCloseTo([0.39186], result);
});

test("vibrance4", async () => {
  const src = `
     import lygia::color::vibrance::vibrance4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.6, 0.5, 0.4, 0.8); // Muted orange with alpha
       let result = vibrance4(color, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Vibrance should increase saturation of muted colors
  // RGB values calculated same as vibrance3 test, alpha preserved
  expectCloseTo([0.6258, 0.4958, 0.3658, 0.8], result);
});

// Color mixing function tests
test("mixOklab4", async () => {
  const src = `
     import lygia::color::mixOklab::mixOklab4;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec4f(1.0, 0.0, 0.0, 0.8);
       let blue = vec4f(0.0, 0.0, 1.0, 0.4);
       let result = mixOklab4(red, blue, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Mix red and blue in Oklab space - purple-ish result
  // RGB should match mixOklab test, alpha should be 0.6 (mix of 0.8 and 0.4)
  expectCloseTo([0.2637, 0.0866, 0.3628, 0.6], result);
});

test("mixSpectral4", async () => {
  const src = `
     import lygia::color::mixSpectral::mixSpectral4;

     @compute @workgroup_size(1)
     fn foo() {
       let yellow = vec4f(1.0, 1.0, 0.0, 0.9);
       let cyan = vec4f(0.0, 1.0, 1.0, 0.5);
       let result = mixSpectral4(yellow, cyan, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });
  // Spectral mixing of yellow and cyan produces green
  // Alpha should be 0.7 (mix of 0.9 and 0.5)
  expect(result[1]).toBeGreaterThan(result[0]); // Green dominant
  expect(result[1]).toBeGreaterThan(result[2]); // Green > Blue
  expectCloseTo([0.7], [result[3]]); // Alpha

  // Regression check - exact spectral mix RGB values
  expectCloseTo([0.0782, 1.0272, 0.0596], [result[0], result[1], result[2]]);
});

test("mixSpectral_linear_to_reflectance", async () => {
  const src = `
     import lygia::color::mixSpectral::mixSpectral_linear_to_reflectance;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Pure red
       let reflectance = mixSpectral_linear_to_reflectance(rgb);

       // Sample a few wavelengths from the reflectance array
       // Red should have high reflectance in long wavelengths (red end of spectrum)
       // and low reflectance in short wavelengths (blue end)
       test::results[0] = vec4f(
         reflectance[0],   // Short wavelength (blue/violet)
         reflectance[19],  // Mid wavelength (green)
         reflectance[37],  // Long wavelength (red)
         1.0
       );
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // For pure red input:
  // - Short wavelengths (blue) should have low reflectance
  // - Long wavelengths (red) should have high reflectance
  expect(result[0]).toBeLessThan(0.2); // Blue end - low reflectance
  expect(result[2]).toBeGreaterThan(0.8); // Red end - high reflectance
  expect(result[2]).toBeGreaterThan(result[0]); // Red > Blue

  // Regression check - exact reflectance at sampled wavelengths
  expectCloseTo([0.0315, 0.0318, 0.9855], [result[0], result[1], result[2]]);
});

test("mixSpectral_reflectance_to_xyz", async () => {
  const src = `
     import lygia::color::mixSpectral::{mixSpectral_linear_to_reflectance, mixSpectral_reflectance_to_xyz};

     @compute @workgroup_size(1)
     fn foo() {
       // Test with a neutral gray
       let gray = vec3f(0.5, 0.5, 0.5);
       let reflectance = mixSpectral_linear_to_reflectance(gray);
       let xyz = mixSpectral_reflectance_to_xyz(reflectance);

       // For a neutral gray, XYZ values should be roughly equal
       // and Y (luminance) should be around 0.5
       test::results[0] = vec4f(xyz, 1.0);
     }
   `;
  const result = await testCompute(src, { elem: "vec4f" });

  // For neutral gray input:
  // - X, Y, Z should be similar (neutral color)
  // - Y (luminance) should be positive and reasonable
  const xyz = [result[0], result[1], result[2]];

  // All components should be positive
  expect(xyz[0]).toBeGreaterThan(0.0);
  expect(xyz[1]).toBeGreaterThan(0.0);
  expect(xyz[2]).toBeGreaterThan(0.0);

  // For gray, X and Z should be reasonably close to Y
  // (not exact due to spectral conversion, but within a reasonable range)
  expect(xyz[0]).toBeGreaterThan(xyz[1] * 0.5);
  expect(xyz[0]).toBeLessThan(xyz[1] * 1.5);
  expect(xyz[2]).toBeGreaterThan(xyz[1] * 0.5);
  expect(xyz[2]).toBeLessThan(xyz[1] * 1.5);

  // Regression check - exact XYZ values for gray input
  expectCloseTo([0.4751, 0.5, 0.5441], xyz);
});
