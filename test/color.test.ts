import { beforeAll, expect, test } from "vitest";
import { type WgslElementType, testComputeShader } from "wesl-tooling";

let gpu: GPU;

beforeAll(async () => {
	const webgpu = await import("webgpu");
	Object.assign(globalThis, webgpu.globals);

	gpu = webgpu.create([]);
});

test("rgb2heat", async () => {
	const src = `
    import lygia::color::space::rgb2heat::rgb2heat;

    @compute @workgroup_size(1)
    fn foo() { 
      let x = rgb2heat(vec3f(.8, .7, .5)); 

      test::results[0] = x;
    }
  `;

	const result = await testShader(src);
	expect(result[0]).approximately(0.854, 0.001);
});

test("rgb2xyz", async () => {
	const src = `
    import lygia::color::space::rgb2xyz::rgb2xyz;

    @compute @workgroup_size(1)
    fn foo() { 
      test::results[0] = rgb2xyz(vec3f(.8, .7, .5)); 
    }
  `;

	const result = await testShader(src, "vec3f");
	expect(result[0]).approximately(0.6705, 0.0001);

	const cie = { CIE_D50: true };
	const resultCie = await testShader(src, "vec3f", cie);
	expect(resultCie[0]).approximately(0.6899, 0.0001);
});

test("rgb2YPbPr", async () => {
	const src = `
    import lygia::color::space::rgb2YPbPr::rgb2YPbPr;

    @compute @workgroup_size(1)
    fn foo() { 
      test::results[0] = rgb2YPbPr(vec3f(.6, .7, .5)); 
    }
  `;

	const result = await testShader(src, "vec3f");
	expect(result[0]).approximately(0.6643, 0.0001);

	const sdtv = { YPBPR_SDTV: true };
	const resultSdtv = await testShader(src, "vec3f", sdtv);
	expect(resultSdtv[0]).approximately(0.6473, 0.0001);
});

test("rgb2yuv", async () => {
	const src = `
    import lygia::color::space::rgb2yuv::rgb2yuv;

    @compute @workgroup_size(1)
    fn foo() { 
      test::results[0] = rgb2yuv(vec3f(.6, .7, .5)); 
    }
  `;

	const result = await testShader(src, "vec3f");
	const expected = [0.6643, -0.0822, -0.0502];
	expectCloseTo(expected, result);

	const sdtv = { YUV_SDTV: true };
	const resultSdtv = await testShader(src, "vec3f", sdtv);
	const expectedSdtv = [0.6473, -0.0725, -0.0415];
	expectCloseTo(expectedSdtv, resultSdtv);
});

test("yuv2rgb", async () => {
	const src = `
    import lygia::color::space::yuv2rgb::yuv2rgb;

    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = yuv2rgb(vec3f(.6, .7, .5));
    }
  `;
	const result = await testShader(src, "vec3f");
	const expected = [1.2402, 0.2593, 2.0896];
	expectCloseTo(expected, result);

	const sdtv = { YUV_SDTV: true };
	const resultSdtv = await testShader(src, "vec3f", sdtv);
	const expectedSdtv = [1.1699, 0.0334, 2.0225];
	expectCloseTo(expectedSdtv, resultSdtv);
});

/** compare two arrays for approximate equality */
function expectCloseTo(a: number[], b: number[], epsilon = 0.0001): void {
	const match = a.every((val, index) => Math.abs(val - b[index]) < epsilon);
	if (match) return;
	expect.fail(`arrays don't match:\n  ${a}\n  ${b}`);
}

/** utility function to test WGSL computer shader */
function testShader(
	src: string,
	elem: WgslElementType = "f32",
	conditions?: Record<string, boolean>,
) {
	return testComputeShader(import.meta.url, gpu, src, elem, conditions);
}
