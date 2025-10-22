import { test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

test("sphereSDF with vec3f", async () => {
  const src = `
    import lygia::sdf::sphereSDF::sphereSDF;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(3.0, 0.0, 4.0);
      let distance = sphereSDF(p);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // length(3, 0, 4) = 5.0
  expectCloseTo([5.0, 0.0, 0.0], result);
});

test("sphereSDF1 with radius", async () => {
  const src = `
    import lygia::sdf::sphereSDF::sphereSDF1;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(3.0, 0.0, 4.0);
      let distance = sphereSDF1(p, 2.0);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // length(3, 0, 4) - 2.0 = 5.0 - 2.0 = 3.0
  expectCloseTo([3.0, 0.0, 0.0], result);
});

test("boxSDF with vec3f", async () => {
  const src = `
    import lygia::sdf::boxSDF::boxSDF;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(2.0, 0.0, 0.0);
      let distance = boxSDF(p);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (2,0,0): default box has size 1, point outside on X axis
  // abs(p) = (2,0,0), d = abs(p) = (2,0,0)
  // min(max(2,0,0), 0) + length(max((2,0,0), 0)) = 0 + length(2,0,0) = 2.0
  expectCloseTo([2.0, 0.0, 0.0], result);
});

test("boxSDF1 with bounds", async () => {
  const src = `
    import lygia::sdf::boxSDF::boxSDF1;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(0.0, 0.0, 0.0);
      let b = vec3f(1.0, 1.0, 1.0);
      let distance = boxSDF1(p, b);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at origin with bounds (1,1,1)
  // d = abs(p) - b = abs(0,0,0) - (1,1,1) = (-1,-1,-1)
  // min(max(-1,-1,-1), 0) + length(max((-1,-1,-1), 0)) = -1 + 0 = -1.0
  expectCloseTo([-1.0, 0.0, 0.0], result);
});

test("cylinderSDF vertical", async () => {
  const src = `
    import lygia::sdf::cylinderSDF::cylinderSDF;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(0.0, 0.0, 0.0);
      let h = vec2f(1.0, 1.0);
      let distance = cylinderSDF(p, h);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at origin with h = (radius=1.0, height=1.0)
  // d = abs(vec2(length(p.xz), p.y)) - h = abs(vec2(0, 0)) - vec2(1,1) = vec2(-1,-1)
  // min(max(-1,-1), 0) + length(max((-1,-1), 0)) = -1 + 0 = -1.0
  expectCloseTo([-1.0, 0.0, 0.0], result);
});

test("cylinderSDF1 with single param", async () => {
  const src = `
    import lygia::sdf::cylinderSDF::cylinderSDF1;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(2.0, 0.0, 0.0);
      let distance = cylinderSDF1(p, 1.0);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (2,0,0), calls cylinderSDF(p, vec2(1.0))
  // d = abs(vec2(length(p.xz), p.y)) - vec2(1.0) = abs(vec2(2, 0)) - vec2(1,1) = vec2(1,-1)
  // min(max(1,-1), 0) + length(max((1,-1), 0)) = 0 + length(1,0) = 1.0
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("cylinderSDF2 with height and radius", async () => {
  const src = `
    import lygia::sdf::cylinderSDF::cylinderSDF2;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(0.0, 0.0, 0.0);
      let distance = cylinderSDF2(p, 1.0, 1.0);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at origin with h=1.0 (radius), r=1.0 (height)
  // d = abs(vec2(length(p.xz), p.y)) - vec2(h,r) = abs(vec2(0,0)) - vec2(1,1) = vec2(-1,-1)
  // min(max(-1,-1), 0) + length(max((-1,-1), 0)) = -1 + 0 = -1.0
  expectCloseTo([-1.0, 0.0, 0.0], result);
});

test("cylinderSDF4 arbitrary orientation", async () => {
  const src = `
    import lygia::sdf::cylinderSDF::cylinderSDF4;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(0.5, 0.5, 0.0);
      let a = vec3f(0.0, 0.0, 0.0);
      let b = vec3f(0.0, 1.0, 0.0);
      let distance = cylinderSDF4(p, a, b, 0.5);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Cylinder from (0,0,0) to (0,1,0) with radius 0.5
  // Point at (0.5,0.5,0) is on the surface at radius 0.5 from axis
  // pa = p - a = (0.5,0.5,0), ba = b - a = (0,1,0)
  // baba = 1, paba = 0.5
  // x = length((0.5,0.5,0)*1 - (0,1,0)*0.5) - 0.5*1 = length(0.5,0,0) - 0.5 = 0.5 - 0.5 = 0
  // y = abs(0.5 - 0.5) - 0.5 = -0.5
  // max(x,y) = max(0,-0.5) = 0, so on surface: distance = 0
  expectCloseTo([0.0, 0.0, 0.0], result, 0.01);
});

test("torusSDF", async () => {
  const src = `
    import lygia::sdf::torusSDF::torusSDF;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(1.0, 0.0, 0.0);
      let t = vec2f(1.0, 0.25);
      let distance = torusSDF(p, t);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (1,0,0): length(xz) = 1, so vec2(1-1, 0) = vec2(0, 0), length = 0, minus 0.25 = -0.25
  expectCloseTo([-0.25, 0.0, 0.0], result, 0.01);
});

test("torusSDF4 with sin/cos", async () => {
  const src = `
    import lygia::math::consts::INV_SQRT2;
    import lygia::sdf::torusSDF::torusSDF4;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec3f(1.0, 0.0, 0.0);
      let sc = vec2f(INV_SQRT2, INV_SQRT2); // sin/cos of 45 degrees
      let distance = torusSDF4(p, sc, 1.0, 0.25);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (1,0,0) with sc=(sin45,cos45)=(INV_SQRT2,INV_SQRT2), ra=1.0, rb=0.25
  // This is a partial torus (45 degree sector)
  // pos.x=abs(1)=1, k=dot((1,0), (INV_SQRT2,INV_SQRT2))=INV_SQRT2
  // sqrt(dot((1,0,0),(1,0,0)) + 1 - 2*1*INV_SQRT2) - 0.25 = sqrt(1 + 1 - Math.SQRT2) - 0.25
  // = sqrt(0.5858) - 0.25 ≈ 0.765 - 0.25 = 0.515
  expectCloseTo([0.515, 0.0, 0.0], result, 0.01);
});

test("rectSDF with vec2f size", async () => {
  const src = `
    import lygia::sdf::rectSDF::rectSDF;

    @compute @workgroup_size(1)
    fn foo() {
      let st = vec2f(0.5, 0.5);
      let s = vec2f(1.0, 1.0);
      let distance = rectSDF(st, s);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (0.5, 0.5) is center with size (1.0, 1.0)
  // uv = st * 2.0 - 1.0 = (0.5,0.5) * 2 - 1 = (0,0)
  // max(abs(0/1), abs(0/1)) = max(0, 0) = 0.0
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("rectSDF1 with scalar size", async () => {
  const src = `
    import lygia::sdf::rectSDF::rectSDF1;

    @compute @workgroup_size(1)
    fn foo() {
      let st = vec2f(0.5, 0.5);
      let distance = rectSDF1(st, 1.0);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (0.5, 0.5) with scalar size 1.0
  // Calls rectSDF(st, vec2(1.0)) which gives 0.0 (center point)
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("rectSDFDefault", async () => {
  const src = `
    import lygia::sdf::rectSDF::rectSDFDefault;

    @compute @workgroup_size(1)
    fn foo() {
      let st = vec2f(0.5, 0.5);
      let distance = rectSDFDefault(st);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (0.5, 0.5) with default size (1.0, 1.0)
  // Calls rectSDF(st, vec2(1.0)) which gives 0.0 (center point)
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("rectSDF3 with rounded corners", async () => {
  const src = `
    import lygia::sdf::rectSDF::rectSDF3;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec2f(0.5, 0.5);
      let b = vec2f(0.8, 0.8);
      let distance = rectSDF3(p, b, 0.1);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (0.5, 0.5) with bounds (0.8, 0.8) and radius 0.1
  // d = abs(p - 0.5) * 4.2 - b + vec2(r) = abs(0,0) * 4.2 - (0.8,0.8) + (0.1,0.1) = (0,0) - (0.8,0.8) + (0.1,0.1) = (-0.7,-0.7)
  // min(max(-0.7,-0.7), 0) + length(max((-0.7,-0.7), 0)) - 0.1 = -0.7 + 0 - 0.1 = -0.8
  expectCloseTo([-0.8, 0.0, 0.0], result);
});

test("rectSDF2Round", async () => {
  const src = `
    import lygia::sdf::rectSDF::rectSDF2Round;

    @compute @workgroup_size(1)
    fn foo() {
      let p = vec2f(0.5, 0.5);
      let distance = rectSDF2Round(p, 0.8, 0.1);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Point at (0.5, 0.5) with b=0.8 and r=0.1
  // Calls rectSDF3(p, vec2(b), r) = rectSDF3(p, vec2(0.8), 0.1)
  // Same calculation as rectSDF3 test above: -0.8
  expectCloseTo([-0.8, 0.0, 0.0], result);
});

test("rectSDF - with custom CENTER_2D via constants", async () => {
  const src = `
    import lygia::sdf::rectSDF::rectSDF;

    @compute @workgroup_size(1)
    fn foo() {
      // Test with custom center at (0.3, 0.3)
      let st = vec2f(0.8, 0.3);
      let s = vec2f(1.0, 1.0);
      let distance = rectSDF(st, s);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, {
    elem: "vec3f",
    conditions: { CENTER_2D: true },
    constants: { CENTER_2D: "vec2f(0.3, 0.3)" }
  });
  // With custom center (0.3, 0.3):
  // uv = (0.8, 0.3) - (0.3, 0.3) = (0.5, 0.0)
  // uv *= 2 = (1.0, 0.0)
  // max(abs(1.0/1.0), abs(0.0/1.0)) = max(1.0, 0.0) = 1.0
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("opUnion", async () => {
  const src = `
    import lygia::sdf::opUnion::opUnion;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = 2.0;
      let d2 = 3.0;
      let distance = opUnion(d1, d2);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Union should return minimum
  expectCloseTo([2.0, 0.0, 0.0], result);
});

test("opUnionSmooth", async () => {
  const src = `
    import lygia::sdf::opUnion::opUnionSmooth;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = 1.0;
      let d2 = 1.0;
      let distance = opUnionSmooth(d1, d2, 0.5);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // d1=1.0, d2=1.0, k=0.5
  // h = saturate(0.5 + 0.5*(d2-d1)/k) = saturate(0.5 + 0.5*0/0.5) = 0.5
  // mix(d2, d1, h) - k*h*(1-h) = mix(1, 1, 0.5) - 0.5*0.5*0.5 = 1.0 - 0.125 = 0.875
  expectCloseTo([0.875, 0.0, 0.0], result);
});

test("opUnionSmooth4 with vec4f", async () => {
  const src = `
    import lygia::sdf::opUnion::opUnionSmooth4;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = vec4f(1.0, 0.5, 0.5, 2.0);
      let d2 = vec4f(0.5, 1.0, 0.5, 3.0);
      let result = opUnionSmooth4(d1, d2, 0.5);
      test::results[0] = vec3f(result.x, result.y, result.a);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // d1.a=2.0, d2.a=3.0, k=0.5
  // h = saturate(0.5 + 0.5*(d2.a-d1.a)/k) = saturate(0.5 + 0.5*1.0/0.5) = saturate(1.5) = 1.0
  // result = mix(d2, d1, h) = mix(d2, d1, 1.0) = d1 = (1.0, 0.5, 0.5, 2.0)
  // result.a -= k*h*(1-h) = 2.0 - 0.5*1.0*0.0 = 2.0
  expectCloseTo([1.0, 0.5, 2.0], result);
});

test("opSubtraction", async () => {
  const src = `
    import lygia::sdf::opSubtraction::opSubtraction;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = 2.0;
      let d2 = 3.0;
      let distance = opSubtraction(d1, d2);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // Subtraction: max(-d1, d2) = max(-2, 3) = 3
  expectCloseTo([3.0, 0.0, 0.0], result);
});

test("opSubtraction4 with vec4f", async () => {
  const src = `
    import lygia::sdf::opSubtraction::opSubtraction4;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = vec4f(1.0, 0.5, 0.5, 2.0);
      let d2 = vec4f(0.5, 1.0, 0.5, 3.0);
      let result = opSubtraction4(d1, d2);
      test::results[0] = vec3f(result.x, result.y, result.a);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // select(d2, -d1, -d1.a > d2.a) = select(d2, -d1, -2.0 > 3.0) = select(d2, -d1, false) = d2
  // Returns d2 = (0.5, 1.0, 0.5, 3.0)
  expectCloseTo([0.5, 1.0, 3.0], result);
});

test("opSubtractionSmooth", async () => {
  const src = `
    import lygia::sdf::opSubtraction::opSubtractionSmooth;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = 1.0;
      let d2 = 1.0;
      let distance = opSubtractionSmooth(d1, d2, 0.5);
      test::results[0] = vec3f(distance, 0.0, 0.0);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // d1=1.0, d2=1.0, k=0.5
  // h = saturate(0.5 - 0.5*(d2+d1)/k) = saturate(0.5 - 0.5*2.0/0.5) = saturate(0.5 - 2.0) = 0.0
  // mix(d2, -d1, h) + k*h*(1-h) = mix(1, -1, 0) + 0.5*0*1 = 1.0 + 0 = 1.0
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("opSubtractionSmooth4 with vec4f", async () => {
  const src = `
    import lygia::sdf::opSubtraction::opSubtractionSmooth4;

    @compute @workgroup_size(1)
    fn foo() {
      let d1 = vec4f(1.0, 0.5, 0.5, 2.0);
      let d2 = vec4f(0.5, 1.0, 0.5, 3.0);
      let result = opSubtractionSmooth4(d1, d2, 0.5);
      test::results[0] = vec3f(result.x, result.y, result.a);
    }
  `;
  const result = await testCompute(src, { elem: "vec3f" });
  // d1.a=2.0, d2.a=3.0, k=0.5
  // h = saturate(0.5 - 0.5*(d2.a+d1.a)/k) = saturate(0.5 - 0.5*5.0/0.5) = saturate(0.5 - 5.0) = 0.0
  // result = mix(d2, -d1, h) = mix(d2, -d1, 0) = d2 = (0.5, 1.0, 0.5, 3.0)
  // result.a += k*h*(1-h) = 3.0 + 0.5*0*1 = 3.0
  expectCloseTo([0.5, 1.0, 3.0], result);
});
