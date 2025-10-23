# XYZ Scale Quick Reference

## WESL vs GLSL

| Conversion | GLSL Scale | WESL Scale | Notes |
|------------|-----------|-----------|-------|
| RGB → XYZ | 0-1 ❌ | 0-100 ✅ | GLSL bug: inconsistent with Lab |
| Lab → XYZ | 0-100 ✅ | 0-100 ✅ | Both correct |
| xyY → XYZ | 0-1 ❌ | 0-100 ✅ | Y inherits XYZ scale |
| XYZ → RGB | expects 0-100 ✅ | expects 0-100 ✅ | Both divide by 100 |

## Bug Impact

**GLSL roundtrip failures:**
- RGB → xyY → RGB: ❌ Returns 0.01x too small
- RGB → Lab → RGB: ✅ Works (Lab outputs correct scale)

**WESL fixes:**
- RGB → xyY → RGB: ✅ Perfect roundtrip
- RGB → Lab → RGB: ✅ Still works

## Test Status: 111/123 passing

**12 tests failing:** All need test expectations updated from 0-1 to 0-100 scale.
All color conversions work correctly, tests just expect old scale.

## Action Required

Update test file: `test/wesl/color-space.test.ts`
- Multiply XYZ values by 100 in expectations
- xyY Y component: multiply by 100
- Add `0.01` tolerance where needed
