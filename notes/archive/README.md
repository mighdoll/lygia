# Archive Directory

This directory contains obsolete documentation files that are no longer actively used but preserved for historical reference.

**Archived:** 2025-10-27

---

## Summary

- **Test Plans:** 22 files (completed work)
- **Obsolete Docs:** 15 files (superseded or one-time use)
- **Total Archived:** 37 files

---

## Test Plans (22 files)

Category-specific test planning documents that were created before tests were implemented. With 99.1% test coverage achieved, these are now historical artifacts.

**Location:** `archive/test-plans/`

**Files:**
- All `test-plan-*.md` files (animation, color, draw, generative, geometry, lighting, math, sdf, space)
- Test plan summaries (geometry, lighting, update)

**Reason for archiving:** Tests for these categories are complete. These planning documents served their purpose but are no longer needed for active development.

---

## Obsolete Documentation (15 files)

**Location:** `archive/obsolete/`

### Superseded Files (4 files)
Files replaced by more comprehensive documentation:

1. **next-steps.md** - Superseded by `status-10-27.md`
2. **next-steps2.md** - Superseded by `status-10-27.md`
3. **untested-wesl-files.md** - Oct 12 data (was 180 files, now 7)
4. **skipped-tests-summary.md** - Oct 12 data (was 16+ tests, now 4)

### Implemented Features (4 files)
Planning documents for features that are now implemented:

5. **test-fragshader.md** - Fragment shader test harness (implemented)
6. **frag-shader-impl.md** - Fragment shader implementation plan (completed)
7. **texture-test-harness-plan.md** - Texture test harness (implemented)
8. **texture-dependent-tests.md** - Texture test strategy (implemented)

### One-Time Use Files (7 files)
Documents created for specific one-time tasks or experiments:

9. **agent-split-generative.md** - Instructions for splitting generative tests (completed)
10. **conditions-fix.md** - Fix notes for @if/@else conditionals (completed)
11. **distribution-test.md** - Distribution build testing (one-time)
12. **image-test.md** - Image testing experiments
13. **image-test-2.md** - Image testing experiments (continuation)
14. **test-constants.md** - Test constant precision notes (one-time fix)
15. **more-interesting.md** - Test improvement brainstorming

---

## Active Documentation (11 files)

These files remain in `notes/` as they're still actively used:

### Essential Reference Files:
1. **DEFERRED-files.md** - 105 deferred files list (texture/sampler heavy)
2. **barrel-files.md** - 21 barrel files to skip (import-only)
3. **indirectly-tested.md** - 175 indirectly tested functions explanation

### Conversion Guidance:
4. **GLSLtoWESL.md** - GLSL→WGSL syntax patterns (updated Oct 20)
5. **GLSL-challenges.md** - Known problematic files
6. **GlslToConvert.md** - Master tracking document
7. **convert-review.md** - Review checklist for 351 WESL files

### Testing & Status:
8. **test-review.md** - What makes good tests
9. **status-10-27.md** - Comprehensive status report (latest)

### Technical Reference:
10. **xyz-scale.md** - XYZ scale pattern documentation
11. **xyz-scale-quick-ref.md** - Quick reference for XYZ scaling

---

## Restoring Archived Files

If you need to reference or restore any archived file:

```bash
# View archived test plan
cat notes/archive/test-plans/test-plan-color-blend.md

# View archived obsolete doc
cat notes/archive/obsolete/next-steps.md

# Restore a file (if needed)
mv notes/archive/obsolete/filename.md notes/
```

---

## History

**2025-10-27:** Initial archival
- Archived 22 completed test plans
- Archived 15 obsolete documentation files
- Reduced active documentation from 48 to 11 files
- Improved notes/ directory organization and maintainability
