#!/usr/bin/env python3
"""
Refactor testCompute() calls to use new options object interface.
"""
import re
import sys
from pathlib import Path
from typing import Tuple

def refactor_file(file_path: Path) -> Tuple[int, int, str, bool]:
    """Refactor a single file and return count of changes."""

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    original_content = content
    changes_count = 0

    # Pattern 1: Single-line testCompute(src, "elem", number)
    # Example: testCompute(src, "vec4f", 2) -> testCompute(src, { elem: "vec4f", size: 2 })
    pattern1 = r'testCompute\(([^,\n]+),\s*("[^"]+"),\s*(\d+)\)'
    matches1 = re.findall(pattern1, content)
    changes_count += len(matches1)
    content = re.sub(
        pattern1,
        lambda m: f'testCompute({m.group(1)}, {{ elem: {m.group(2)}, size: {m.group(3)} }})',
        content
    )

    # Pattern 2: Single-line testCompute(src, "elem", variable) where variable is NOT a digit
    # Example: testCompute(src, "vec3f", defines) -> testCompute(src, { elem: "vec3f", conditions: defines })
    pattern2 = r'testCompute\(([^,\n]+),\s*("[^"]+"),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)'
    matches2 = re.findall(pattern2, content)
    changes_count += len(matches2)
    content = re.sub(
        pattern2,
        lambda m: f'testCompute({m.group(1)}, {{ elem: {m.group(2)}, conditions: {m.group(3)} }})',
        content
    )

    # Pattern 3: Single-line testCompute(src, "elem") where elem is NOT "f32"
    # Example: testCompute(src, "vec4f") -> testCompute(src, { elem: "vec4f" })
    # But skip testCompute(src, "f32") -> testCompute(src)
    pattern3 = r'testCompute\(([^,\n]+),\s*("(?!f32")[^"]+")(?!\s*,)'
    matches3 = re.findall(pattern3, content)
    changes_count += len(matches3)
    content = re.sub(
        pattern3,
        lambda m: f'testCompute({m.group(1)}, {{ elem: {m.group(2)} }}',
        content
    )

    # Pattern 4: Single-line testCompute(src, "f32") -> testCompute(src)
    pattern4 = r'testCompute\(([^,\n]+),\s*"f32"\s*\)'
    matches4 = re.findall(pattern4, content)
    changes_count += len(matches4)
    content = re.sub(
        pattern4,
        lambda m: f'testCompute({m.group(1)})',
        content
    )

    # Pattern 5: Multiline testCompute with 4 arguments: (src, "elem", conditions, constants)
    # Example:
    #   testCompute(
    #     src,
    #     "vec2f",
    #     { CENTER_2D: true },
    #     { CENTER_2D: "vec2f(0.3, 0.7)" }
    #   )
    # ->
    #   testCompute(src, {
    #     elem: "vec2f",
    #     conditions: { CENTER_2D: true },
    #     constants: { CENTER_2D: "vec2f(0.3, 0.7)" }
    #   })
    pattern5 = r'testCompute\(\s*\n\s*([^,\n]+),\s*\n\s*("[^"]+"),\s*\n\s*(\{[^}]+\}),\s*\n\s*(\{[^}]+\}),?\s*\n\s*\)'
    def replace5(m):
        src = m.group(1)
        elem = m.group(2)
        conditions = m.group(3)
        constants = m.group(4)
        return f'testCompute({src}, {{\n    elem: {elem},\n    conditions: {conditions},\n    constants: {constants}\n  }})'

    matches5 = re.findall(pattern5, content)
    changes_count += len(matches5)
    content = re.sub(pattern5, replace5, content)

    # Pattern 6: Multiline testCompute with 3 arguments: (src, "elem", conditions_or_size)
    # Example:
    #   testCompute(
    #     src,
    #     "vec3f",
    #     { SOME_DEFINE: true }
    #   )
    # ->
    #   testCompute(src, {
    #     elem: "vec3f",
    #     conditions: { SOME_DEFINE: true }
    #   })
    pattern6 = r'testCompute\(\s*\n\s*([^,\n]+),\s*\n\s*("[^"]+"),\s*\n\s*(\{[^}]+\}),?\s*\n\s*\)'
    def replace6(m):
        src = m.group(1)
        elem = m.group(2)
        conditions = m.group(3)
        return f'testCompute({src}, {{\n    elem: {elem},\n    conditions: {conditions}\n  }})'

    matches6 = re.findall(pattern6, content)
    changes_count += len(matches6)
    content = re.sub(pattern6, replace6, content)

    # Count testCompute calls in this file for reporting
    total_calls = len(re.findall(r'testCompute\(', content))

    return changes_count, total_calls, content, content != original_content


def main():
    test_dir = Path("/Users/lee/wesl/lygia/test/wesl")
    test_files = sorted(test_dir.glob("*.test.ts"))

    total_files_processed = 0
    total_files_changed = 0
    total_changes = 0
    results = []

    for test_file in test_files:
        changes, total_calls, new_content, has_changes = refactor_file(test_file)

        if total_calls > 0:
            total_files_processed += 1

            if has_changes:
                # Write back
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                total_files_changed += 1
                total_changes += changes
                results.append(f"  ✓ {test_file.name}: {changes} calls updated (of {total_calls} total)")
            else:
                results.append(f"  - {test_file.name}: {total_calls} calls (no changes needed)")

    # Print summary
    print("=" * 70)
    print("testCompute() Refactoring Summary")
    print("=" * 70)
    print()

    for result in results:
        print(result)

    print()
    print("=" * 70)
    print(f"Files processed: {total_files_processed}")
    print(f"Files changed: {total_files_changed}")
    print(f"Total calls updated: {total_changes}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
