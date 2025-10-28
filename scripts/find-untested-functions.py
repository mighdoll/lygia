#!/usr/bin/env python3
"""
Find LYGIA WESL functions that are not covered by tests.

This script:
1. Scans all .wesl files and extracts function names
2. Scans unit tests (test/wesl/*.test.ts) and extracts imported lygia:: modules
3. Scans visual regression tests (test/wesl-examples/**/*.test.ts and shaders/*.wesl)
4. Reports which functions don't have test coverage (unit, visual, or indirect)

Coverage categories:
  - Tested directly: Unit tests in test/wesl/
  - Tested visually: Visual regression tests in test/wesl-examples/
  - Tested indirectly: Component-wise wrappers or functions called by tested functions
  - Genuinely untested: Not covered by any of the above

Usage:
    python3 scripts/find-untested-functions.py                  # Show all untested functions
    python3 scripts/find-untested-functions.py --count          # Show counts by category
    python3 scripts/find-untested-functions.py --summary        # Show summary with breakdown
    python3 scripts/find-untested-functions.py --files          # List only untested .wesl files
    python3 scripts/find-untested-functions.py --show-visual    # Show only visual regression tested functions
    python3 scripts/find-untested-functions.py --show-indirect  # Show only indirectly tested functions
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple


def find_wesl_files(root_dir: Path) -> List[Path]:
    """Find all .wesl files in the repository (library files only, not test examples)."""
    wesl_files = []
    for path in root_dir.rglob("*.wesl"):
        path_str = str(path)
        # Skip node_modules, build directories, and test examples
        if ("node_modules" not in path_str and
            ".git" not in path_str and
            "test/wesl-examples" not in path_str):
            wesl_files.append(path)
    return wesl_files


def extract_module_path(wesl_file: Path, root_dir: Path) -> str:
    """Convert file path to LYGIA module path.

    Example: math/saturate.wesl -> lygia::math::saturate
             color/blend/add.wesl -> lygia::color::blend::add
    """
    rel_path = wesl_file.relative_to(root_dir)
    parts = list(rel_path.parts[:-1]) + [rel_path.stem]  # Remove .wesl extension
    return "lygia::" + "::".join(parts)


def extract_functions_from_wesl(wesl_file: Path) -> List[str]:
    """Extract function names from a WESL file.

    Looks for patterns like: fn functionName(...) -> type
    """
    functions = []
    try:
        with open(wesl_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Match function declarations: fn name(...)
            # Matches: fn functionName(...) or fn functionName2(...)
            pattern = r'\bfn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            matches = re.finditer(pattern, content)
            for match in matches:
                func_name = match.group(1)
                # Skip internal helper functions (starting with underscore)
                if not func_name.startswith('_'):
                    functions.append(func_name)
    except Exception as e:
        print(f"Warning: Could not read {wesl_file}: {e}", file=sys.stderr)

    return functions


def find_test_files(root_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Find all test.ts files in test/wesl/ and test/wesl-examples/.

    Returns:
        (unit_test_files, visual_test_files) - separate lists for unit and visual regression tests
    """
    unit_test_files = []
    visual_test_files = []

    # Unit tests in test/wesl/
    test_dir = root_dir / "test" / "wesl"
    if test_dir.exists():
        for path in test_dir.glob("*.test.ts"):
            unit_test_files.append(path)

    # Visual regression tests in test/wesl-examples/
    examples_dir = root_dir / "test" / "wesl-examples"
    if examples_dir.exists():
        for path in examples_dir.rglob("*.test.ts"):
            visual_test_files.append(path)

    return unit_test_files, visual_test_files


def extract_imports_from_test(test_file: Path) -> Set[str]:
    """Extract imported LYGIA modules from a test file.

    Looks for patterns like: import lygia::math::saturate::saturate;
    Returns the full module paths (including function names).
    """
    imports = set()
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Match LYGIA imports: import lygia::path::to::function;
            pattern = r'import\s+(lygia::[a-zA-Z0-9_:]+);'
            matches = re.finditer(pattern, content)
            for match in matches:
                import_path = match.group(1)
                imports.add(import_path)
    except Exception as e:
        print(f"Warning: Could not read {test_file}: {e}", file=sys.stderr)

    return imports


def extract_imports_from_shader_files(root_dir: Path) -> Set[str]:
    """Extract LYGIA imports from shader files used in visual regression tests.

    Scans .wesl files in test/wesl-examples/shaders/ directory.
    Returns the full module paths (including function names).
    """
    imports = set()
    shaders_dir = root_dir / "test" / "wesl-examples" / "shaders"

    if not shaders_dir.exists():
        return imports

    for shader_file in shaders_dir.glob("*.wesl"):
        try:
            with open(shader_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Match LYGIA imports: import lygia::path::to::function;
                pattern = r'import\s+(lygia::[a-zA-Z0-9_:]+);'
                matches = re.finditer(pattern, content)
                for match in matches:
                    import_path = match.group(1)
                    imports.add(import_path)
        except Exception as e:
            print(f"Warning: Could not read {shader_file}: {e}", file=sys.stderr)

    return imports


def categorize_module(module_path: str) -> str:
    """Extract category from module path.

    Example: lygia::math::saturate -> math
             lygia::color::blend::add -> color/blend
    """
    parts = module_path.split("::")
    if len(parts) < 2:
        return "unknown"

    # For multi-level categories like color/blend
    if len(parts) >= 4:
        return f"{parts[1]}/{parts[2]}"
    return parts[1]


def load_indirectly_tested(script_dir: Path) -> Set[str]:
    """Load list of indirectly tested functions from indirectly-tested.txt.

    Returns a set of full import paths (e.g., lygia::math::cubic::cubic2)
    Lines starting with # or empty lines are ignored.
    """
    indirectly_tested = set()
    indirect_file = script_dir / "indirectly-tested.txt"

    if not indirect_file.exists():
        return indirectly_tested

    try:
        with open(indirect_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    indirectly_tested.add(line)
    except Exception as e:
        print(f"Warning: Could not read {indirect_file}: {e}", file=sys.stderr)

    return indirectly_tested


def main():
    # Parse command line arguments
    show_count = "--count" in sys.argv
    show_summary = "--summary" in sys.argv
    show_files = "--files" in sys.argv
    skip_indirect = "--skip-indirect" in sys.argv
    show_indirect = "--show-indirect" in sys.argv
    show_visual = "--show-visual" in sys.argv

    # Get repository root (parent of scripts directory)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # Load indirectly tested functions
    indirectly_tested = load_indirectly_tested(script_dir)
    if indirectly_tested:
        print(f"Loaded {len(indirectly_tested)} indirectly tested functions from indirectly-tested.txt", file=sys.stderr)

    print("Scanning for WESL files and functions...", file=sys.stderr)

    # Find all WESL files and extract functions
    wesl_files = find_wesl_files(root_dir)
    module_functions: Dict[str, Dict[str, List[str]]] = {}

    for wesl_file in wesl_files:
        module_path = extract_module_path(wesl_file, root_dir)
        functions = extract_functions_from_wesl(wesl_file)

        if functions:
            rel_path = wesl_file.relative_to(root_dir)
            module_functions[module_path] = {
                'functions': functions,
                'file': str(rel_path)
            }

    print(f"Found {len(module_functions)} WESL modules with {sum(len(m['functions']) for m in module_functions.values())} functions", file=sys.stderr)

    # Find all test files and extract imports
    print("Scanning test files...", file=sys.stderr)
    unit_test_files, visual_test_files = find_test_files(root_dir)

    # Extract imports from unit tests
    unit_tested_imports: Set[str] = set()
    for test_file in unit_test_files:
        imports = extract_imports_from_test(test_file)
        unit_tested_imports.update(imports)

    # Extract imports from visual regression shaders
    visual_tested_imports = extract_imports_from_shader_files(root_dir)

    # Combined: all directly tested (unit + visual)
    all_tested_imports = unit_tested_imports | visual_tested_imports

    print(f"Found {len(unit_tested_imports)} unit tested imports in {len(unit_test_files)} test files", file=sys.stderr)
    print(f"Found {len(visual_tested_imports)} visual regression tested imports in {len(visual_test_files)} visual test files", file=sys.stderr)

    # Compare: find untested functions
    untested_by_category = defaultdict(list)
    indirect_by_category = defaultdict(list)
    visual_by_category = defaultdict(list)
    files_with_any_test = set()  # Track files that have at least one tested function
    total_functions = 0
    total_unit_tested = 0
    total_visual_tested = 0
    total_indirect = 0

    for module_path, data in sorted(module_functions.items()):
        functions = data['functions']
        file_path = data['file']
        total_functions += len(functions)

        for func_name in functions:
            # Check if this function is tested by looking for:
            # lygia::path::to::module::functionName
            full_import = f"{module_path}::{func_name}"

            is_tested = False
            category = categorize_module(module_path)

            if full_import in unit_tested_imports:
                total_unit_tested += 1
                files_with_any_test.add(file_path)
                is_tested = True

            if full_import in visual_tested_imports:
                # Function is tested via visual regression
                total_visual_tested += 1
                files_with_any_test.add(file_path)
                visual_by_category[category].append({
                    'module': module_path,
                    'function': func_name,
                    'file': file_path
                })
                is_tested = True

            if full_import in indirectly_tested:
                # Function is indirectly tested
                total_indirect += 1
                indirect_by_category[category].append({
                    'module': module_path,
                    'function': func_name,
                    'file': file_path
                })
                is_tested = True

            if not is_tested:
                # Function is genuinely untested
                untested_by_category[category].append({
                    'module': module_path,
                    'function': func_name,
                    'file': file_path
                })

    # Calculate total untested: functions not in any test category
    all_tested_functions = (unit_tested_imports | visual_tested_imports | indirectly_tested)
    all_functions = {f"{m}::{f}" for m, d in module_functions.items() for f in d['functions']}
    total_untested = len(all_functions - all_tested_functions)

    # Output results
    if show_summary:
        print(f"\nSummary:")
        print(f"  Total functions: {total_functions}")
        print(f"  Tested directly: {total_unit_tested} ({100*total_unit_tested/total_functions:.1f}%)")
        if total_visual_tested > 0:
            print(f"  Tested visually: {total_visual_tested} ({100*total_visual_tested/total_functions:.1f}%)")
        if total_indirect > 0:
            print(f"  Tested indirectly: {total_indirect} ({100*total_indirect/total_functions:.1f}%)")
        effective_tested = total_unit_tested + total_visual_tested + total_indirect
        print(f"  Effective coverage: {effective_tested} ({100*effective_tested/total_functions:.1f}%)")
        print(f"  Genuinely untested: {total_untested} ({100*total_untested/total_functions:.1f}%)")
        return

    if show_files:
        # Show only files with NO tested functions (completely untested files)
        files_with_untested = set()
        for category in untested_by_category.values():
            for item in category:
                files_with_untested.add(item['file'])

        # Files with no tests = files with untested functions - files with any test
        completely_untested_files = files_with_untested - files_with_any_test

        if not completely_untested_files:
            print("\n🎉 All WESL files have at least some test coverage!")
        else:
            print(f"\nFiles with no test coverage ({len(completely_untested_files)} files):\n")
            for file_path in sorted(completely_untested_files):
                print(file_path)
            print(f"\nTotal: {len(completely_untested_files)} files with no tests")
            total_tested = total_unit_tested + total_visual_tested
            print(f"Coverage: {total_tested}/{total_functions} ({100*total_tested/total_functions:.1f}%)")
    elif show_count:
        if show_visual:
            print(f"\nVisual regression tested functions by category:")
            print(f"{'Category':<20} {'Count'}")
            print("=" * 35)
            for category in sorted(visual_by_category.keys()):
                count = len(visual_by_category[category])
                print(f"{category:<20} {count:>5}")
            print("=" * 35)
            print(f"{'Total':<20} {total_visual_tested:>5}")
        elif show_indirect:
            print(f"\nIndirectly tested functions by category:")
            print(f"{'Category':<20} {'Count'}")
            print("=" * 35)
            for category in sorted(indirect_by_category.keys()):
                count = len(indirect_by_category[category])
                print(f"{category:<20} {count:>5}")
            print("=" * 35)
            print(f"{'Total':<20} {total_indirect:>5}")
        else:
            print(f"\nUntested functions by category:")
            print(f"{'Category':<20} {'Count'}")
            print("=" * 35)
            for category in sorted(untested_by_category.keys()):
                count = len(untested_by_category[category])
                print(f"{category:<20} {count:>5}")
            print("=" * 35)
            print(f"{'Total':<20} {total_untested:>5}")
            if total_indirect > 0 and not skip_indirect:
                print(f"\n(Note: {total_indirect} functions are indirectly tested - use --show-indirect to see them)")

        effective_tested = total_unit_tested + total_visual_tested + total_indirect
        total_tested = total_unit_tested + total_visual_tested
        print(f"\nCoverage: {total_unit_tested} unit + {total_visual_tested} visual + {total_indirect} indirect = {effective_tested}/{total_functions} ({100*effective_tested/total_functions:.1f}%)")
    else:
        # Show detailed list
        if show_visual:
            # Show visual regression tested functions
            if not visual_by_category:
                print("\n✨ No visual regression tested functions found")
            else:
                print(f"\nVisual regression tested functions ({total_visual_tested} total):\n")

                for category in sorted(visual_by_category.keys()):
                    items = visual_by_category[category]
                    print(f"\n{category}/ ({len(items)} visual):")
                    print("-" * 60)

                    for item in sorted(items, key=lambda x: (x['file'], x['function'])):
                        print(f"  {item['file']}")
                        print(f"    fn {item['function']}()")
                        print(f"    import {item['module']}::{item['function']}")

                print(f"\n" + "=" * 60)
                print(f"Total: {total_visual_tested} visual regression tested")
        elif show_indirect:
            # Show indirectly tested functions
            if not indirect_by_category:
                print("\n✨ No indirectly tested functions found")
            else:
                print(f"\nIndirectly tested functions ({total_indirect} total):\n")

                for category in sorted(indirect_by_category.keys()):
                    items = indirect_by_category[category]
                    print(f"\n{category}/ ({len(items)} indirect):")
                    print("-" * 60)

                    for item in sorted(items, key=lambda x: (x['file'], x['function'])):
                        print(f"  {item['file']}")
                        print(f"    fn {item['function']}()")
                        print(f"    import {item['module']}::{item['function']}")

                print(f"\n" + "=" * 60)
                print(f"Total: {total_indirect} indirectly tested")
        else:
            # Show genuinely untested functions
            if not untested_by_category:
                print("\n🎉 All WESL functions are tested!")
            else:
                print(f"\nGenuinely untested functions ({total_untested} total):\n")

                for category in sorted(untested_by_category.keys()):
                    items = untested_by_category[category]
                    print(f"\n{category}/ ({len(items)} untested):")
                    print("-" * 60)

                    for item in sorted(items, key=lambda x: (x['file'], x['function'])):
                        print(f"  {item['file']}")
                        print(f"    fn {item['function']}()")
                        print(f"    import {item['module']}::{item['function']}")

                print(f"\n" + "=" * 60)
                print(f"Total: {total_untested} genuinely untested")
                if total_visual_tested > 0:
                    print(f"       {total_visual_tested} visual regression tested (use --show-visual to see them)")
                if total_indirect > 0:
                    print(f"       {total_indirect} indirectly tested (use --show-indirect to see them)")
                if total_visual_tested > 0 or total_indirect > 0:
                    effective_tested = total_unit_tested + total_visual_tested + total_indirect
                    print(f"Effective coverage: {effective_tested}/{total_functions} ({100*effective_tested/total_functions:.1f}%)")


if __name__ == "__main__":
    main()
