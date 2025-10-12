#!/usr/bin/env python3
"""
Find LYGIA WESL functions that are not covered by tests.

This script:
1. Scans all .wesl files and extracts function names
2. Scans all .test.ts files and extracts imported lygia:: modules
3. Reports which functions don't have test coverage

Usage:
    python3 scripts/find-untested-functions.py                  # Show all untested functions
    python3 scripts/find-untested-functions.py --count          # Show counts by category
    python3 scripts/find-untested-functions.py --summary        # Show summary only
    python3 scripts/find-untested-functions.py --files          # List only untested .wesl files
    python3 scripts/find-untested-functions.py --skip-indirect  # Exclude indirectly tested functions
    python3 scripts/find-untested-functions.py --show-indirect  # Show only indirectly tested functions
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple


def find_wesl_files(root_dir: Path) -> List[Path]:
    """Find all .wesl files in the repository."""
    wesl_files = []
    for path in root_dir.rglob("*.wesl"):
        # Skip node_modules and other build directories
        if "node_modules" not in str(path) and ".git" not in str(path):
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


def find_test_files(root_dir: Path) -> List[Path]:
    """Find all test.ts files."""
    test_files = []
    test_dir = root_dir / "test" / "wesl"
    if test_dir.exists():
        for path in test_dir.glob("*.test.ts"):
            test_files.append(path)
    return test_files


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
    test_files = find_test_files(root_dir)
    tested_imports: Set[str] = set()

    for test_file in test_files:
        imports = extract_imports_from_test(test_file)
        tested_imports.update(imports)

    print(f"Found {len(tested_imports)} tested imports in {len(test_files)} test files", file=sys.stderr)

    # Compare: find untested functions
    untested_by_category = defaultdict(list)
    indirect_by_category = defaultdict(list)
    files_with_any_test = set()  # Track files that have at least one tested function
    total_functions = 0
    total_tested = 0
    total_indirect = 0

    for module_path, data in sorted(module_functions.items()):
        functions = data['functions']
        file_path = data['file']
        total_functions += len(functions)

        for func_name in functions:
            # Check if this function is tested by looking for:
            # lygia::path::to::module::functionName
            full_import = f"{module_path}::{func_name}"

            if full_import in tested_imports:
                total_tested += 1
                files_with_any_test.add(file_path)  # Mark this file as having a test
            elif full_import in indirectly_tested:
                # Function is indirectly tested
                total_indirect += 1
                category = categorize_module(module_path)
                indirect_by_category[category].append({
                    'module': module_path,
                    'function': func_name,
                    'file': file_path
                })
            else:
                # Function is genuinely untested
                category = categorize_module(module_path)
                untested_by_category[category].append({
                    'module': module_path,
                    'function': func_name,
                    'file': file_path
                })

    total_untested = total_functions - total_tested - total_indirect

    # Output results
    if show_summary:
        print(f"\nSummary:")
        print(f"  Total functions: {total_functions}")
        print(f"  Tested directly: {total_tested} ({100*total_tested/total_functions:.1f}%)")
        if total_indirect > 0:
            print(f"  Tested indirectly: {total_indirect} ({100*total_indirect/total_functions:.1f}%)")
            effective_tested = total_tested + total_indirect
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
            print(f"Coverage: {total_tested}/{total_functions} ({100*total_tested/total_functions:.1f}%)")
    elif show_count:
        if show_indirect:
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

        effective_tested = total_tested + total_indirect
        print(f"\nCoverage: {total_tested} direct + {total_indirect} indirect = {effective_tested}/{total_functions} ({100*effective_tested/total_functions:.1f}%)")
    else:
        # Show detailed list
        if show_indirect:
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
                if total_indirect > 0:
                    print(f"       {total_indirect} indirectly tested (use --show-indirect to see them)")
                    effective_tested = total_tested + total_indirect
                    print(f"Effective coverage: {effective_tested}/{total_functions} ({100*effective_tested/total_functions:.1f}%)")


if __name__ == "__main__":
    main()
