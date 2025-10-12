#!/usr/bin/env python3
"""
Split a large test file into multiple smaller, categorized test files.

Usage:
    python3 split-test-file.py <input-file> <config-json>

The config JSON should define categories and their test patterns:
{
  "header": "import { expect, test } from \"vitest\";\nimport { expectCloseTo, testCompute } from \"./testUtil.ts\";\n\n",
  "categories": {
    "output-filename.test.ts": {
      "tests": ["test1", "test2", "test3"],
      "patterns": ["pattern1", "pattern2"]  // optional: match tests containing these strings
    }
  }
}
"""

import re
import sys
import json
from pathlib import Path

def split_test_file(input_file: str, categories: dict, header: str = None):
    """
    Split a test file into multiple categorized files.

    Args:
        input_file: Path to the input test file
        categories: Dict mapping output filenames to their test lists/patterns
        header: Optional header to prepend to each output file
    """
    # Read the entire file
    with open(input_file, 'r') as f:
        content = f.read()

    # Default header if not provided
    if header is None:
        header = '''import { expect, test } from "vitest";
import { expectCloseTo, testCompute } from "./testUtil.ts";

'''

    # Split into individual tests
    # This regex splits on lines that start with "test(" (possibly with .skip)
    tests = re.split(r'\n(?=test(?:\.skip)?\()', content)

    # First part typically has imports (skip it, we use our own header)
    test_blocks = tests[1:] if tests else []

    # Build a map of test name -> test content
    test_map = {}
    for test_block in test_blocks:
        if not test_block.strip():
            continue

        # Extract test name - handles both test("name") and test.skip("name")
        match = re.match(r'test(?:\.skip)?\("([^"]+)"', test_block)
        if match:
            test_name = match.group(1)
            test_map[test_name] = test_block

    # Group tests by category
    categorized_tests = {filename: [] for filename in categories.keys()}
    uncategorized = []

    for test_name, test_content in test_map.items():
        matched = False

        for filename, config in categories.items():
            # Check explicit test list
            if 'tests' in config and test_name in config['tests']:
                categorized_tests[filename].append(test_content)
                matched = True
                break

            # Check patterns
            if 'patterns' in config:
                for pattern in config['patterns']:
                    if pattern in test_name:
                        categorized_tests[filename].append(test_content)
                        matched = True
                        break
                if matched:
                    break

        if not matched:
            uncategorized.append((test_name, test_content))

    # Write output files
    input_path = Path(input_file)
    output_dir = input_path.parent

    for filename, test_list in categorized_tests.items():
        if not test_list:
            print(f"Warning: {filename} has no tests")
            continue

        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(header)
            f.write('\n'.join(test_list))

        print(f"Created {filename} with {len(test_list)} tests")

    # Report uncategorized tests
    if uncategorized:
        print(f"\nWarning: {len(uncategorized)} uncategorized tests:")
        for name, _ in uncategorized:
            print(f"  - {name}")

    # Summary
    total = sum(len(tests) for tests in categorized_tests.values())
    print(f"\nTotal: {total} tests categorized")
    return categorized_tests, uncategorized


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    config_file = sys.argv[2]

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    header = config.get('header', None)
    categories = config.get('categories', {})

    if not categories:
        print("Error: No categories defined in config")
        sys.exit(1)

    # Perform the split
    split_test_file(input_file, categories, header)
    print(f"\nDone! Original file: {input_file}")


if __name__ == '__main__':
    main()
