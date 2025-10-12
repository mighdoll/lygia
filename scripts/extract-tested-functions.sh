#!/bin/bash
# Extract all tested function names from test files
# This helps identify which WESL files have test coverage

echo "=== TESTED WESL FUNCTIONS ==="
echo

for test_file in test/wesl/*.test.ts; do
  echo "# $(basename $test_file)"
  # Extract import statements and convert to file paths
  grep "import.*lygia::" "$test_file" | \
    sed 's/.*lygia:://' | \
    sed 's/::[^:]*$//' | \
    sed 's/::/\//g' | \
    sort -u
  echo
done
