#!/bin/bash
# Find WESL files that don't have test coverage
#
# Usage:
#   ./find-untested-wesl.sh           # List all untested WESL files
#   ./find-untested-wesl.sh --count   # Count by category

MODE="list"
if [ "$1" = "--count" ] || [ "$1" = "-c" ]; then
  MODE="count"
fi

# Get list of tested modules (convert lygia::foo::bar to foo/bar)
tested=$(grep "import.*lygia::" test/wesl/*.test.ts | \
  sed 's/.*lygia:://' | \
  sed 's/::[^:]*$//' | \
  sed 's/::/\//g' | \
  sort -u)

# Get all WESL files
all_wesl=$(find . -name "*.wesl" -type f | grep -v node_modules | sed 's|^\./||' | sed 's|\.wesl$||' | sort)

# Find untested files
untested=$(echo "$all_wesl" | while read wesl_path; do
  # Check if this path is in the tested list
  if ! echo "$tested" | grep -q "^${wesl_path}$"; then
    echo "$wesl_path"
  fi
done)

if [ "$MODE" = "count" ]; then
  # Count by category
  echo "$untested" | awk -F/ '{print $1}' | sort | uniq -c | sort -rn
else
  # List all untested files
  echo "$untested"
fi
