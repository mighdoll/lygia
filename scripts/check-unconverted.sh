#!/bin/bash
# Check for GLSL files that don't have corresponding WESL files
#
# Usage:
#   ./check-unconverted.sh              # Show all unconverted files
#   ./check-unconverted.sh --skip-barrels  # Exclude barrel files (import-only)

SKIP_BARRELS=false

# Parse arguments
if [ "$1" = "--skip-barrels" ] || [ "$1" = "-s" ]; then
  SKIP_BARRELS=true
fi

# Barrel files to skip (import-only files with no implementations)
BARREL_FILES=(
  "./math.glsl"
  "./sdf.glsl"
  "./animation/easing.glsl"
  "./animation/easing/back.glsl"
  "./animation/easing/bounce.glsl"
  "./animation/easing/circular.glsl"
  "./animation/easing/cubic.glsl"
  "./animation/easing/elastic.glsl"
  "./animation/easing/exponential.glsl"
  "./animation/easing/linear.glsl"
  "./animation/easing/quadratic.glsl"
  "./animation/easing/quartic.glsl"
  "./animation/easing/quintic.glsl"
  "./animation/easing/sine.glsl"
  "./color/blend.glsl"
  "./color/layer.glsl"
  "./color/composite.glsl"
  "./geometry/aabb.glsl"
  "./geometry/triangle.glsl"
  "./sample.glsl"
  "./color/palette/wada.glsl"
)

# Function to check if a file is in the barrel list
is_barrel_file() {
  local file="$1"
  for barrel in "${BARREL_FILES[@]}"; do
    if [ "$file" = "$barrel" ]; then
      return 0
    fi
  done
  return 1
}

find . -name "*.glsl" -type f | grep -v node_modules | while IFS= read -r f; do
  wesl_file="${f%.glsl}.wesl"
  if [ ! -f "$wesl_file" ]; then
    # Skip barrel files if requested
    if $SKIP_BARRELS && is_barrel_file "$f"; then
      continue
    fi
    echo "$f"
  fi
done
