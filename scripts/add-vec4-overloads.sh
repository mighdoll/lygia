#!/bin/bash

# Script to add missing vec4 overloads to color/space WESL files

cd /Users/lee/wesl/lygia/color/space

# Files that need vec4 overload added: func(vec3) -> vec3 plus func4(vec4) -> vec4

files=(
    "lms2rgb"
    "oklab2rgb"
    "oklab2srgb"
    "rgb2YCbCr"
    "rgb2YPbPr"
    "rgb2hcv"
    "rgb2hcy"
    "rgb2heat"
    "rgb2hsl"
    "rgb2hsv"
    "rgb2hue"
    "rgb2lab"
    "rgb2lch"
    "rgb2lms"
    "rgb2luma"
    "rgb2oklab"
    "rgb2xyY"
    "rgb2xyz"
    "rgb2yiq"
    "rgb2yuv"
    "srgb2lab"
    "srgb2lch"
    "srgb2luma"
    "srgb2oklab"
    "srgb2xyz"
    "xyY2rgb"
    "xyY2srgb"
    "xyY2xyz"
    "xyz2lab"
    "xyz2rgb"
    "xyz2srgb"
    "xyz2xyY"
    "yiq2rgb"
    "yuv2rgb"
)

for fn in "${files[@]}"; do
    file="${fn}.wesl"
    if [ -f "$file" ]; then
        # Check if already has vec4 overload
        if ! grep -q "fn ${fn}4" "$file"; then
            echo "Adding vec4 overload to $file"
            # Add the vec4 overload at the end
            echo "" >> "$file"
            echo "fn ${fn}4(color: vec4f) -> vec4f {" >> "$file"
            # Determine which component to use (rgb, xyz, or general)
            if [[ "$fn" == *"rgb"* ]] || [[ "$fn" == *"RGB"* ]]; then
                echo "    return vec4f(${fn}(color.rgb), color.a);" >> "$file"
            elif [[ "$fn" == *"xyz"* ]] || [[ "$fn" == *"XYZ"* ]] || [[ "$fn" == *"xyY"* ]]; then
                echo "    return vec4f(${fn}(color.xyz), color.a);" >> "$file"
            else
                echo "    return vec4f(${fn}(color.rgb), color.a);" >> "$file"
            fi
            echo "}" >> "$file"
        else
            echo "Skipping $file (already has vec4 overload)"
        fi
    fi
done

echo "Done!"
