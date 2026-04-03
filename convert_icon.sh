#!/usr/bin/env bash
# convert_icon.sh — Convert a 1024x1024 PNG to .icns
#
# Usage: ./convert_icon.sh temseg_icon.png
#
# Requires: sips (built into macOS), iconutil (built into macOS)

set -euo pipefail

INPUT="${1:-temseg_icon.png}"
OUTPUT="${INPUT%.png}.icns"
ICONSET="${INPUT%.png}.iconset"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found"
    exit 1
fi

echo "Creating iconset from $INPUT..."
mkdir -p "$ICONSET"

# Generate all required sizes
sips -z 16 16     "$INPUT" --out "$ICONSET/icon_16x16.png"      >/dev/null
sips -z 32 32     "$INPUT" --out "$ICONSET/icon_16x16@2x.png"   >/dev/null
sips -z 32 32     "$INPUT" --out "$ICONSET/icon_32x32.png"      >/dev/null
sips -z 64 64     "$INPUT" --out "$ICONSET/icon_32x32@2x.png"   >/dev/null
sips -z 128 128   "$INPUT" --out "$ICONSET/icon_128x128.png"    >/dev/null
sips -z 256 256   "$INPUT" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256   "$INPUT" --out "$ICONSET/icon_256x256.png"    >/dev/null
sips -z 512 512   "$INPUT" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512   "$INPUT" --out "$ICONSET/icon_512x512.png"    >/dev/null
sips -z 1024 1024 "$INPUT" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

echo "Building .icns..."
iconutil -c icns "$ICONSET" -o "$OUTPUT"

# Cleanup
rm -rf "$ICONSET"

echo "Done: $OUTPUT"
