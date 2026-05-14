#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# build_mac.sh — Build TEMseg.app for macOS Apple Silicon
#
# Prerequisites:
#   - Python 3.12+ with the backend venv set up (uv sync / install.py done)
#   - Node.js + npm (for frontend build)
#   - PyInstaller: pip install pyinstaller
#   - weight_manifest.json in project root (URLs can be placeholders for build)
#
# Usage:
#   chmod +x build_mac.sh
#   ./build_mac.sh
#
# Output:
#   dist/TEMseg.app
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  TEMseg macOS Build"
echo "========================================"

# -------------------------------------------
# Step 1: Build frontend static export
# -------------------------------------------
echo ""
echo "[1/4] Building frontend..."

if [ ! -d "frontend/node_modules" ]; then
    echo "  Installing npm dependencies..."
    (cd frontend && npm install)
fi

(cd frontend && npm run build)

if [ ! -f "frontend/out/index.html" ]; then
    echo "ERROR: Frontend build failed — frontend/out/index.html not found"
    exit 1
fi
echo "  Frontend build OK"

# -------------------------------------------
# Step 2: Check weight manifest exists
# -------------------------------------------
echo ""
echo "[2/4] Checking weight manifest..."

if [ ! -f "weight_manifest.json" ]; then
    echo "ERROR: weight_manifest.json not found in project root"
    echo "  Copy it from the template and fill in URLs + checksums"
    exit 1
fi
echo "  Manifest found"

# -------------------------------------------
# Step 3: Ensure PyInstaller is installed
# -------------------------------------------
echo ""
echo "[3/4] Checking PyInstaller..."

if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "  Installing PyInstaller..."
    uv pip install pyinstaller
fi
echo "  PyInstaller OK"

# -------------------------------------------
# Step 4: Run PyInstaller
# -------------------------------------------
echo ""
echo "[4/4] Running PyInstaller..."
echo "  This will take a few minutes..."

# Clean previous builds
rm -rf build/TEMseg dist/TEMseg dist/TEMseg.app

pyinstaller temseg.spec --noconfirm

# -------------------------------------------
# Step 4b: Fix rsciio — PyInstaller fails to bundle its subdirectories
# -------------------------------------------
RSCIIO_SRC=$(python -c "import rsciio; from pathlib import Path; print(Path(rsciio.__file__).parent)")
RSCIIO_DEST="dist/TEMseg.app/Contents/Frameworks/rsciio"

if [ -d "$RSCIIO_SRC" ]; then
    echo "  Copying rsciio from $RSCIIO_SRC ..."
    # Remove the broken/empty one PyInstaller created
    rm -rf "$RSCIIO_DEST"
    # Copy the entire package from the venv
    cp -R "$RSCIIO_SRC" "$RSCIIO_DEST"
    # Verify emd is there
    if [ -f "$RSCIIO_DEST/emd/specifications.yaml" ]; then
        echo "  rsciio/emd OK"
    else
        echo "  WARNING: rsciio/emd/specifications.yaml still missing!"
    fi
else
    echo "  WARNING: Could not find rsciio source at $RSCIIO_SRC"
fi

if [ -d "dist/TEMseg.app" ]; then
    echo ""
    echo "========================================"
    echo "  BUILD SUCCESSFUL"
    echo "========================================"
    echo ""
    echo "  Output: dist"
    echo ""
    SIZE=$(du -sh "dist" | cut -f1)
    echo "  Size: $SIZE"
    echo ""
    # echo "  To run:"
    # echo "    open dist/TEMseg.app"
    echo ""
    echo "  Note: Model weights will be downloaded on first launch"
    echo "  to ~/Library/Application Support/TEMseg/weights/"
    echo ""
else
    echo ""
    echo "ERROR: Build failed — dist/TEMseg.app not found"
    echo "Check the output above for errors."
    exit 1
fi
