#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# build_mac.sh — Build TEMseg.app for macOS Apple Silicon
#
# Prerequisites:
#   - Python 3.11+ with the backend venv set up (uv sync / install.py done)
#   - Node.js + npm (for frontend build)
#   - PyInstaller: uv pip install pyinstaller
#   - weight_manifest.json in project root (URLs can be placeholders for build)
#
# Usage:
#   chmod +x build_mac.sh
#   ./build_mac.sh                       # incremental build (fastest)
#   ./build_mac.sh --skip-frontend       # skip frontend rebuild
#   ./build_mac.sh --clean               # wipe build/ and dist/ first
#   ./build_mac.sh --clean --skip-frontend
#
# Output:
#   dist/TEMseg.app
# ============================================================================

# -------------------------------------------
# Parse flags
# -------------------------------------------
CLEAN=0
SKIP_FRONTEND=0
for arg in "$@"; do
    case "$arg" in
        --clean)          CLEAN=1 ;;
        --skip-frontend)  SKIP_FRONTEND=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--clean] [--skip-frontend]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Embed the current git branch in the output name so builds are traceable.
# Sanitise: replace any non-alphanumeric char with '-', strip leading/trailing dashes.
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null | sed 's/[^a-zA-Z0-9]/-/g; s/^-//; s/-$//' || echo "unknown")
APP_NAME="TEMseg-${BRANCH}.app"

echo "========================================"
echo "  TEMseg macOS Build"
echo "  Branch: $BRANCH"
echo "  Output: dist/$APP_NAME"
[ "$CLEAN" -eq 1 ]         && echo "  Mode: CLEAN"
[ "$SKIP_FRONTEND" -eq 1 ] && echo "  Skipping frontend rebuild"
echo "========================================"

# -------------------------------------------
# Step 1: Check weight manifest (cheap)
# -------------------------------------------
echo ""
echo "[1/4] Checking weight manifest..."

if [ ! -f "weight_manifest.json" ]; then
    echo "ERROR: weight_manifest.json not found in project root"
    echo "  Copy it from the template and fill in URLs + checksums"
    exit 1
fi
echo "  Manifest found"

if [ ! -f "temseg_icon.icns" ]; then
    echo "WARNING: temseg_icon.icns not found - build will use default PyInstaller icon"
fi

# -------------------------------------------
# Step 2: Ensure PyInstaller is installed (cheap)
# -------------------------------------------
echo ""
echo "[2/4] Checking PyInstaller..."

if ! uv run python -c "import PyInstaller" 2>/dev/null; then
    echo "  Installing PyInstaller..."
    uv pip install pyinstaller
fi
echo "  PyInstaller OK"

# -------------------------------------------
# Step 3: Build frontend static export
# -------------------------------------------
echo ""
if [ "$SKIP_FRONTEND" -eq 0 ]; then
    echo "[3/4] Building frontend..."

    if [ ! -d "frontend/node_modules" ]; then
        echo "  Installing npm dependencies..."
        (cd frontend && npm install)
    fi

    (cd frontend && npm run build)

    if [ ! -f "frontend/out/index.html" ]; then
        echo "ERROR: Frontend build failed - frontend/out/index.html not found"
        exit 1
    fi
    echo "  Frontend build OK"
else
    echo "[3/4] Skipping frontend build (--skip-frontend)"
    if [ ! -f "frontend/out/index.html" ]; then
        echo "ERROR: Cannot skip frontend - frontend/out/index.html doesn't exist yet"
        echo "       Run without --skip-frontend first"
        exit 1
    fi
fi

# -------------------------------------------
# Step 4: Run PyInstaller
# -------------------------------------------
echo ""
echo "[4/4] Running PyInstaller..."

if [ "$CLEAN" -eq 1 ]; then
    echo "  Clean build requested - wiping build/ and dist/ ..."
    rm -rf build/TEMseg dist/TEMseg dist/TEMseg.app "dist/${APP_NAME}"
    echo "  This will take ~10 minutes..."
else
    echo "  Incremental build - reusing build/ cache..."
    echo "  (Pass --clean for a full rebuild if spec/deps changed)"
fi

uv run pyinstaller temseg.spec --noconfirm

# Rename the default output to include branch name.
if [ -d "dist/TEMseg.app" ]; then
    mv "dist/TEMseg.app" "dist/${APP_NAME}"
fi

# -------------------------------------------
# Step 4b: Fix rsciio - PyInstaller fails to bundle its subdirectories
# -------------------------------------------
RSCIIO_SRC=$(uv run python -c "import rsciio; from pathlib import Path; print(Path(rsciio.__file__).parent)")
RSCIIO_DEST="dist/${APP_NAME}/Contents/Frameworks/rsciio"

if [ -d "$RSCIIO_SRC" ]; then
    echo "  Copying rsciio from $RSCIIO_SRC ..."
    rm -rf "$RSCIIO_DEST"
    cp -R "$RSCIIO_SRC" "$RSCIIO_DEST"
    if [ -f "$RSCIIO_DEST/emd/specifications.yaml" ]; then
        echo "  rsciio/emd OK"
    else
        echo "  WARNING: rsciio/emd/specifications.yaml still missing!"
    fi
else
    echo "  WARNING: Could not find rsciio source at $RSCIIO_SRC"
fi

# -------------------------------------------
# Done
# -------------------------------------------
if [ -d "dist/${APP_NAME}" ]; then
    echo ""
    echo "========================================"
    echo "  BUILD SUCCESSFUL"
    echo "========================================"
    echo ""
    echo "  Output: dist/${APP_NAME}"
    SIZE=$(du -sh "dist/${APP_NAME}" | cut -f1)
    echo "  Size:   $SIZE"
    echo ""
    echo "  Note: Model weights will be downloaded on first launch"
    echo "  to ~/Library/Application Support/TEMseg/weights/"
    echo ""
else
    echo ""
    echo "ERROR: Build failed - dist/${APP_NAME} not found"
    echo "Check the output above for errors."
    exit 1
fi
