#!/bin/bash
# build_mac.sh - TEMseg macOS build (TIER 1)
#
# Self-contained tier-1 pipeline: uses temseg_tier1.spec (this directory) and
# namespaced output dirs so it never collides with the original build or tier2.
# See README.md in this directory for the full list of changes vs
# scripts/mac/build_mac.sh.
#
# Usage: ./scripts/mac/tier1/build_mac.sh [--clean] [--skip-frontend] [--force-frontend]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "")"
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$REPO_ROOT"

TIER="tier1"
SPEC_FILE="scripts/mac/${TIER}/temseg_${TIER}.spec"
DIST_DIR="dist/${TIER}"
WORK_DIR="build/${TIER}"
STAMP_FILE="${DIST_DIR}/.frontend-hash"

VENV_PYINSTALLER="${REPO_ROOT}/.venv/bin/pyinstaller"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

CLEAN_BUILD=false
SKIP_FRONTEND=false
FORCE_FRONTEND=${TEMSEG_FORCE_FRONTEND:-false}

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --force-frontend)
            FORCE_FRONTEND=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--skip-frontend] [--force-frontend]"
            exit 1
            ;;
    esac
done

# timing helpers
NOW_S() { date +%s; }
T_TOTAL_START="$(NOW_S)"

echo "========================================"
echo "TEMseg Build Script (macOS, ${TIER})"
echo "========================================"

# STEP 0: Clean previous build artifacts if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "STEP 0: Cleaning previous build artifacts..."
    rm -rf "${WORK_DIR}" "${DIST_DIR}"
    echo "Clean complete."
fi

# STEP 1: Frontend build (auto-skipped when frontend/ is unchanged)
echo "STEP 1: Building frontend..."
T0="$(NOW_S)"

frontend_hash() {
    # Hash of committed tree + any uncommitted changes under frontend/.
    # Changes value whenever tracked content or the set of changed/untracked
    # files changes, so the stamp only goes stale when the frontend really did.
    {
        git ls-tree -r HEAD -- frontend 2>/dev/null || echo "no-git"
        git status --porcelain=v1 -- frontend 2>/dev/null || echo "no-git"
    } | shasum -a 256 | awk '{print $1}'
}
CURR_HASH="$(frontend_hash)"
PREV_HASH=""
if [ -f "$STAMP_FILE" ]; then
    PREV_HASH="$(cat "$STAMP_FILE")"
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "Installing Node Modules..."
    (cd frontend && npm install)
fi

if [ "$SKIP_FRONTEND" = "true" ]; then
    echo "Skipping frontend build (--skip-frontend)"
elif [ "$FORCE_FRONTEND" != "true" ] && [ -f "frontend/out/index.html" ] && [ "$CURR_HASH" = "$PREV_HASH" ]; then
    echo "Unchanged since last build (hash ${CURR_HASH:0:12}) - skipping."
else
    echo "Building Next.js static export..."
    (cd frontend && npm run build)
    mkdir -p "$DIST_DIR"
    echo "$CURR_HASH" > "$STAMP_FILE"
    echo "Frontend export completed."
fi
T_FRONTEND=$(( $(NOW_S) - T0 ))

# STEP 2: Check environment
echo "STEP 2: Checking Python environment..."
T0="$(NOW_S)"

if [ ! -f .env ]; then
    echo ".env does not exist, creating one..."
    touch .env
else
    echo ".env already exists, skipping..."
fi

if [ ! -x "$VENV_PYINSTALLER" ]; then
    echo "Error: PyInstaller not found at $VENV_PYINSTALLER"
    echo "Hint: 'uv pip install pyinstaller' (PyInstaller is NOT in uv.lock)"
    exit 1
fi

# Dependency sanity gate - several build inputs (torch, PyInstaller, filterpy,
# ...) are installed OUTSIDE uv.lock via uv pip / install tooling. Anything
# sync-like (uv sync, uv run) will silently REMOVE them; PyInstaller would
# then only fail several minutes into the build. find_spec is import-free and
# takes <1s total.
MISSING=""
for mod in torch torchvision numpy ultralytics hyperspy rsciio webview PyInstaller; do
    if ! "$VENV_PYTHON" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)"; then
        MISSING="${MISSING} ${mod}"
    fi
done
if [ -n "$MISSING" ]; then
    echo "Error: missing python modules in .venv:${MISSING}"
    echo "Install them first (uv pip install ...), then rerun."
    exit 1
fi
echo "PyInstaller found: $("$VENV_PYINSTALLER" --version) - all modules present, building app bundle..."
T_ENV=$(( $(NOW_S) - T0 ))

# STEP 4: Build executable with PyInstaller (tier-1 spec, tier-namespaced dirs)
echo "STEP 4: Building application..."
T0="$(NOW_S)"
echo "This may take several minutes..."
"$VENV_PYINSTALLER" "$SPEC_FILE" \
    --distpath "$DIST_DIR" \
    --workpath "$WORK_DIR" \
    --noconfirm
echo "Application build completed!"
T_PYINSTALLER=$(( $(NOW_S) - T0 ))

# STEP 4b: Bundle rsciio (RosettaSciIO) files explicitly to fix "specifications.yaml not found"
# Legacy safety net kept from the original script; cheap and harmless.
echo "STEP 4b: Bundling rsciio files..."
T0="$(NOW_S)"
RSCIIO_PATH="$("$VENV_PYTHON" -c "import rsciio, os; print(os.path.dirname(rsciio.__file__))")"
RSCIIO_DEST="${DIST_DIR}/TEMseg.app/Contents/Frameworks/rsciio"

if [ -d "$RSCIIO_PATH" ]; then
    echo "Copying rsciio from $RSCIIO_PATH to bundle..."
    mkdir -p "$(dirname "$RSCIIO_DEST")"
    if [ -d "$RSCIIO_DEST" ]; then
        rm -rf "$RSCIIO_DEST"
    fi
    cp -R "$RSCIIO_PATH" "$RSCIIO_DEST"
    echo "rsciio files copied successfully."
else
    echo "WARNING: rsciio package not found at $RSCIIO_PATH"
fi
T_POST=$(( $(NOW_S) - T0 ))

# STEP 5: Rename application based on git branch
echo "STEP 5: Preparing application..."
T0="$(NOW_S)"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
SAFE_BRANCH="${BRANCH//\//-}.app"
TARGET_APP="${DIST_DIR}/TEMseg-${SAFE_BRANCH}"
SOURCE_APP="${DIST_DIR}/TEMseg.app"

if [ -d "$SOURCE_APP" ]; then
    if [ -d "$TARGET_APP" ]; then
        echo "Removing existing $TARGET_APP..."
        rm -rf "$TARGET_APP"
    fi

    mv "$SOURCE_APP" "$TARGET_APP"
    echo "Successfully built and renamed to $TARGET_APP"
    echo "App size: $(du -sh "$TARGET_APP" | cut -f1)"
    echo "You can now distribute your app!"
else
    echo "Error: Application not built successfully"
    exit 1
fi
T_RENAME=$(( $(NOW_S) - T0 ))
T_TOTAL=$(( $(NOW_S) - T_TOTAL_START ))

# timing summary + log
mkdir -p "$WORK_DIR"
LOG_CSV="${WORK_DIR}/build_timings.csv"
SHORT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [ ! -f "$LOG_CSV" ]; then
    echo "date,tier,branch,sha,frontend_s,env_s,pyinstaller_s,post_s,rename_s,total_s" > "$LOG_CSV"
fi
echo "$(date '+%Y-%m-%d %H:%M:%S'),${TIER},${BRANCH},${SHORT_SHA},${T_FRONTEND},${T_ENV},${T_PYINSTALLER},${T_POST},${T_RENAME},${T_TOTAL}" >> "$LOG_CSV"

echo ""
echo "========================================"
echo "Build timing summary"
echo "========================================"
printf "  frontend    : %ss\n" "${T_FRONTEND}"
printf "  env check   : %ss\n" "${T_ENV}"
printf "  pyinstaller : %ss\n" "${T_PYINSTALLER}"
printf "  post (rsciio): %ss\n" "${T_POST}"
printf "  rename      : %ss\n" "${T_RENAME}"
printf "  TOTAL       : %ss\n" "${T_TOTAL}"
echo "(appended to ${LOG_CSV})"
echo ""
echo "Build complete!"
