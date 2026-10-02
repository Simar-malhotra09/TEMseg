#!/usr/bin/env bash
# nightly build + API test for TEMseg.
#
# Builds the tier-2 .app (clean, reusing the existing frontend export), runs
# the API test suite against it, then removes the temporary build artifacts so
# the working tree is left clean. All output goes to ~/temseg_nightly/ and only
# the 10 most recent logs are kept.
#
# Intended to be run from cron, e.g.:
#   0 3 * * * /Users/0saker/Desktop/code/tem-particle-seg/scripts/mac/nightly_test.sh --quick
#
# Usage:
#   ./scripts/mac/nightly_test.sh               # quick API suite
#   ./scripts/mac/nightly_test.sh --full        # full API suite
#   ./scripts/mac/nightly_test.sh --full --all-models

set -uo pipefail

# Cron runs with a minimal PATH; make sure git/curl/python3 etc. are reachable.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

TIER="tier2"
DIST_DIR="dist/$TIER"
WORK_DIR="build/$TIER"

LOG_DIR="$HOME/temseg_nightly"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly_$(date +%Y%m%d_%H%M%S).log"

tier2_build_ok=0
tier2_test_ok=0
coreml_build_ok=0
coreml_test_ok=0
quick_mode=false
full_mode=false
all_models=false

for arg in "$@"; do
    case "$arg" in
        --quick) quick_mode=true ;;
        --full) full_mode=true ;;
        --all-models) all_models=true ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$quick_mode" = true ] && [ "$full_mode" = true ]; then
    echo "Choose either --quick or --full, not both." >&2
    exit 2
fi

if [ "$quick_mode" = false ] && [ "$full_mode" = false ]; then
    if [ "$all_models" = true ]; then
        full_mode=true
    else
        quick_mode=true
    fi
fi

PYTEST="$REPO_ROOT/.venv/bin/pytest"
if [ ! -x "$PYTEST" ]; then
    echo "pytest is missing at $PYTEST. Install the project dev dependencies first." >&2
    exit 2
fi

SAFE_BRANCH=$(printf '%s' "$(git rev-parse --abbrev-ref HEAD)" | sed 's/[^a-zA-Z0-9]/-/g; s/^-//; s/-$//')
TIER2_APP="dist/tier2/TEMseg-${SAFE_BRANCH}.app"
COREML_APP="dist/coreml/TEMseg-coreml-${SAFE_BRANCH}.app"
PYTEST_MODE=()
if [ "$full_mode" = true ]; then
    PYTEST_MODE+=(--full)
else
    PYTEST_MODE+=(--quick)
fi
if [ "$all_models" = true ]; then
    PYTEST_MODE+=(--all-models)
fi

cleanup_build() {
    # Preserve the build-timings CSVs alongside the logs, then drop the
    # temporary build artifacts (the .apps and PyInstaller work dirs).
    for t in "$WORK_DIR" "build/coreml"; do
        if [ -f "$t/build_timings.csv" ]; then
            cp "$t/build_timings.csv" \
                "$LOG_DIR/build_timings_$(date +%Y%m%d_%H%M%S)_$(basename "$t").csv" 2>/dev/null || true
        fi
    done
    rm -rf "$DIST_DIR" "$WORK_DIR" "dist/coreml" "build/coreml"
}
trap cleanup_build EXIT

{
    echo "=== TEMseg nightly $(date) ==="
    echo "repo : $REPO_ROOT"
    echo "git  : $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
    echo ""

    echo "=== [1/4] build tier2 ONNX (clean, skip frontend) ==="
    if ./scripts/mac/tier2/build_mac.sh --clean --skip-frontend; then
        tier2_build_ok=1
        echo ""
        echo "=== [2/4] packaged app pytest (tier2 ONNX) ==="
        if "$PYTEST" -q -s tests/mac_app/test_packaged_app.py \
            --app "$TIER2_APP" --expected-engine onnx "${PYTEST_MODE[@]}"; then
            tier2_test_ok=1
        fi
    fi

    echo ""
    echo "=== [3/4] build coreml variant (skip frontend) ==="
    if ./scripts/mac/coreml/build_mac.sh --skip-frontend; then
        coreml_build_ok=1
        if [ -d "$COREML_APP" ]; then
            echo ""
            echo "=== [4/4] packaged app pytest (Core ML) ==="
            if "$PYTEST" -q -s tests/mac_app/test_packaged_app.py \
                --app "$COREML_APP" --expected-engine coreml "${PYTEST_MODE[@]}"; then
                coreml_test_ok=1
            fi
        else
            echo "ERROR: coreml build produced no $COREML_APP"
            coreml_build_ok=0
        fi
    fi
} > "$LOG" 2>&1

# prune old logs and timing CSVs (keep 10 of each)
ls -1t "$LOG_DIR"/nightly_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f 2>/dev/null || true
ls -1t "$LOG_DIR"/build_timings_*.csv 2>/dev/null | tail -n +11 | xargs -r rm -f 2>/dev/null || true

echo "Nightly finished (tier2_build=$tier2_build_ok tier2_test=$tier2_test_ok coreml_build=$coreml_build_ok coreml_test=$coreml_test_ok) — log: $LOG"
if [ "$tier2_build_ok" = 1 ] && [ "$tier2_test_ok" = 1 ] \
    && [ "$coreml_build_ok" = 1 ] && [ "$coreml_test_ok" = 1 ]; then
    exit 0
else
    exit 1
fi
