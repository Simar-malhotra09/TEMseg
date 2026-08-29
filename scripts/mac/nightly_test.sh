#!/usr/bin/env bash
# nightly build + API smoke test for TEMseg.
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

build_ok=0
test_ok=0

cleanup_build() {
    # Preserve the build-timings CSV alongside the logs, then drop the
    # temporary build artifacts (the .app and PyInstaller work dir).
    if [ -f "$WORK_DIR/build_timings.csv" ]; then
        cp "$WORK_DIR/build_timings.csv" \
            "$LOG_DIR/build_timings_$(date +%Y%m%d_%H%M%S).csv" 2>/dev/null || true
    fi
    rm -rf "$DIST_DIR" "$WORK_DIR"
}
trap cleanup_build EXIT

{
    echo "=== TEMseg nightly $(date) ==="
    echo "repo : $REPO_ROOT"
    echo "git  : $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
    echo ""

    echo "=== [1/2] build tier2 (clean, skip frontend) ==="
    if ./scripts/mac/tier2/build_mac.sh --clean --skip-frontend; then
        build_ok=1
    fi

    echo ""
    echo "=== [2/2] API test ==="
    if ./scripts/mac/test_api.sh "$@"; then
        test_ok=1
    fi
} > "$LOG" 2>&1

# prune old logs and timing CSVs (keep 10 of each)
ls -1t "$LOG_DIR"/nightly_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f 2>/dev/null || true
ls -1t "$LOG_DIR"/build_timings_*.csv 2>/dev/null | tail -n +11 | xargs -r rm -f 2>/dev/null || true

echo "Nightly finished (build=$build_ok test=$test_ok) — log: $LOG"
if [ "$build_ok" = 1 ] && [ "$test_ok" = 1 ]; then
    exit 0
else
    exit 1
fi
