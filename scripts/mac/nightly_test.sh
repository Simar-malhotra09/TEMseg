#!/usr/bin/env bash
# nightly_test.sh - nightly build + API smoke test for TEMseg.
#
# Builds the tier-2 .app (clean, reusing the existing frontend export) and runs
# the API test suite against it. All output goes to ~/temseg_nightly/ and only
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

LOG_DIR="$HOME/temseg_nightly"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly_$(date +%Y%m%d_%H%M%S).log"

build_ok=0
test_ok=0

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

# prune old logs (keep 10)
ls -1t "$LOG_DIR"/nightly_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f 2>/dev/null || true

echo "Nightly finished (build=$build_ok test=$test_ok) — log: $LOG"
if [ "$build_ok" = 1 ] && [ "$test_ok" = 1 ]; then
    exit 0
else
    exit 1
fi
