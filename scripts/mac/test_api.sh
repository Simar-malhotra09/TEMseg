#!/usr/bin/env bash
# test_api.sh - TEMseg backend API smoke/integration test (no GUI needed)
#
# Launches a built TEMseg .app, drives the FastAPI backend over HTTP, verifies
# responses, then cleans up. Designed to be run manually or nightly in CI to
# catch early regressions (imports, bundling, model loading, API contract).
#
# This does NOT replace a visual check - it cannot judge mask QUALITY, the
# pywebview window, or the JS export bridge. It only verifies the backend and
# static serving behave.
#
# Usage:
#   ./scripts/mac/test_api.sh                  # auto-detect app, full suite
#   ./scripts/mac/test_api.sh --app dist/tier2/TEMseg-main.app
#   ./scripts/mac/test_api.sh --quick          # fast core path only
#   ./scripts/mac/test_api.sh --all-models     # also run YoloMaskRCNN + MaskRCNN-Synthetic
#   ./scripts/mac/test_api.sh --gui            # launch with the real GUI (default: headless)
#   ./scripts/mac/test_api.sh --keep-running   # don't kill app when done
#   ./scripts/mac/test_api.sh --strict         # fail on empty segmentation
#
# Env:
#   BASE_URL      backend base URL (default http://localhost:8080)
#   FRONTEND_URL  frontend base URL (default http://localhost:3001)
#   TIMEOUT       per-request curl timeout in seconds (default 180)
#   STARTUP_WAIT  max seconds to wait for backend (default 180)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL="${BASE_URL:-http://localhost:8080}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3001}"
TIMEOUT="${TIMEOUT:-180}"
STARTUP_WAIT="${STARTUP_WAIT:-180}"

APP=""
QUICK=false
STRICT=false
KEEP_RUNNING=false
ALL_MODELS=false
GUI=false

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -40
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --app)          APP="$2"; shift 2 ;;
        --quick)        QUICK=true; shift ;;
        --strict)       STRICT=true; shift ;;
        --all-models)   ALL_MODELS=true; shift ;;
        --gui)          GUI=true; shift ;;
        --keep-running) KEEP_RUNNING=true; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Test data fixtures (co-located with this script so the test is self-contained;
# these are NOT bundled into the app - the spec only packages backend/src,
# frontend/out, launcher.py, weight_manifest.json).
# ---------------------------------------------------------------------------
FIXTURES="$SCRIPT_DIR/fixtures"
EMD="${EMD:-$FIXTURES/sample.emd}"      # rsciio/hyperspy path, known-good for segmentation
TIF="${TIF:-$FIXTURES/sample.tif}"      # tifffile path
JPG="${JPG:-$FIXTURES/sample.jpg}"      # cv2 path
NPY="${NPY:-$FIXTURES/mask.npy}"        # numpy path (also used as the GT mask)
GT_MASK="${GT_MASK:-$FIXTURES/mask.npy}"

# Fail early if a fixture is missing (keeps errors clear and actionable).
for _f in "$EMD" "$TIF" "$JPG" "$NPY" "$GT_MASK"; do
    [ -f "$_f" ] || { echo "Missing fixture: $_f"; exit 2; }
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
SKIP=0
FAILED_TESTS=()
SESSION_IDS=()
LOGDIR="$(mktemp -d /tmp/temseg_api_test.XXXXXX)"
APP_PID=""

R='\033[0;31m'; G='\033[0;32m'; Y='\033[0;33m'; N='\033[0m'
pass() { PASS=$((PASS+1)); echo -e "  ${G}✓${N} $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${R}✗${N} $1"; FAILED_TESTS+=("$1"); }
skip() { SKIP=$((SKIP+1)); echo -e "  ${Y}−${N} $1 (skipped)"; }
section() { echo ""; echo "── $1 ──"; }

# curl wrapper: sets global HTTP_CODE and HTTP_BODY.
#   req METHOD URL [extra curl args...]
req() {
    local method="$1" url="$2"; shift 2
    local out
    out=$(curl -sS --max-time "$TIMEOUT" -w $'\n__HTTP_STATUS__%{http_code}' \
        -X "$method" "$url" "$@") || { HTTP_CODE="000"; HTTP_BODY="curl failed"; return; }
    HTTP_BODY="${out%$'\n__HTTP_STATUS__'*}"
    HTTP_CODE="${out##*$'\n__HTTP_STATUS__'}"
}

# curl wrapper for binary/streaming responses: writes body to a file instead of
# a shell variable (command substitution would strip null bytes from ZIP/PNG).
#   req_file METHOD URL OUTFILE [extra curl args...]
req_file() {
    local method="$1" url="$2" outfile="$3"; shift 3
    HTTP_CODE=$(curl -sS --max-time "$TIMEOUT" -o "$outfile" -w '%{http_code}' \
        -X "$method" "$url" "$@")
    HTTP_BODY=""
}

# Extract a JSON field using a Python expression. stdin = JSON.
jget() {
    python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null
}

# Assert the last HTTP code equals $1.
assert_code() {
    local want="$1"
    [ "$HTTP_CODE" = "$want" ] || { fail "expected HTTP $want, got $HTTP_CODE"; return 1; }
    return 0
}

# Assert last HTTP code is 200 or 400 (business-level OK), i.e. no 5xx crash.
assert_no_5xx() {
    case "$HTTP_CODE" in
        200|201|400|404|422) return 0 ;;
        *) fail "unexpected HTTP $HTTP_CODE (body: $(echo "$HTTP_BODY" | head -c 300))"; return 1 ;;
    esac
}

cleanup() {
    echo ""
    echo "── cleanup ──"
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        [ "$KEEP_RUNNING" = true ] && echo "  leaving app running (pid $APP_PID)" \
            || { echo "  stopping app (pid $APP_PID)"; kill "$APP_PID" 2>/dev/null || true; wait "$APP_PID" 2>/dev/null || true; }
    fi
    # Remove only the sessions this run created (sessions live inside the bundle
    # under backend_src/sessions in frozen mode).
    if [ -n "${APP:-}" ] && [ ${#SESSION_IDS[@]} -gt 0 ]; then
        local sess_root="${APP}/Contents/Resources/backend_src/sessions"
        for sid in "${SESSION_IDS[@]}"; do
            rm -rf "$sess_root/$sid" 2>/dev/null || true
        done
        echo "  removed ${#SESSION_IDS[@]} test session(s)"
    fi
    echo "  logs: $LOGDIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Launch app
# ---------------------------------------------------------------------------
shopt -s nullglob
if [ -z "$APP" ]; then
    for p in dist/tier2/TEMseg-*.app dist/tier1/TEMseg-*.app dist/coreml/TEMseg-coreml-*.app dist/TEMseg.app dist/TEMseg-*.app; do
        [ -d "$p" ] && APP="$p" && break
    done
fi
[ -n "$APP" ] || { echo "No .app found. Build first or pass --app."; exit 2; }
APP="$(cd "$(dirname "$APP")" && pwd)/$(basename "$APP")"

APP_BIN="$APP/Contents/MacOS/TEMseg"
if [ ! -x "$APP_BIN" ]; then
    # coreml variant names its binary after the variant
    APP_BIN="$(find "$APP/Contents/MacOS" -maxdepth 1 -type f -perm -111 | head -1)"
fi
[ -x "$APP_BIN" ] || { echo "Binary not found: $APP_BIN"; exit 2; }

echo "========================================"
echo " TEMseg API test"
echo "  app      : $APP"
echo "  backend  : $BASE_URL"
echo "  frontend : $FRONTEND_URL"
echo "  mode     : $([ "$QUICK" = true ] && echo quick || echo full)"
echo "========================================"

if [ "$GUI" = true ]; then
    echo "Launching $APP_BIN (GUI mode)..."
    "$APP_BIN" > "$LOGDIR/app.log" 2>&1 &
else
    echo "Launching $APP_BIN (headless, TEMSEG_HEADLESS=1)..."
    TEMSEG_HEADLESS=1 "$APP_BIN" > "$LOGDIR/app.log" 2>&1 &
fi
APP_PID=$!

echo "Waiting for backend (max ${STARTUP_WAIT}s)..."
UP=false
for _ in $(seq 1 $((STARTUP_WAIT / 2))); do
    sleep 2
    if curl -sf --max-time 2 "$BASE_URL/" >/dev/null 2>&1; then UP=true; break; fi
    kill -0 "$APP_PID" 2>/dev/null || { echo "App died during startup - tail of log:"; tail -40 "$LOGDIR/app.log"; exit 1; }
done
[ "$UP" = true ] || { echo "Backend never came up - tail of log:"; tail -60 "$LOGDIR/app.log"; exit 1; }
echo "Backend is up."

# ===========================================================================
section "1. Health + models"
# ===========================================================================
req GET "$BASE_URL/"
assert_code 200 && pass "GET / returns 200" || true

req GET "$BASE_URL/models"
if assert_code 200; then
    MODELS="$(echo "$HTTP_BODY" | jget 'd')"
    for m in YoloSAM YoloMaskRCNN MaskRCNN-Synthetic; do
        echo "$HTTP_BODY" | grep -q "$m" && pass "model present: $m" || fail "model missing: $m"
    done
else
    :
fi

# ===========================================================================
section "2. Uploads (all format branches)"
# ===========================================================================
upload() {
    local label="$1" file="$2"
    req POST "$BASE_URL/images/upload" -F "file=@$file"
    if assert_code 200; then
        local sid
        sid="$(echo "$HTTP_BODY" | jget 'd["session_id"]')"
        if [ -n "$sid" ] && [ "$sid" != "None" ]; then
            SESSION_IDS+=("$sid")
            pass "$label upload → session $sid"
            UPLOAD_SID="$sid"
        else
            fail "$label upload: no session_id in response"
        fi
    fi
    echo "$sid"
}

# EMD (primary - used for segmentation later)
req POST "$BASE_URL/images/upload" -F "file=@$EMD"
if assert_code 200; then
    EMD_SID="$(echo "$HTTP_BODY" | jget 'd["session_id"]')"
    [ -n "$EMD_SID" ] && SESSION_IDS+=("$EMD_SID")
    FMT="$(echo "$HTTP_BODY" | jget 'd["image_info"]["original_format"]')"
    PIX="$(echo "$HTTP_BODY" | jget 'd["image_info"]["pixel_size"]')"
    pass "EMD upload → session $EMD_SID (format=$FMT, pixel_size=$PIX)"
fi

# TIFF
req POST "$BASE_URL/images/upload" -F "file=@$TIF"
if assert_code 200; then
    TIF_SID="$(echo "$HTTP_BODY" | jget 'd["session_id"]')"
    [ -n "$TIF_SID" ] && SESSION_IDS+=("$TIF_SID")
    SHAPE="$(echo "$HTTP_BODY" | jget 'd["image_info"]["image_shape"]')"
    pass "TIFF upload → session $TIF_SID (shape=$SHAPE)"
fi

# JPG
req POST "$BASE_URL/images/upload" -F "file=@$JPG"
if assert_code 200; then
    JPG_SID="$(echo "$HTTP_BODY" | jget 'd["session_id"]')"
    [ -n "$JPG_SID" ] && SESSION_IDS+=("$JPG_SID")
    pass "JPG upload → session $JPG_SID"
fi

# NPY
req POST "$BASE_URL/images/upload" -F "file=@$NPY"
if assert_code 200; then
    NPY_SID="$(echo "$HTTP_BODY" | jget 'd["session_id"]')"
    [ -n "$NPY_SID" ] && SESSION_IDS+=("$NPY_SID")
    pass "NPY upload → session $NPY_SID"
fi

# Metadata / preview / file on the EMD session
if [ -n "${EMD_SID:-}" ]; then
    req GET "$BASE_URL/images/$EMD_SID/metadata"
    assert_code 200 && pass "GET metadata for EMD session" || true

    req_file GET "$BASE_URL/images/$EMD_SID/preview" /dev/null
    assert_code 200 && pass "GET preview (original_preview.png)" || true

    req_file GET "$BASE_URL/images/$EMD_SID/file" /dev/null
    assert_code 200 && pass "GET original file" || true
fi

# ===========================================================================
section "3. Segmentation (YoloSAM)"
# ===========================================================================
MASK_URL=""
if [ -n "${EMD_SID:-}" ]; then
    req POST "$BASE_URL/segment/" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"$EMD_SID\",\"model\":\"YoloSAM\",\"blackout\":false,\"inverse_blackout\":false,\"regions\":[],\"colorize\":true}"
    if assert_code 200; then
        MASK_URL="$(echo "$HTTP_BODY" | jget 'd.get("mask_url")')"
        STATS_EMPTY="$(echo "$HTTP_BODY" | jget 'str(d.get("stats") == {})')"
        pass "segment YoloSAM ($MASK_URL)"
        if [ "$MASK_URL" = "None" ] || [ "$MASK_URL" = "" ]; then
            if [ "$STRICT" = true ]; then fail "segment returned no mask (no particles)"; else
                echo -e "  ${Y}⚠${N} segment returned no particles for $EMD (possible regression)"; fi
        fi
    fi
fi

if [ -n "$MASK_URL" ] && [ "$MASK_URL" != "None" ]; then
    req_file GET "$BASE_URL$MASK_URL" /dev/null
    assert_code 200 && pass "GET mask.png" || true
fi

# ===========================================================================
section "4. Instances + stats"
# ===========================================================================
INSTANCES_JSON=""
if [ -n "${EMD_SID:-}" ]; then
    req POST "$BASE_URL/masks/$EMD_SID/instances"
    if assert_code 200; then
        INSTANCES_JSON="$HTTP_BODY"
        N_INST="$(echo "$HTTP_BODY" | jget 'len(d["instances"])')"
        pass "GET instances (count=$N_INST)"
    fi

    req GET "$BASE_URL/masks/$EMD_SID/stats"
    assert_code 200 && pass "GET stats.json" || true
fi

# ===========================================================================
section "5. Export"
# ===========================================================================
if [ -n "${EMD_SID:-}" ]; then
    ZIP="$LOGDIR/export.zip"
    req_file POST "$BASE_URL/export/$EMD_SID" "$ZIP" \
        -H "Content-Type: application/json" \
        -d '{"items":["original_image","seg_mask_png","seg_mask_npy","instances_json","stats_csv","coco_json"]}'
    if assert_code 200; then
        if unzip -t "$ZIP" >/dev/null 2>&1; then
            N_ZIP="$(unzip -l "$ZIP" | tail -1 | awk '{print $2}')"
            pass "export ZIP valid ($N_ZIP files)"
        else
            fail "export did not produce a valid ZIP"
        fi
    fi

    req POST "$BASE_URL/export/$EMD_SID/hist_csv?metric=diameter"
    assert_code 200 && pass "export histogram CSV (diameter)" || true
fi

# ===========================================================================
section "6. Frontend static server"
# ===========================================================================
req GET "$FRONTEND_URL/"
assert_code 200 && echo "$HTTP_BODY" | grep -qiE "<html|<div id" && pass "frontend / serves HTML" || fail "frontend / not serving HTML"

req GET "$FRONTEND_URL/workspace"
assert_code 200 && pass "frontend /workspace (SPA fallback)" || true

# ===========================================================================
if [ "$QUICK" = true ]; then
    section "QUICK MODE - skipping full-suite checks"
else
    section "7. Refinement / bootstrap (best-effort, no-5xx)"
    N_INST="${N_INST:-0}"
    if [ -n "$INSTANCES_JSON" ] && [ "$N_INST" -gt 0 ] 2>/dev/null; then
        X0="$(echo "$INSTANCES_JSON" | jget 'd["instances"][0]["bbox"]["x"] + d["instances"][0]["bbox"]["w"] // 2')"
        Y0="$(echo "$INSTANCES_JSON" | jget 'd["instances"][0]["bbox"]["y"] + d["instances"][0]["bbox"]["h"] // 2')"
        ID0="$(echo "$INSTANCES_JSON" | jget 'd["instances"][0]["id"]')"

        req POST "$BASE_URL/masks/$EMD_SID/from-points" \
            -H "Content-Type: application/json" \
            -d "{\"points\":[[$X0,$Y0]],\"pending\":[],\"max_image_fraction\":0.05,\"min_area\":100}"
        assert_no_5xx && pass "from-points (200/400 business OK)" || true

        req POST "$BASE_URL/masks/$EMD_SID/from-boxes" \
            -H "Content-Type: application/json" \
            -d "{\"boxes\":[[$((X0-20)),$((Y0-20)),$((X0+20)),$((Y0+20))]],\"pending\":[],\"max_image_fraction\":0.05,\"min_area\":100}"
        assert_no_5xx && pass "from-boxes (200/400 business OK)" || true

        req POST "$BASE_URL/masks/$EMD_SID/propose-similar" \
            -H "Content-Type: application/json" \
            -d '{"method":"cosine","sim_threshold":0.75,"max_proposals":5,"nms_distance":20}'
        assert_no_5xx && pass "propose-similar (200/400 business OK)" || true

        req POST "$BASE_URL/masks/$EMD_SID/split" \
            -H "Content-Type: application/json" \
            -d "{\"instance_id\":$ID0,\"points\":[[$X0,$Y0]]}"
        assert_no_5xx && pass "split (200/400 business OK)" || true
    else
        skip "refinement endpoints (no instances detected)"
    fi

    # =========================================================================
    if [ "$ALL_MODELS" = true ]; then
        section "8. Other models (lazy-load)"
        for MODEL in YoloMaskRCNN MaskRCNN-Synthetic; do
            req POST "$BASE_URL/segment/" \
                -H "Content-Type: application/json" \
                -d "{\"session_id\":\"$EMD_SID\",\"model\":\"$MODEL\",\"blackout\":false,\"inverse_blackout\":false,\"regions\":[],\"colorize\":false}"
            assert_code 200 && pass "segment $MODEL" || fail "segment $MODEL (HTTP $HTTP_CODE)"
        done
    fi

    # =========================================================================
    section "9. Ground truth"
    req POST "$BASE_URL/gt/$EMD_SID" -F "file=@$GT_MASK"
    assert_code 200 && pass "GT upload" || true

    req POST "$BASE_URL/gt/$EMD_SID/compute" \
        -H "Content-Type: application/json" \
        -d '{"blackout":false,"inverse_blackout":false,"regions":[]}'
    assert_no_5xx && pass "GT compute (200/4xx business OK)" || true

    # =========================================================================
    section "10. RF recovery"
    req POST "$BASE_URL/rf/train" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"$EMD_SID\"}"
    assert_no_5xx && pass "RF train (200/4xx business OK)" || true

    req POST "$BASE_URL/rf/propose" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"$EMD_SID\",\"top_n\":3,\"bg_scribbles\":[{\"points\":[[5,5],[5,20],[20,5]]}]}"
    assert_no_5xx && pass "RF propose (200/4xx business OK)" || true

    # =========================================================================
    section "11. Config"
    req GET "$BASE_URL/config/shape-rules"
    assert_code 200 && pass "GET shape-rules" || true

    req POST "$BASE_URL/config/shape-rules/reset"
    assert_code 200 && pass "reset shape-rules" || true
fi

# ===========================================================================
section "Summary"
# ===========================================================================
echo "  passed : $PASS"
echo "  failed : $FAIL"
echo "  skipped: $SKIP"
if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do echo "  - $t"; done
    echo ""
    echo "App log tail:"
    tail -30 "$LOGDIR/app.log"
    exit 1
fi
echo "All API checks passed."
exit 0
