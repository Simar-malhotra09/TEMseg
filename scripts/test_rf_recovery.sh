#!/usr/bin/env bash
# Test the full RF-recovery pipeline against a running TEMseg server.
# Usage: bash scripts/test_rf_recovery.sh [IMAGE_PATH] [PORT]
# Defaults: image = temseg_export_failing_obj/original_image.png, port = 8000

set -uo pipefail   # no -e: we handle errors manually so output is always visible

IMAGE="${1:-temseg_export_failing_obj/original_image.png}"
PORT="${2:-8080}"
BASE="http://localhost:${PORT}"

if [[ ! -f "$IMAGE" ]]; then
  echo "[ERR] Image not found: $IMAGE"
  exit 1
fi

echo "=== TEMseg RF Recovery Test ==="
echo "Image : $IMAGE"
echo "Server: $BASE"
echo ""

# Helper: curl that always shows the response body, even on HTTP errors.
# Prints HTTP status on its own line after the body.
req() {
  local method="$1"; shift
  local url="$1";    shift
  local http_code body
  # capture body + http code; -w appends the status after body
  body=$(curl -s -w "\n[HTTP %{http_code}]" -X "$method" "$url" "$@" 2>&1)
  echo "$body"
}

# ── 0. Health check ──────────────────────────────────────────────────────────
echo "--- 0. Health check ---"
HEALTH=$(curl -s --connect-timeout 3 "$BASE/" 2>&1)
if [[ $? -ne 0 ]]; then
  echo "[ERR] Could not reach $BASE — is the server running? (python start.py)"
  echo "curl output: $HEALTH"
  exit 1
fi
echo "$HEALTH"
echo ""

# ── 1. Upload ────────────────────────────────────────────────────────────────
echo "--- 1. Upload image ---"
UPLOAD=$(curl -s -w "\n[HTTP %{http_code}]" -X POST "$BASE/images/upload" \
  -F "file=@${IMAGE}" \
  -H "Accept: application/json")
echo "$UPLOAD"

SESSION=$(echo "$UPLOAD" | python3 -c "
import sys, json
lines = sys.stdin.read()
body = ''.join(l for l in lines.splitlines() if not l.startswith('[HTTP'))
print(json.loads(body)['session_id'])
" 2>&1)

if [[ $? -ne 0 || -z "$SESSION" ]]; then
  echo "[ERR] Failed to parse session_id from upload response"
  exit 1
fi

echo ""
echo "Session ID: $SESSION"
echo ""

# ── 2. Segment ───────────────────────────────────────────────────────────────
echo "--- 2. Segment (YOLO+SAM + RF recovery) ---"
echo "(this may take 30-90s on first run)"
SEG=$(curl -s -w "\n[HTTP %{http_code}]" -X POST "$BASE/segment/" \
  -H "Content-Type: application/json" \
  --max-time 180 \
  -d "{
    \"session_id\": \"$SESSION\",
    \"model\": \"yolosam\",
    \"regions\": [],
    \"blackout\": false,
    \"inverse_blackout\": false,
    \"colorize\": true
  }")
echo "$SEG"

SEG_BODY=$(echo "$SEG" | grep -v '^\[HTTP')
DETECTIONS=$(echo "$SEG_BODY" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('metadata',{}).get('detections','?'))" 2>/dev/null || echo "?")
T1=$(echo "$SEG_BODY" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('time_elapsed','?'))" 2>/dev/null || echo "?")
echo ""
echo "Detections: $DETECTIONS  |  Time: ${T1}s"
echo ""

# ── 3. Manual RF train ───────────────────────────────────────────────────────
echo "--- 3. POST /rf/train (re-train on session) ---"
TRAIN=$(curl -s -w "\n[HTTP %{http_code}]" -X POST "$BASE/rf/train" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d "{\"session_id\": \"$SESSION\", \"min_area\": 50}")
echo "$TRAIN"
echo ""

# ── 4. Second segment — RF cache warm ────────────────────────────────────────
echo "--- 4. Segment again (RF cache warm) ---"
SEG2=$(curl -s -w "\n[HTTP %{http_code}]" -X POST "$BASE/segment/" \
  -H "Content-Type: application/json" \
  --max-time 180 \
  -d "{
    \"session_id\": \"$SESSION\",
    \"model\": \"yolosam\",
    \"regions\": [],
    \"blackout\": false,
    \"inverse_blackout\": false,
    \"colorize\": true
  }")
echo "$SEG2"

SEG2_BODY=$(echo "$SEG2" | grep -v '^\[HTTP')
T2=$(echo "$SEG2_BODY" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('time_elapsed','?'))" 2>/dev/null || echo "?")
echo ""
echo "Time run 1: ${T1}s  |  Time run 2: ${T2}s"
echo ""

# ── 5. Evict cache ────────────────────────────────────────────────────────────
echo "--- 5. DELETE /rf/cache/$SESSION ---"
EVICT=$(curl -s -w "\n[HTTP %{http_code}]" -X DELETE "$BASE/rf/cache/$SESSION")
echo "$EVICT"
echo ""

echo "=== Done. Check server logs for [SEG-RF] lines ==="
