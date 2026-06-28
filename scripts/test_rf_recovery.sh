#!/usr/bin/env bash
# Test the full RF-recovery pipeline against a running TEMseg server.
# Usage: bash scripts/test_rf_recovery.sh [IMAGE_PATH] [PORT]
# Defaults: image = temseg_export_failing_obj/original_image.png, port = 8000

set -euo pipefail

IMAGE="${1:-temseg_export_failing_obj/original_image.png}"
PORT="${2:-8000}"
BASE="http://localhost:${PORT}"

if [[ ! -f "$IMAGE" ]]; then
  echo "[ERR] Image not found: $IMAGE"
  exit 1
fi

echo "=== TEMseg RF Recovery Test ==="
echo "Image : $IMAGE"
echo "Server: $BASE"
echo ""

# ── 1. Upload ────────────────────────────────────────────────────────────────
echo "--- 1. Upload image ---"
UPLOAD=$(curl -sf -X POST "$BASE/images/upload" \
  -F "file=@${IMAGE}" \
  -H "Accept: application/json")

echo "$UPLOAD" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD"

SESSION=$(echo "$UPLOAD" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo ""
echo "Session ID: $SESSION"
echo ""

# ── 2. Segment ───────────────────────────────────────────────────────────────
echo "--- 2. Segment (YOLO+SAM + RF recovery) ---"
SEG=$(curl -sf -X POST "$BASE/segment/" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"model\": \"yolosam\",
    \"regions\": [],
    \"blackout\": false,
    \"inverse_blackout\": false,
    \"colorize\": true
  }")

echo "$SEG" | python3 -m json.tool 2>/dev/null || echo "$SEG"

DETECTIONS=$(echo "$SEG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metadata',{}).get('detections','?'))" 2>/dev/null || echo "?")
echo ""
echo "Detections: $DETECTIONS"
echo ""

# ── 3. Manual RF train (re-trains and replaces cache) ────────────────────────
echo "--- 3. POST /rf/train (manual re-train on session) ---"
TRAIN=$(curl -sf -X POST "$BASE/rf/train" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"min_area\": 50}")

echo "$TRAIN" | python3 -m json.tool 2>/dev/null || echo "$TRAIN"
echo ""

# ── 4. Second segment — RF cache already warm ────────────────────────────────
echo "--- 4. Segment again (RF cache warm, should be faster) ---"
SEG2=$(curl -sf -X POST "$BASE/segment/" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"model\": \"yolosam\",
    \"regions\": [],
    \"blackout\": false,
    \"inverse_blackout\": false,
    \"colorize\": true
  }")

echo "$SEG2" | python3 -m json.tool 2>/dev/null || echo "$SEG2"

T1=$(echo "$SEG"  | python3 -c "import sys,json; print(json.load(sys.stdin).get('time_elapsed','?'))" 2>/dev/null || echo "?")
T2=$(echo "$SEG2" | python3 -c "import sys,json; print(json.load(sys.stdin).get('time_elapsed','?'))" 2>/dev/null || echo "?")
echo ""
echo "Time run 1: ${T1}s"
echo "Time run 2: ${T2}s"
echo ""

# ── 5. Evict cache ────────────────────────────────────────────────────────────
echo "--- 5. DELETE /rf/cache/$SESSION (evict) ---"
EVICT=$(curl -sf -X DELETE "$BASE/rf/cache/$SESSION")
echo "$EVICT" | python3 -m json.tool 2>/dev/null || echo "$EVICT"
echo ""

echo "=== Done. Check server logs for [SEG-RF] lines ==="
