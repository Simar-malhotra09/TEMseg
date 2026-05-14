#!/usr/bin/env bash
set -euo pipefail

APP="dist/TEMseg.app/Contents/MacOS/TEMseg"
BASE="http://localhost:8080"
DATA="data"

# Colors
R='\033[0;31m'; G='\033[0;32m'; N='\033[0m'
ok()  { echo -e "${G}✓${N} $1"; }
fail() { echo -e "${R}✗${N} $1" >&2; exit 1; }

# 1. Launch app
if [[ ! -x "$APP" ]]; then fail "Binary not found: $APP"; fi
echo "Launching $APP..."
"$APP" > /tmp/temseg_test.log 2>&1 &
PID=$!
sleep 35  # wait for weight check + model init (yolo_sam can take 20-30s)

# 2. Health check
curl -sf "$BASE/" > /dev/null || fail "Backend not responding"
ok "SUCCESS: Backend up"

# 3. List models
curl -sf "$BASE/models" > /dev/null || fail "Models endpoint failed"
ok "SUCCESS: GET /models"

# 4. Upload test images
upload() {
  local f=$1
  curl -sf --max-time 60 -F "file=@$f" "$BASE/images/upload" || fail "Upload failed: $f"
}

SESSION1=$(upload "$DATA/batch5/0031.emd" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
ok "SUCCESS: Upload batch5/0031.emd → $SESSION1"

SESSION2=$(upload "$DATA/failing-mul-objs/2-2-DA3-96k-df-0.5.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
ok "SUCCESS: Upload failing-mul-objs image → $SESSION2"

# 5. Segment (full frame, no regions) — 120s timeout per image
segment() {
  local sid=$1
  curl -sf --max-time 120 -X POST "$BASE/segment/" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$sid\",\"model\":\"YoloSAM\",\"blackout\":false,\"inverse_blackout\":false,\"regions\":[],\"colorize\":true}" \
    || fail "Segment failed: $sid"
}

segment "$SESSION1" > /dev/null
ok "SUCCESS: Segment $SESSION1 (yolo_sam)"

segment "$SESSION2" > /dev/null
ok "SUCCESS: Segment $SESSION2 (yolo_sam)"

# 6. Get instances
curl -sf -X POST "$BASE/masks/$SESSION1/instances" > /dev/null || fail "Get instances failed"
ok "SUCCESS: GET instances $SESSION1"

# 7. Export
curl -sf -X POST "$BASE/export/$SESSION1" \
  -H "Content-Type: application/json" \
  -d '{"items":["seg_mask_png","instances_json","stats_csv"]}' \
  > /tmp/temseg_export.zip || fail "Export failed"
ok "SUCCESS: Export $SESSION1"

# 8. Cleanup
kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
rm -f /tmp/temseg_export.zip
ok "All tests passed — no crashes"
