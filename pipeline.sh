#!/usr/bin/env bash
# pipeline.sh — calls seg_server.py endpoints in order:
#   segment → classify → mesh → infer_joints → rig
#
# Usage:
#   ./pipeline.sh <image_path> [tag] [base_url] [user_id]
#
# Examples:
#   ./pipeline.sh my_image.png "dragon"
#   ./pipeline.sh my_image.png "car" "http://192.168.1.10:6000"

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
IMAGE_PATH="${1:-}"
TAG="${2:-}"
BASE_URL="${3:-http://localhost:6000}"
USER_ID="${4:-fb712dd7-73cc-43a5-8158-74f7cb8a7fb4}"

# Root results dir — must match RESULTS_DIR in seg_server.py
RESULTS_DIR="${RESULTS_DIR:-results}"

if [[ -z "$IMAGE_PATH" ]]; then
  echo "Usage: $0 <image_path> [tag] [base_url] [user_id]"
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Error: file not found: $IMAGE_PATH"
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BOLD}[pipeline]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
fail() { echo -e "${RED}  ✗ $*${NC}"; exit 1; }

jq_or_raw() {
  if command -v jq &>/dev/null; then
    echo "$1" | jq .
  else
    echo "$1"
  fi
}

get_field() {
  # $1 = JSON string, $2 = field name
  if command -v jq &>/dev/null; then
    echo "$1" | jq -r ".$2 // empty"
  else
    echo "$1" | grep -o "\"$2\":\"[^\"]*\"" | head -1 | sed 's/.*":"//' | sed 's/"//'
  fi
}

poll_status() {
  # Poll a status endpoint until status is "ok" or "error"
  # $1 = URL, $2 = friendly name, $3 = max wait seconds
  local url="$1" name="$2" max_wait="${3:-300}"
  local elapsed=0 interval=5

  log "Polling $name status..."
  while true; do
    local resp
    resp=$(curl -sf "$url" || echo '{"status":"error","error":"curl failed"}')
    local status
    status=$(get_field "$resp" "status")

    case "$status" in
      ok)
        ok "$name complete"
        echo "$resp"
        return 0
        ;;
      error)
        local err
        err=$(get_field "$resp" "error")
        fail "$name failed: $err"
        ;;
      *)
        local progress
        progress=$(get_field "$resp" "progress")
        echo "  … $name: $status ${progress:+(${progress}%)}"
        ;;
    esac

    if (( elapsed >= max_wait )); then
      fail "$name timed out after ${max_wait}s"
    fi
    sleep "$interval"
    (( elapsed += interval ))
  done
}

# ── Step 1: /segment ──────────────────────────────────────────────────────────
log "Step 1/5 — /segment (removing background)"

HTTP_CODE=$(curl -s -X POST \
  --data-binary "@${IMAGE_PATH}" \
  -H "Content-Type: application/octet-stream" \
  --output /tmp/pipeline_segmented.png \
  -w "%{http_code}" \
  "${BASE_URL}/segment")

if [[ "$HTTP_CODE" != "200" ]]; then
  fail "/segment returned HTTP $HTTP_CODE"
fi

if [[ ! -s /tmp/pipeline_segmented.png ]]; then
  fail "/segment returned empty response"
fi

ok "Segmented image saved to /tmp/pipeline_segmented.png"

# ── Step 2: /classify ─────────────────────────────────────────────────────────
log "Step 2/5 — /classify (identifying object)"

TAG_PARAM=""
[[ -n "$TAG" ]] && TAG_PARAM="&tag=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$TAG")"

CLASSIFY_RESP=$(curl -sf -X POST \
  --data-binary "@/tmp/pipeline_segmented.png" \
  -H "Content-Type: application/octet-stream" \
  "${BASE_URL}/classify?user_id=${USER_ID}${TAG_PARAM}")

if [[ -z "$CLASSIFY_RESP" ]]; then
  fail "/classify returned empty response"
fi

CLASSIFY_ID=$(get_field "$CLASSIFY_RESP" "classify_id")
OBJECT_TYPE=$(get_field "$CLASSIFY_RESP" "object_type")
CATEGORY=$(get_field "$CLASSIFY_RESP" "category")
NEEDS_AUG=$(get_field "$CLASSIFY_RESP" "needs_augmentation")

if [[ -z "$CLASSIFY_ID" ]]; then
  echo "Raw response:"
  jq_or_raw "$CLASSIFY_RESP"
  fail "Could not extract classify_id from /classify response"
fi

ok "classify_id:   $CLASSIFY_ID"
ok "object_type:   $OBJECT_TYPE"
ok "category:      $CATEGORY"
ok "needs_augment: $NEEDS_AUG"

# Per-classify_id subdirectory (matches seg_server's _rdir())
CLASSIFY_DIR="${RESULTS_DIR}/${CLASSIFY_ID}"

# ── Step 3: /mesh ─────────────────────────────────────────────────────────────
log "Step 3/5 — /mesh (generating 3D mesh via Meshy)"

# New subdir layout: results/{classify_id}/mesh.glb
LOCAL_GLB="${CLASSIFY_DIR}/mesh.glb"

if [[ -f "$LOCAL_GLB" ]]; then
  ok "Mesh already exists locally, skipping API call: $LOCAL_GLB"
  GLB_LOCAL_URL="${BASE_URL}/results/${CLASSIFY_ID}/mesh.glb"
  GLB_URL=""
else
  MESH_RESP=$(curl -sf -X POST \
    --data-binary "@/tmp/pipeline_segmented.png" \
    -H "Content-Type: application/octet-stream" \
    "${BASE_URL}/mesh?classify_id=${CLASSIFY_ID}")

  MESH_STATUS=$(get_field "$MESH_RESP" "status")
  MESH_TASK_ID=$(get_field "$MESH_RESP" "task_id")
  GLB_URL=$(get_field "$MESH_RESP" "glb_url")
  GLB_LOCAL_URL=$(get_field "$MESH_RESP" "glb_local_url")

  if [[ "$MESH_STATUS" == "ok" ]]; then
    ok "Mesh already cached on server"
    ok "GLB URL: ${GLB_LOCAL_URL:-$GLB_URL}"
  elif [[ "$MESH_STATUS" == "processing" && -n "$MESH_TASK_ID" ]]; then
    ok "Mesh task started: $MESH_TASK_ID"
    MESH_STATUS_RESP=$(poll_status \
      "${BASE_URL}/mesh/status/${MESH_TASK_ID}" \
      "mesh" \
      300)
    GLB_URL=$(get_field "$MESH_STATUS_RESP" "glb_url")
    GLB_LOCAL_URL=$(get_field "$MESH_STATUS_RESP" "glb_local_url")
    ok "GLB URL: ${GLB_LOCAL_URL:-$GLB_URL}"
  else
    echo "Raw /mesh response:"
    jq_or_raw "$MESH_RESP"
    fail "/mesh failed (status=$MESH_STATUS)"
  fi
fi

# ── Step 4: /infer_joints ─────────────────────────────────────────────────────
log "Step 4/5 — /infer_joints (placing skeleton joints)"

JOINTS_RESP=$(curl -sf -X POST \
  --data-binary "@/tmp/pipeline_segmented.png" \
  -H "Content-Type: application/octet-stream" \
  "${BASE_URL}/infer_joints?classify_id=${CLASSIFY_ID}")

if [[ -z "$JOINTS_RESP" ]]; then
  fail "/infer_joints returned empty response"
fi

JOINTS_COUNT=$(echo "$JOINTS_RESP" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('joint_hints',[])) )" 2>/dev/null || echo "?")

MODEL_USED=$(get_field "$JOINTS_RESP" "model_used")

ok "Joints placed: $JOINTS_COUNT joint hints (model: $MODEL_USED)"

# ── Step 5: /rig ──────────────────────────────────────────────────────────────
log "Step 5/5 — /rig (rigging mesh with skeleton)"

RIG_RESP=$(curl -sf -X POST \
  "${BASE_URL}/rig?classify_id=${CLASSIFY_ID}&user_id=${USER_ID}")

RIG_STATUS=$(get_field "$RIG_RESP" "status")
RIG_TASK_ID=$(get_field "$RIG_RESP" "task_id")
RIGGED_URL=$(get_field "$RIG_RESP" "rigged_url")

if [[ "$RIG_STATUS" == "ok" && -n "$RIGGED_URL" ]]; then
  ok "Rig already cached: $RIGGED_URL"
elif [[ "$RIG_STATUS" == "processing" && -n "$RIG_TASK_ID" ]]; then
  ok "Rig task started: $RIG_TASK_ID"
  RIG_STATUS_RESP=$(poll_status \
    "${BASE_URL}/rig/status/${RIG_TASK_ID}" \
    "rig" \
    300)
  RIGGED_URL=$(get_field "$RIG_STATUS_RESP" "rigged_url")
  ok "Rigged GLB: $RIGGED_URL"
else
  echo "Raw /rig response:"
  jq_or_raw "$RIG_RESP"
  fail "/rig failed (status=$RIG_STATUS)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Pipeline complete!${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  classify_id  : $CLASSIFY_ID"
echo "  results dir  : ${CLASSIFY_DIR}"
echo "  object_type  : $OBJECT_TYPE"
echo "  category     : $CATEGORY"
echo "  joints       : $JOINTS_COUNT (via $MODEL_USED)"
echo "  rigged GLB   : $RIGGED_URL"
echo ""
