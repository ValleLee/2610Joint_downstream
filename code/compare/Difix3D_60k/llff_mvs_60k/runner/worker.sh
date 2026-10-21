#!/usr/bin/env bash
set -Eeuo pipefail
GPU_ID="${1:?usage: worker.sh <physical_gpu_id>}"
case "$GPU_ID" in 0|1|2|3|4|5|7) ;; *) echo "GPU $GPU_ID is not allowed" >&2; exit 2;; esac
METHOD_ROOT=/data/lilinwei/3D/baselines/output/1007_fsgs_mvs_baselines/Difix3D_60k
ROOT=/tmp/lilinwei/1007baselines
RUN_ID=seed2024_fsgs_official_mvs_Difix3D_60k
QUEUE="$ROOT/queue/${RUN_ID}.tsv"
LOCK="$ROOT/queue/${RUN_ID}.lock"
STATUS_DIR="$ROOT/status/$RUN_ID"
RUNNER="$METHOD_ROOT/configs/run_scene.sh"
mkdir -p "$(dirname "$QUEUE")" "$STATUS_DIR" "$METHOD_ROOT/logs"

claim_task() {
  local scene=""
  exec 9>"$LOCK"
  flock -x 9
  if [ -s "$QUEUE" ]; then
    IFS= read -r scene < "$QUEUE" || true
    local tmp="$QUEUE.tmp.$$"
    tail -n +2 "$QUEUE" > "$tmp" || true
    mv "$tmp" "$QUEUE"
  fi
  flock -u 9
  printf '%s' "$scene"
}

printf 'WORKER_START gpu=%s time=%s\n' "$GPU_ID" "$(date -Is)"
while :; do
  SCENE="$(claim_task)"
  [ -n "$SCENE" ] || break
  STATUS="$STATUS_DIR/$SCENE.status"
  if [ -f "$STATUS" ] && grep -qx 'SUCCESS' "$STATUS"; then
    printf 'SKIP_ALREADY_SUCCESS gpu=%s scene=%s\n' "$GPU_ID" "$SCENE"
    continue
  fi
  printf 'WORKER_TASK gpu=%s scene=%s time=%s\n' "$GPU_ID" "$SCENE" "$(date -Is)"
  if "$RUNNER" "$GPU_ID" "$SCENE" run; then
    printf 'WORKER_TASK_SUCCESS gpu=%s scene=%s time=%s\n' "$GPU_ID" "$SCENE" "$(date -Is)"
  else
    rc=$?
    printf 'WORKER_TASK_FAILED gpu=%s scene=%s rc=%s time=%s\n' "$GPU_ID" "$SCENE" "$rc" "$(date -Is)"
  fi
done
printf 'WORKER_DONE gpu=%s time=%s\n' "$GPU_ID" "$(date -Is)"
