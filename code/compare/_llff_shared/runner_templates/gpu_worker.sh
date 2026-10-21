#!/usr/bin/env bash
set -Eeuo pipefail
GPU="$1"; ROOT=/tmp/lilinwei/1007baselines; Q="${TASK_QUEUE:-$ROOT/queue/tasks.tsv}"; WORKER_TAG="${WORKER_TAG:-gpu${GPU}}"; FAILED_LOG="${FAILED_TASKS_LOG:-$ROOT/failed_tasks.tsv}"
if [ "$GPU" = 6 ]; then echo 'GPU 6 is forbidden'; exit 2; fi
mkdir -p "$ROOT/logs" "$ROOT/status"
while true; do
  CLAIM=$(python "$ROOT/claim_task.py" --gpu "$GPU" --queue "$Q")
  if [[ "$CLAIM" == DONE ]]; then echo "queue empty at $(date -Is)"; exit 0; fi
  if [[ "$CLAIM" == WAIT* ]]; then sleep 45; continue; fi
  IFS=$'\t' read -r METHOD SCENE FREE MINFREE <<< "$CLAIM"
  echo "CLAIM gpu=$GPU method=$METHOD scene=$SCENE free_mb=$FREE required_mb=$MINFREE at=$(date -Is)" | tee -a "$ROOT/logs/worker_${WORKER_TAG}.log"
  set +e
  case "$METHOD" in
    Difix3D_10k) "$ROOT/run_difix_scene.sh" 10k "$GPU" "$SCENE" ;;
    Difix3D_60k) "$ROOT/run_difix_scene.sh" 60k "$GPU" "$SCENE" ;;
    *) "$ROOT/run_gs_scene.sh" "$METHOD" "$GPU" "$SCENE" ;;
  esac
  RC=$?
  set -e
  echo "RETURN gpu=$GPU method=$METHOD scene=$SCENE rc=$RC at=$(date -Is)" | tee -a "$ROOT/logs/worker_${WORKER_TAG}.log"
  if [ "$RC" -ne 0 ]; then
    echo -e "$METHOD\t$SCENE\t$GPU\t$RC\t$(date -Is)" >> "$FAILED_LOG"
  fi
done
