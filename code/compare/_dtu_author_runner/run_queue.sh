#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?GPU id}
QUEUE=${2:?task list}
[[ "$GPU" != 6 ]] || { echo 'GPU 6 is prohibited' >&2; exit 2; }
LOG=/tmp/lilinwei/1007fsgsmvs_dtu/official_random_6k/logs/worker_gpu_${GPU}.log
while read -r METHOD SCAN; do
  [[ -n "${METHOD:-}" ]] || continue
  echo "START $METHOD $SCAN GPU$GPU $(date -Is)" | tee -a "$LOG"
  /data/lilinwei/3D/1006difix/1007fsgsmvs/dtu/output/official_random_6k/code/run_one_official.sh \
    "$METHOD" "$SCAN" "$GPU" >> "$LOG" 2>&1
  echo "FINISH $METHOD $SCAN GPU$GPU $(date -Is)" | tee -a "$LOG"
done < "$QUEUE"
