#!/usr/bin/env bash
set -Eeuo pipefail
VARIANT="${1:?10k or 60k required}"
GPU_ID="${2:?gpu required}"
SCENE="${3:?scene required}"
ROOT=/tmp/lilinwei/1007baselines
BASE=/data/lilinwei/3D/baselines/Difix3D
PAIR="$BASE/run/20261007_recheck_pair"
DATA="$ROOT/data/$SCENE"
MVS="$DATA/3_views/dense/fused.ply"
if [ "$VARIANT" = 10k ]; then
  RUN="$PAIR/10k_targetpose"; METHOD=Difix3D_10k; PY=/tmp/lilinwei/Difix3D_Recheck_20261007/env/bin/python
  ADAPTER="$RUN/adapter/simple_trainer_difix3d.py"; MAX=10000; FIRST=5000; INTERVAL=200; SEED=2024; RUN_TAG="${BASELINE_RUN_TAG:-seed2024}"; NORM_ARGS=()
else
  RUN="$PAIR/60k_no_normalize"; METHOD=Difix3D_60k; PY=/tmp/lilinwei/Difix3D_Recheck_20261007/env/bin/python
  ADAPTER="$ROOT/repos/Difix3D_60k_adapter/simple_trainer_difix3d.py"; MAX=60000; FIRST=30000; INTERVAL=2000; SEED=2024; RUN_TAG="${BASELINE_RUN_TAG:-seed2024}"; NORM_ARGS=(--no-normalize-world-space)
fi
SCRATCH="$ROOT/scratch/$RUN_TAG/$METHOD/$SCENE"
ARCHIVE="${BASELINES_OUTPUT_ROOT:-/data/lilinwei/3D/baselines/output/1007_mvs_baselines}/$METHOD"
LOG="$ROOT/logs/${METHOD}_${SCENE}_${RUN_TAG}.log"; STATUS="$ROOT/status/${METHOD}_${SCENE}_${RUN_TAG}.status"
MANIFEST="$RUN/data_mapping/$SCENE.json"
SUPPORT="$PAIR/adapter_support"
FUSED=/data/lilinwei/3D/0918pland/code/third_party/gaussian_splatting_planD/submodules/fused-ssim
mkdir -p "$SCRATCH" "$ARCHIVE/outputs/$SCENE" "$ROOT/logs" "$ROOT/status"
export CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONNOUSERSITE=1 BASELINE_SEED=2024 PYTHONHASHSEED=2024 
export HF_HOME=/tmp/lilinwei/Difix3D_Recheck_20261007/huggingface HF_HUB_CACHE=/tmp/lilinwei/Difix3D_Recheck_20261007/huggingface/hub
export DIFIX_MODEL_DIR="$BASE/weights/difix_ref" TORCH_EXTENSIONS_DIR=/tmp/lilinwei/Difix3D_Recheck_20261007/torch_extensions
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SUPPORT:$RUN/adapter:$BASE/examples/gsplat:$BASE:$FUSED:/tmp/lilinwei/1007baselines/repos/CoR-GS"
exec > >(tee -a "$LOG") 2>&1
printf 'START method=%s scene=%s physical_gpu=%s seed=%s time=%s\n' "$METHOD" "$SCENE" "$GPU_ID" "$SEED" "$(date -Is)"
printf 'data=%s mvs=%s mvs_sha256=%s adapter_sha=%s\n' "$DATA" "$MVS" "$(sha256sum "$MVS"|awk '{print $1}')" "$(sha256sum "$ADAPTER"|awk '{print $1}')"
printf 'RUNNING\n' > "$STATUS"
test -s "$TORCH_EXTENSIONS_DIR/gsplat_cuda/gsplat_cuda.so"; test -s "$MANIFEST"
if [ "$VARIANT" = "60k" ]; then DATA="$ROOT/data/llff/$SCENE"; fi
common=(--data-dir "$DATA" --result-dir "$SCRATCH" --strict-split-json "$MANIFEST" --mvs-ply "$MVS" --data-factor 4 --strict-no-undistort --test-every 1)
FIX=(); if [ "$VARIANT" = 10k ]; then for step in $(seq 5000 200 9400); do FIX+=("$step"); done; else for step in $(seq 30000 2000 60000); do FIX+=("$step"); done; fi
if [ "$VARIANT" = 60k ]; then
  "$PY" "$ADAPTER" default "${common[@]}" "${NORM_ARGS[@]}" --max-steps 30000 --eval-steps 30000 --save-steps 30000 --fix-steps 1000000 --disable-difix 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}; if [ "$RC" -ne 0 ]; then printf 'FAILED stage=base rc=%s\n' "$RC"; printf 'FAILED\n' > "$STATUS"; exit "$RC"; fi
  CKPT="$SCRATCH/ckpts/ckpt_29999_rank0.pt"; test -s "$CKPT"
  "$PY" "$ADAPTER" default "${common[@]}" "${NORM_ARGS[@]}" --ckpt "$CKPT" --max-steps 60000 --eval-steps 60000 --save-steps 60000 --fix-steps "${FIX[@]}" 2>&1 | tee -a "$LOG"
else
  "$PY" "$ADAPTER" default "${common[@]}" --seed 2024 --max-steps 10000 --eval-steps 5000 10000 --save-steps 5000 10000 --fix-steps "${FIX[@]}" 2>&1 | tee -a "$LOG"
fi
RC=${PIPESTATUS[0]}; if [ "$RC" -ne 0 ]; then printf 'FAILED stage=train rc=%s\n' "$RC" | tee -a "$LOG"; printf 'FAILED\n' > "$STATUS"; exit "$RC"; fi
if [ "$VARIANT" = 10k ]; then STEPS=(5000 10000); else STEPS=(30000 60000); fi
for step in "${STEPS[@]}"; do
  rd="$SCRATCH/renders/eval/ours_$(printf '%05d' "$step")"; ck="$SCRATCH/ckpts/ckpt_${step}_rank0.pt"; ply="$SCRATCH/point_cloud/iteration_${step}/point_cloud.ply"
  test -d "$rd/renders"; test -d "$rd/gt"; test -s "$ply"
  want=$("$PY" -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["test_names"]))' "$MANIFEST")
  test "$(find "$rd/renders" -maxdepth 1 -name '*.png'|wc -l)" -eq "$want"; test "$(find "$rd/gt" -maxdepth 1 -name '*.png'|wc -l)" -eq "$want"
  out="$ARCHIVE/outputs/$SCENE/eval/ours_$step"; mkdir -p "$out/renders" "$out/gt"
  cp -a "$rd/renders/." "$out/renders/"; cp -a "$rd/gt/." "$out/gt/"
  mkdir -p "$ARCHIVE/outputs/$SCENE/point_cloud/iteration_$step"; cp -a "$ply" "$ARCHIVE/outputs/$SCENE/point_cloud/iteration_$step/point_cloud.ply"
  if [ "$step" = "${STEPS[-1]}" ]; then cp -a "$ck" "$ARCHIVE/outputs/$SCENE/"; fi
done
cp -L "$MVS" "$ARCHIVE/outputs/$SCENE/input_mvs_fused.ply"
cp -a "$MANIFEST" "$ARCHIVE/data_mapping/$SCENE.json"
cp -a "$SCRATCH/cfg.yml" "$ARCHIVE/outputs/$SCENE/cfg.yml"
printf 'SUCCESS method=%s scene=%s gpu=%s seed=%s max_steps=%s eval_steps=%s time=%s\n' "$METHOD" "$SCENE" "$GPU_ID" "$SEED" "$MAX" "${STEPS[*]}" "$(date -Is)" | tee -a "$LOG"
mkdir -p "$ARCHIVE/logs"
cp -a "$LOG" "$ARCHIVE/logs/$SCENE.log"
rm -rf "$SCRATCH"
printf 'SUCCESS\n' > "$STATUS"
