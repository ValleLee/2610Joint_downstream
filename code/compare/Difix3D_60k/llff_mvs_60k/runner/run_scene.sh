#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: run_scene.sh <physical_gpu_id> <scene> [preflight|run]}"
SCENE="${2:?scene required}"
MODE="${3:-run}"
case "$GPU_ID" in 0|1|2|3|4|5|7) ;; *) echo "GPU $GPU_ID is not allowed; GPU 6 is excluded" >&2; exit 2;; esac
case "$SCENE" in fern|flower|fortress|horns|leaves|orchids|room|trex) ;; *) echo "Unknown scene: $SCENE" >&2; exit 2;; esac
case "$MODE" in preflight|run) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac

ROOT=/tmp/lilinwei/1007baselines
METHOD_ROOT=/data/lilinwei/3D/baselines/output/1007_fsgs_mvs_baselines/Difix3D_60k
BASE=/data/lilinwei/3D/baselines/Difix3D
PAIR="$BASE/run/20261007_recheck_pair"
RUN="$PAIR/60k_no_normalize"
DATA="$ROOT/data/llff/$SCENE"
MVS="$ROOT/data/$SCENE/3_views/dense/fused.ply"
MANIFEST="$METHOD_ROOT/data_mapping/$SCENE.json"
ADAPTER="$METHOD_ROOT/code/adapter/simple_trainer_difix3d.py"
SUPPORT="$METHOD_ROOT/code/adapter_support"
PY=/tmp/lilinwei/Difix3D_Recheck_20261007/env/bin/python
TORCH_EXT_DIR=/tmp/lilinwei/Difix3D_Recheck_20261007/torch_extensions
FUSED=/data/lilinwei/3D/0918pland/code/third_party/gaussian_splatting_planD/submodules/fused-ssim
EXPECTED_SHA="$(/data/lilinwei/.conda/envs/d23e/bin/python - "$SCENE" <<'PY'
import json,sys
p='/data/lilinwei/3D/baselines/output/1007_fsgs_mvs_baselines/configs/input_manifest.json'
print(json.load(open(p))['scenes'][sys.argv[1]]['mvs_sha256'])
PY
)"
ACTUAL_SHA="$(sha256sum "$MVS" 2>/dev/null | awk '{print $1}')"

test -d "$DATA/sparse" || { echo "Missing LLFF sparse model: $DATA/sparse" >&2; exit 2; }
test -d "$DATA/images_4" || { echo "Missing images_4: $DATA/images_4" >&2; exit 2; }
test -s "$MVS" || { echo "Missing staged MVS input: $MVS" >&2; exit 2; }
test "$ACTUAL_SHA" = "$EXPECTED_SHA" || { echo "MVS SHA mismatch: $ACTUAL_SHA != $EXPECTED_SHA" >&2; exit 2; }
test -s "$MANIFEST"; test -s "$ADAPTER"; test -s "$PY"; test -s "$FUSED/"; test -s "$SUPPORT/pycolmap.py"
test -s "$TORCH_EXT_DIR/gsplat_cuda/gsplat_cuda.so"

if [ "$MODE" = preflight ]; then
  PYTHONNOUSERSITE=1 "$PY" - "$TORCH_EXT_DIR/gsplat_cuda/gsplat_cuda.so" <<'PY'
import sys,torch,gsplat,diffusers
assert torch.cuda.is_available()
torch.ops.load_library(sys.argv[1])
print('python_cuda_extension_preflight_ok',torch.__version__,torch.version.cuda,gsplat.__version__,diffusers.__version__)
PY
  /data/lilinwei/.conda/envs/d23e/bin/python - "$SCENE" "$MANIFEST" <<'PY'
import json,sys
scene=json.load(open(sys.argv[2]))
cfg=json.load(open('/data/lilinwei/3D/baselines/output/1007_fsgs_mvs_baselines/configs/input_manifest.json'))['scenes'][sys.argv[1]]
assert scene['training_names']==cfg['train_names']
assert scene['test_names']==cfg['test_names']
assert len(scene['test_names'])==cfg['eval_count']
print('split_preflight_ok',sys.argv[1],len(scene['training_names']),len(scene['test_names']))
PY
  printf 'PREFLIGHT_OK scene=%s MVS_sha256=%s\n' "$SCENE" "$ACTUAL_SHA"
  exit 0
fi

RUN_ID=seed2024_fsgs_official_mvs_Difix3D_60k
SCRATCH="$ROOT/scratch/$RUN_ID/$SCENE"
STATUS_DIR="$ROOT/status/$RUN_ID"
STATUS="$STATUS_DIR/$SCENE.status"
LOG="$METHOD_ROOT/logs/$SCENE.log"
ARCHIVE="$METHOD_ROOT/outputs/$SCENE"
if [ -e "$STATUS" ] || [ -e "$SCRATCH" ]; then
  echo "Refusing to overwrite an existing run state: status=$STATUS scratch=$SCRATCH" >&2
  exit 2
fi
mkdir -p "$SCRATCH" "$STATUS_DIR" "$ARCHIVE" "$METHOD_ROOT/logs"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONNOUSERSITE=1 BASELINE_SEED=2024 PYTHONHASHSEED=2024
export HF_HOME=/tmp/lilinwei/Difix3D_Recheck_20261007/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export DIFIX_MODEL_DIR="$BASE/weights/difix_ref"
export TORCH_EXTENSIONS_DIR="$TORCH_EXT_DIR"
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SUPPORT:$METHOD_ROOT/code/adapter:$BASE/examples/gsplat:$BASE:$FUSED:/tmp/lilinwei/1007baselines/repos/CoR-GS"
exec > >(tee "$LOG") 2>&1

START="$(date +%s)"
printf 'START method=Difix3D_60k scene=%s physical_gpu=%s seed=2024 time=%s\n' "$SCENE" "$GPU_ID" "$(date -Is)"
printf 'data=%s mvs=%s mvs_sha256=%s adapter_sha256=%s\n' "$DATA" "$MVS" "$ACTUAL_SHA" "$(sha256sum "$ADAPTER" | awk '{print $1}')"
printf 'schedule=base 0-30000 without Difix; resume same scene checkpoint to 60000 with target-pose fixes every 2000 steps\n'
printf 'RUNNING_STAGE_A\n' > "$STATUS"
common=(--data-dir "$DATA" --result-dir "$SCRATCH" --strict-split-json "$MANIFEST" --mvs-ply "$MVS" --data-factor 4 --strict-no-undistort --test-every 1)

run_logged() {
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

if run_logged "$PY" "$ADAPTER" default "${common[@]}" --no-normalize-world-space \
    --max-steps 30000 --eval-steps 30000 --save-steps 30000 --fix-steps 1000000 --disable-difix; then
  :
else
  rc=$?; printf 'FAILED_STAGE_A rc=%s time=%s\n' "$rc" "$(date -Is)"; printf 'FAILED_STAGE_A\n' > "$STATUS"; exit "$rc"
fi
BASE_CKPT="$SCRATCH/ckpts/ckpt_29999_rank0.pt"
test -s "$BASE_CKPT"
cp -a "$BASE_CKPT" "$ARCHIVE/ckpt_29999_rank0.pt"
printf 'RUNNING_STAGE_B\n' > "$STATUS"
FIX_STEPS=(); for step in $(seq 30000 2000 60000); do FIX_STEPS+=("$step"); done
if run_logged "$PY" "$ADAPTER" default "${common[@]}" --no-normalize-world-space \
    --ckpt "$BASE_CKPT" --max-steps 60000 --eval-steps 60000 --save-steps 60000 \
    --fix-steps "${FIX_STEPS[@]}"; then
  :
else
  rc=$?; printf 'FAILED_STAGE_B rc=%s time=%s\n' "$rc" "$(date -Is)"; printf 'FAILED_STAGE_B\n' > "$STATUS"; exit "$rc"
fi

N_TEST="$(/data/lilinwei/.conda/envs/d23e/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["test_names"]))' "$MANIFEST")"
for step in 30000 60000; do
  RENDER_DIR="$SCRATCH/renders/eval/ours_$(printf '%05d' "$step")"
  PLY="$SCRATCH/point_cloud/iteration_${step}/point_cloud.ply"
  test -d "$RENDER_DIR/renders" && test -d "$RENDER_DIR/gt" && test -s "$PLY"
  test "$(find "$RENDER_DIR/renders" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$N_TEST"
  test "$(find "$RENDER_DIR/gt" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$N_TEST"
  OUT_EVAL="$ARCHIVE/eval/ours_${step}"
  mkdir -p "$OUT_EVAL/renders" "$OUT_EVAL/gt" "$ARCHIVE/point_cloud/iteration_${step}"
  cp -a "$RENDER_DIR/renders/." "$OUT_EVAL/renders/"
  cp -a "$RENDER_DIR/gt/." "$OUT_EVAL/gt/"
  cp -a "$PLY" "$ARCHIVE/point_cloud/iteration_${step}/point_cloud.ply"
done
FINAL_CKPT="$SCRATCH/ckpts/ckpt_60000_rank0.pt"
test -s "$FINAL_CKPT"
cp -a "$FINAL_CKPT" "$ARCHIVE/ckpt_60000_rank0.pt"
cp -L "$MVS" "$ARCHIVE/input_mvs_fused.ply"
cp -a "$SCRATCH/cfg.yml" "$ARCHIVE/cfg.yml"
if [ -d "$SCRATCH/stats" ]; then cp -a "$SCRATCH/stats" "$ARCHIVE/"; fi
test "$(sha256sum "$ARCHIVE/input_mvs_fused.ply" | awk '{print $1}')" = "$EXPECTED_SHA"
ELAPSED=$(( $(date +%s) - START ))
printf 'SUCCESS method=Difix3D_60k scene=%s physical_gpu=%s seed=2024 updates=60000 views=%s elapsed_seconds=%s time=%s\n' "$SCENE" "$GPU_ID" "$N_TEST" "$ELAPSED" "$(date -Is)"
printf 'SUCCESS\n' > "$STATUS"
# Keep scratch checkpoints, TensorBoard events, and intermediate renders while space is available.
