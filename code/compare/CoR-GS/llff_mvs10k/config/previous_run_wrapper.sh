#!/usr/bin/env bash
set -Eeuo pipefail
METHOD="${1:?method required}"
GPU_ID="${2:?gpu required}"
SCENE="${3:?scene required}"
ROOT=/tmp/lilinwei/1007baselines
case "$METHOD" in
  FSGS|CoR-GS|DNGaussian|DropoutGS) SEED=2024 ;;
  *) echo "unknown method $METHOD"; exit 2 ;;
esac
RUN_TAG="seed2024"
export BASELINE_SEED=2024 PYTHONHASHSEED=2024
DATA="$ROOT/data/$SCENE"
if [ "$METHOD" = "CoR-GS" ]; then DATA="$ROOT/data/llff/$SCENE"; fi
REPO="$ROOT/repos/$METHOD"
PY=/data/lilinwei/.conda/envs/d23e/bin/python
SCRATCH="$ROOT/scratch/$RUN_TAG/$METHOD/$SCENE"
ARCHIVE="/data/lilinwei/3D/baselines/output/1007_mvs_baselines/$METHOD"
LOG="$ROOT/logs/${METHOD}_${SCENE}_${RUN_TAG}.log"
STATUS="$ROOT/status/${METHOD}_${SCENE}.status"
mkdir -p "$ROOT/logs" "$ROOT/status" "$SCRATCH" "$ARCHIVE/outputs/$SCENE"
export CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=4
case "$METHOD" in
  FSGS)
    PY=/data/lilinwei/.conda/envs/do3/bin/python
    unset PYTHONNOUSERSITE
    export PYTHONPATH="/tmp/lilinwei/1007baselines/python310_compat:/data/lilinwei/.conda/envs/do3/clean_site:/data/lilinwei/.conda/envs/do3/lib/python3.10/site-packages:$REPO/submodules/diff-gaussian-rasterization-confidence:$REPO/submodules/simple-knn:$REPO" ;;
  CoR-GS) export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO/submodules/diff-gaussian-rasterization-confidence:$REPO/submodules/simple-knn:$REPO" ;;
  DNGaussian) export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO/diff-gaussian-rasterization:$REPO/gridencoder:$REPO" ;;
  DropoutGS) export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO/submodules/diff-gaussian-rasterization:$REPO/gridencoder:$REPO/shencoder:$REPO" ;;
  *) echo "unknown method $METHOD"; exit 2 ;;
esac
cd "$REPO"
exec > >(tee -a "$LOG") 2>&1
printf 'START method=%s scene=%s physical_gpu=%s seed=%s time=%s\n' "$METHOD" "$SCENE" "$GPU_ID" "$SEED" "$(date -Is)"
printf 'input=%s mvs=%s mvs_points=%s mvs_sha256=%s\n' "$DATA" "$DATA/3_views/dense/fused.ply" "$(sed -n 's/^element vertex //p' "$DATA/3_views/dense/fused.ply" | head -1)" "$(sha256sum "$DATA/3_views/dense/fused.ply" | awk '{print $1}')"
printf 'train_count=%s eval_count=%s\n' "$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["fsgs_train_names"]))' "$DATA/fsgs_do3_prepare_report.json")" "$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["fsgs_test_names"]))' "$DATA/fsgs_do3_prepare_report.json")"
printf 'gpu_snapshot_start\n'; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
printf 'RUNNING\n' > "$STATUS"
run_step(){ local label="$1"; shift; printf '\nCOMMAND[%s]' "$label"; printf ' %q' "$@"; printf '\n'; set +e; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; set -e; if [ "$rc" -ne 0 ]; then printf 'FAILED step=%s rc=%s time=%s\n' "$label" "$rc" "$(date -Is)"; printf 'FAILED\n' > "$STATUS"; exit "$rc"; fi; }
case "$METHOD" in
FSGS)
  run_step train "$PY" -u train.py --source_path "$DATA" --model_path "$SCRATCH" --images images_4 --resolution 1 --eval --n_views 3 --sample_pseudo_interval 1 --iterations 10000 --test_iterations 10000 --save_iterations 10000 --checkpoint_iterations 10000
  run_step render "$PY" -u render.py --source_path "$DATA" --model_path "$SCRATCH" --iteration 10000 --skip_train
  NATIVE="$SCRATCH/test/ours_10000"; ITER=10000
  ;;
CoR-GS)
  run_step train "$PY" -u train.py --source_path "$DATA" --model_path "$SCRATCH" --images images_4 --resolution 1 --eval --n_views 3 --random_background --iterations 10000 --position_lr_max_steps 10000 --densify_until_iter 10000 --densify_grad_threshold 0.0005 --gaussiansN 2 --coprune --coprune_threshold 5 --coreg --sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000 --save_iterations 10000 --checkpoint_iterations 10000
  run_step render "$PY" -u render.py --source_path "$DATA" --model_path "$SCRATCH" --iteration 10000 --skip_train --render_depth
  NATIVE="$SCRATCH/test/ours_10000"; ITER=10000
  ;;
DNGaussian)
  test -d "$DATA/depth_maps"
  run_step train "$PY" -u train_llff.py -s "$DATA" --model_path "$SCRATCH" --images images_4 --resolution 1 --dataset LLFF --eval --n_sparse 3 --mvs_pcd --iterations 10000 --lambda_dssim 0.25 --densify_grad_threshold 0.0008 --prune_threshold 0.01 --densify_until_iter 10000 --percent_dense 0.01 --position_lr_init 0.0005 --position_lr_final 0.000005 --position_lr_max_steps 9500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --shape_pena 0.002 --opa_pena 0.001 --scale_pena 0 --test_iterations 10000 --save_iterations 10000 --checkpoint_iterations 10000 --near 10
  run_step render "$PY" -u render.py -s "$DATA" --model_path "$SCRATCH" --images images_4 --resolution 1 --iteration 10000 --near 10 --skip_train --skip_test
  NATIVE="$SCRATCH/eval/ours_10000"; ITER=10000
  ;;
DropoutGS)
  case "$SCENE" in
    leaves|horns) EDGE=0.0005; SIZE=50 ;;
    fortress|flower|fern|room) EDGE=0.001; SIZE=80 ;;
    trex|orchids) EDGE=0.001; SIZE=50 ;;
  esac
  test -d "$DATA/depth_maps"
  run_step train "$PY" -u train_llff_mvs.py -s "$DATA" --model_path "$SCRATCH" --images images_4 --resolution 1 --dataset LLFF --eval --n_sparse 3 --mvs_pcd --iterations 10000 --lambda_dssim 0.25 --densify_grad_threshold 0.0008 --prune_threshold 0.01 --densify_until_iter 10000 --percent_dense 0.01 --position_lr_init 0.0005 --position_lr_final 0.000005 --position_lr_max_steps 9500 --position_lr_start 500 --split_opacity_thresh 0.1 --error_tolerance 0.01 --scaling_lr 0.005 --shape_pena 0.002 --opa_pena 0.001 --scale_pena 0 --near 10 --test_iterations 10000 --save_iterations 10000 --dropout --dropout_rate 0.4 --consistency_lambda 0.4 --edge_threshold "$EDGE" --size_threshold_rate "$SIZE" --checkpoint_iterations 10000
  run_step render "$PY" -u render.py -s "$DATA" --model_path "$SCRATCH" --images images_4 --resolution 1 --iteration 10000 --near 10 --skip_train --skip_test
  NATIVE="$SCRATCH/eval/ours_10000"; ITER=10000
  ;;
esac
P="$SCRATCH/point_cloud/iteration_${ITER}/point_cloud.ply"
test -s "$P"; test -d "$NATIVE/renders"; test -d "$NATIVE/gt"
EXPECTED=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["fsgs_test_names"]))' "$DATA/fsgs_do3_prepare_report.json")
if [ "$METHOD" = "CoR-GS" ]; then
  NR=$(find "$NATIVE/renders" -maxdepth 1 -type f -name '*.png' ! -name '*_depth.png' | wc -l)
else
  NR=$(find "$NATIVE/renders" -maxdepth 1 -type f -name '*.png' | wc -l)
fi
NG=$(find "$NATIVE/gt" -maxdepth 1 -type f -name '*.png' | wc -l)
test "$NR" -eq "$EXPECTED"; test "$NG" -eq "$EXPECTED"
mkdir -p "$ARCHIVE/data_mapping" "$ARCHIVE/outputs/$SCENE/eval/ours_${ITER}/renders" "$ARCHIVE/outputs/$SCENE/eval/ours_${ITER}/gt" "$ARCHIVE/outputs/$SCENE/point_cloud/iteration_${ITER}"
if [ "$METHOD" = "CoR-GS" ]; then
  find "$NATIVE/renders" -maxdepth 1 -type f -name '*.png' ! -name '*_depth.png' -exec cp -a -t "$ARCHIVE/outputs/$SCENE/eval/ours_${ITER}/renders/" {} +
  mkdir -p "$ARCHIVE/outputs/$SCENE/depth_visualizations/iteration_${ITER}"
  find "$NATIVE/renders" -maxdepth 1 -type f -name '*_depth.png' -exec cp -a -t "$ARCHIVE/outputs/$SCENE/depth_visualizations/iteration_${ITER}/" {} +
else
  cp -a "$NATIVE/renders/." "$ARCHIVE/outputs/$SCENE/eval/ours_${ITER}/renders/"
fi
cp -a "$NATIVE/gt/." "$ARCHIVE/outputs/$SCENE/eval/ours_${ITER}/gt/"
cp -a "$P" "$ARCHIVE/outputs/$SCENE/point_cloud/iteration_${ITER}/point_cloud.ply"
CK="$SCRATCH/chkpnt${ITER}.pth"; test -s "$CK"; cp -a "$CK" "$ARCHIVE/outputs/$SCENE/"
cp -L "$DATA/3_views/dense/fused.ply" "$ARCHIVE/outputs/$SCENE/input_mvs_fused.ply"
cp -a "$DATA/fsgs_do3_prepare_report.json" "$ARCHIVE/outputs/$SCENE/data_split.json"
cp -a "/data/lilinwei/3D/baselines/output/1007_mvs_baselines/configs/canonical_data_mapping/$SCENE.json" "$ARCHIVE/data_mapping/$SCENE.json"
for f in cfg_args opt_args cameras.json input.ply; do
  if [ -s "$SCRATCH/$f" ]; then cp -a "$SCRATCH/$f" "$ARCHIVE/outputs/$SCENE/$f"; fi
done
if [ -s "$SCRATCH/point_cloud_gs2/iteration_${ITER}/point_cloud.ply" ]; then
  mkdir -p "$ARCHIVE/outputs/$SCENE/point_cloud_gs2/iteration_${ITER}"
  cp -a "$SCRATCH/point_cloud_gs2/iteration_${ITER}/point_cloud.ply" "$ARCHIVE/outputs/$SCENE/point_cloud_gs2/iteration_${ITER}/point_cloud.ply"
fi
printf 'SUCCESS method=%s scene=%s gpu=%s seed=%s iterations=%s renders=%s time=%s\n' "$METHOD" "$SCENE" "$GPU_ID" "$SEED" "$ITER" "$NR" "$(date -Is)" | tee -a "$LOG"
mkdir -p "$ARCHIVE/logs"
cp -a "$LOG" "$ARCHIVE/logs/$SCENE.log"
rm -rf "$SCRATCH"
printf 'SUCCESS\n' > "$STATUS"
