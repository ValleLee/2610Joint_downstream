#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:?method: DNGaussian or DropoutGS}
SCAN=${2:?scan name, e.g. scan8}
GPU=${3:?visible GPU id}
if [[ "$GPU" == 6 ]]; then echo "GPU 6 is prohibited" >&2; exit 2; fi
case "$METHOD" in
  DNGaussian|DropoutGS) ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

BASE=/data/lilinwei/3D/1006difix/1007fsgsmvs/dtu
REPO=/data/lilinwei/3D/baselines/$METHOD
TMP=/tmp/lilinwei/1007fsgsmvs_dtu/official_random_6k
DEST=$BASE/output/official_random_6k/$METHOD/$SCAN
PY=/data/lilinwei/.conda/envs/d23e/bin/python
RUN=$TMP/models/$METHOD/$SCAN
DATA_ORIG=/tmp/lilinwei/1007fsgsmvs_dtu/overlay/DTU/$SCAN
DEPTH8=$TMP/data_8bit/$SCAN/depth_maps
DATA=$TMP/source/$METHOD/$SCAN
mkdir -p "$RUN" "$(dirname "$DEST")" "$TMP/torch_extensions/$METHOD"
if [[ -e "$DEST/run_manifest.json" ]]; then echo "$DEST already has a manifest; refusing overwrite" >&2; exit 3; fi

if [[ ! -f "$DEPTH8/conversion_manifest.json" ]]; then
  mkdir -p "$(dirname "$DEPTH8")"
  LOCK=$TMP/data_8bit/$SCAN.prepare_lock
  if mkdir "$LOCK" 2>/dev/null; then
    "$PY" "$BASE/output/official_random_6k/code/prepare_depth_maps_8bit.py" \
      "$DATA_ORIG/depth_maps" "$DEPTH8" > "$TMP/logs/depth8_$SCAN.log" 2>&1
    rmdir "$LOCK"
  else
    until [[ -f "$DEPTH8/conversion_manifest.json" ]]; do sleep 2; done
  fi
fi
mkdir -p "$DATA/sparse/0"
ln -sfn "$DATA_ORIG/images" "$DATA/images"
ln -sfn "$DEPTH8" "$DATA/depth_maps"
for f in cameras.bin images.bin points3D.bin; do
  cp -f "$DATA_ORIG/sparse/0/$f" "$DATA/sparse/0/$f"
done
rm -f "$DATA/sparse/0/points3D_random.ply"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONNOUSERSITE=1
RASTER_PKG=$TMP/python_packages/$METHOD
export PYTHONPATH="$RASTER_PKG:$REPO/gridencoder:$REPO/shencoder:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$TMP/torch_extensions/$METHOD"

cd "$REPO"
COMMON=(--dataset DTU -s "$DATA" --model_path "$RUN" -r 4 --eval --n_sparse 3 --rand_pcd
  --iterations 6000 --lambda_dssim 0.6
  --densify_grad_threshold 0.001 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.1
  --position_lr_init 0.0016 --position_lr_final 0.000016 --position_lr_max_steps 5500 --position_lr_start 500
  --test_iterations 100 1000 2000 3000 4500 6000
  --save_iterations 100 500 1000 3000 6000
  --error_tolerance 0.01 --opacity_lr 0.05 --scaling_lr 0.003
  --shape_pena 0.005 --opa_pena 0.001 --scale_pena 0.005)
if [[ "$METHOD" == DNGaussian ]]; then
  "$PY" train_dtu.py "${COMMON[@]}" > "$DEST.tmp.log" 2>&1
else
  "$PY" train_dtu.py "${COMMON[@]}" --dropout --dropout_rate 0.3 --consistency_lambda 0.5 \
    --edge_threshold 0.001 --size_threshold_rate 1 > "$DEST.tmp.log" 2>&1
fi

"$PY" render.py -s "$DATA" --model_path "$RUN" -r 4 --iteration 6000 --skip_train --skip_test \
  >> "$DEST.tmp.log" 2>&1

mkdir -p "$DEST/renders" "$DEST/point_cloud" "$DEST/initialization"
cp "$RUN/chkpnt_latest.pth" "$DEST/"
cp "$RUN/point_cloud/iteration_6000/point_cloud.ply" "$DEST/point_cloud/point_cloud.ply"
cp "$RUN/input.ply" "$DEST/initialization/points3D_random.ply"
gzip -1 -c "$RUN/chkpnt_latest.pth" > "$DEST/checkpoint_6000.pth.gz"
gzip -t "$DEST/checkpoint_6000.pth.gz"
rm "$DEST/chkpnt_latest.pth"
cp "$RUN/cfg_args" "$RUN/opt_args" "$DEST/"

# Official reader uses sorted 49-camera records and the 25 CoR-GS test indices.
# render.py emits those cameras as 00000.png ... 00024.png with shuffle=False.
"$PY" - "$SCAN" "$RUN/eval/ours_6000/renders" "$DEST/renders" "$BASE/data/splits/cor_gs_3view/$SCAN.json" <<'PY'
import json, re, shutil, sys
from pathlib import Path
scan, src, dst, split_path = sys.argv[1:]
src, dst = Path(src), Path(dst)
split = json.loads(Path(split_path).read_text())
names = sorted(x['name'] for x in split['image_records'])
reserved = {25,22,28,40,44,48,0,8,13}
excluded = {3,4,5,6,7,16,17,18,19,20,21,36,37,38,39}
test = [names[i] for i in range(49) if i not in reserved | excluded]
files = sorted(src.glob('*.png'))
if len(files) != 25 or len(test) != 25:
    raise SystemExit(f'{scan}: expected 25 renders/test views, got {len(files)}/{len(test)}')
for i, name in enumerate(test):
    shutil.copy2(src / f'{i:05d}.png', dst / name)
print('render_mapping_verified', scan, len(test))
PY

cp "$DEST.tmp.log" "$DEST/train_render.log"
rm "$DEST.tmp.log"
"$PY" - "$METHOD" "$SCAN" "$GPU" "$RUN" "$DEST" "$REPO" "$DATA" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
method, scan, gpu, run, dest, repo, data = sys.argv[1:]
run, dest, repo, data = map(Path, (run, dest, repo, data))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
paths=['scripts/train_dtu.sh','train_dtu.py','scene/dataset_readers.py','render.py','utils/general_utils.py']
manifest={'method':method,'scan':scan,'gpu':int(gpu),'official_seed':0,
 'training_rgb_views':['rect_023_3_r5000.png','rect_026_3_r5000.png','rect_029_3_r5000.png'],
 'test_view_count':25,'iterations':6000,'initialization':'official rand_pcd (1000 points)',
 'random_pointcloud_bound':'sparse/0/points3D.bin from 49-camera COLMAP model (additional geometry prior)',
 'data_path':str(data),'depth_maps':'DNGaussian official DPT per-image normalized maps, converted from 16-bit PNG to loader-compatible 8-bit; only three train cameras enter depth loss',
 'repo_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'source_sha256':{p:sha(repo/p) for p in paths if (repo/p).is_file()},
 'checkpoint_gzip_sha256':sha(dest/'checkpoint_6000.pth.gz'),
 'point_cloud_sha256':sha(dest/'point_cloud/point_cloud.ply'),
 'init_cloud_sha256':sha(dest/'initialization/points3D_random.ply'),
 'render_count':len(list((dest/'renders').glob('*.png')))}
if manifest['render_count'] != 25: raise SystemExit('wrong archived render count')
(dest/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
PY

# Remove the temporary checkpoints, TensorBoard files, and training render cache only
# after final artifacts and the manifest are verified in the persistent DTU output.
test -s "$DEST/checkpoint_6000.pth.gz"
test "$(find "$DEST/renders" -maxdepth 1 -name '*.png' | wc -l)" -eq 25
rm -rf "$RUN"
df -h /tmp | tail -1 >> "$DEST/train_render.log"
echo "DONE $METHOD $SCAN GPU$GPU"
