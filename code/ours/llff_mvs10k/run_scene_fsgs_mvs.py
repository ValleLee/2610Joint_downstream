#!/usr/bin/env python3
"""Stage one scene safely, then execute the copied FSGS COLMAP script."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/data/lilinwei/3D/D23E/DO3/Scenedata/nerf_llff_data")
SPLIT_ROOT = Path("/data/lilinwei/3D/D23E/Aref/FSGS_LLFF_data")
TEMP_ROOT = Path("/tmp/lilinwei/1007fsgsmvs")
COLMAP_BIN = Path("/data/lilinwei/.conda/envs/colmap/bin/colmap")
OFFICIAL_SCRIPT = ROOT / "fsgs_colmap_llff.py"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_vertices(path: Path) -> int:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"no vertex count in PLY header: {path}")


def prepare(scene: str, gpu: int) -> Path:
    source = SOURCE_ROOT / scene
    canonical = SPLIT_ROOT / scene
    split = json.loads((canonical / "fsgs_do3_prepare_report.json").read_text())
    train_names = split["fsgs_train_names"]
    if len(train_names) != 3:
        raise ValueError(f"expected exactly three training images for {scene}")
    source_sparse = source / "sparse/0"
    for name in train_names:
        if not (source / "images" / name).is_file():
            raise FileNotFoundError(source / "images" / name)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (source_sparse / name).is_file():
            raise FileNotFoundError(source_sparse / name)

    scene_root = TEMP_ROOT / "colmap" / scene
    manifest_path = scene_root / "fsgs_mvs_manifest.json"
    cloud = scene_root / "fsgs_mvs_fused.ply"
    if manifest_path.is_file() and cloud.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete" and digest(cloud) == manifest.get("point_cloud_sha256"):
            return cloud

    # The original script converts sparse/0 in place and removes 3_views.
    # Give it an isolated scene copy so those operations cannot touch source data.
    staged_root = TEMP_ROOT / "fsgs_dataset"
    staged_scene = staged_root / scene
    # A stopped/incomplete run can be restarted from the immutable source data.
    if scene_root.exists():
        shutil.rmtree(scene_root)
    if staged_scene.exists() or staged_scene.is_symlink():
        shutil.rmtree(staged_scene)
    (staged_scene / "sparse").mkdir(parents=True)
    shutil.copytree(source_sparse, staged_scene / "sparse/0")
    (staged_scene / "images").symlink_to((source / "images").resolve(), target_is_directory=True)

    scene_root.mkdir(parents=True)
    log_path = scene_root / "fsgs_colmap.log"
    env = os.environ.copy()
    env["PATH"] = str(COLMAP_BIN.parent) + os.pathsep + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "1"
    command = [sys.executable, str(OFFICIAL_SCRIPT), "--scene", scene,
               "--base_path", str(staged_root), "--n_views", "3"]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"official_script={OFFICIAL_SCRIPT}\nphysical_gpu={gpu}\n")
        log.write(f"official_source={SOURCE_ROOT / scene}\ntrain_images={json.dumps(train_names)}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

    official_cloud = staged_scene / "3_views/dense/fused.ply"
    if result.returncode or not official_cloud.is_file():
        raise RuntimeError(f"official FSGS COLMAP script failed; inspect {log_path}")
    shutil.copy2(official_cloud, cloud)
    vertices = count_vertices(cloud)
    if vertices <= 0:
        raise RuntimeError(f"official FSGS generated an empty cloud for {scene}")

    manifest = {
        "scene": scene, "status": "complete", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "physical_gpu": gpu, "colmap": "COLMAP 3.12.6",
        "official_fsgs_script": "tools/colmap_llff.py",
        "script_copy": str(OFFICIAL_SCRIPT), "script_copy_sha256": digest(OFFICIAL_SCRIPT),
        "training_images": train_names, "point_triangulator_input_points": 0,
        "camera_poses_and_intrinsics": "selected poses and camera model copied from supplied sparse/0",
        "source_sparse_model_sha256": {n: digest(source_sparse / n)
            for n in ("cameras.bin", "images.bin", "points3D.bin")},
        "feature_settings": {"max_image_size": 4032, "max_num_features": 32768,
            "estimate_affine_shape": True, "domain_size_pooling": True},
        "dense_settings": "as in copied FSGS script, using installed COLMAP 3.12.6 CLI defaults",
        "initial_point_cloud": str(cloud), "point_count": vertices,
        "point_cloud_sha256": digest(cloud), "log": str(log_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Keep the fused cloud, manifest, and log; drop the large COLMAP database,
    # source image copies, depth/normal maps, and intermediate sparse models.
    shutil.rmtree(staged_scene, ignore_errors=True)
    return cloud


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu not in range(8) or args.gpu == 6:
        raise SystemExit("GPU 6 is excluded; choose 0, 1, 2, 3, 4, 5, or 7")
    print(prepare(args.scene, args.gpu), flush=True)
