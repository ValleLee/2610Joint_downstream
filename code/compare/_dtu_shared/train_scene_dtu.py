#!/usr/bin/env python3
"""Train and archive one method/scene with the frozen DTU inputs."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from PIL import Image

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
TEMP = Path("/tmp/lilinwei/1007fsgsmvs_dtu")
METHODS = CODE / "methods"
PY_GS = Path("/data/lilinwei/.conda/envs/d23e/bin/python")
PY_FSGS = Path("/data/lilinwei/.conda/envs/do3/bin/python")
PY_R4 = Path("/data/lilinwei/3D/1006difix/1007mast3r/envs/mast3r/bin/python")
PY_DIFIX = Path("/tmp/lilinwei/Difix3D_Recheck_20261007/env/bin/python")
ALLOWED = {"FSGS", "CoR-GS", "DropoutGS", "DNGaussian", "Difix3D_10k", "our_method"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def env_for(method: str, gpu: int) -> tuple[dict[str, str], Path, Path]:
    if gpu == 6 or gpu not in range(8):
        raise ValueError("GPU 6 禁用；只接受 0-5 或 7")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "BASELINE_SEED": "2024",
                "PYTHONHASHSEED": "2024", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"})
    if method == "FSGS":
        repo = METHODS / method
        env["PYTHONPATH"] = os.pathsep.join(map(str, [
            Path("/tmp/lilinwei/1007baselines/python310_compat"),
            Path("/data/lilinwei/.conda/envs/do3/clean_site"),
            Path("/data/lilinwei/.conda/envs/do3/lib/python3.10/site-packages"),
            repo / "submodules/diff-gaussian-rasterization-confidence",
            repo / "submodules/simple-knn", repo]))
        return env, repo, PY_FSGS
    if method in ("CoR-GS", "DropoutGS", "DNGaussian"):
        repo = METHODS / method
        env["PYTHONNOUSERSITE"] = "1"
        pieces = []
        if method in ("CoR-GS", "FSGS"):
            pieces += [repo / "submodules/diff-gaussian-rasterization-confidence", repo / "submodules/simple-knn"]
        elif method == "DropoutGS":
            pieces += [repo / "submodules/diff-gaussian-rasterization", repo / "gridencoder", repo / "shencoder"]
        else:
            pieces += [repo / "diff-gaussian-rasterization", repo / "gridencoder"]
        env["PYTHONPATH"] = os.pathsep.join(map(str, [*pieces, repo]))
        return env, repo, PY_GS
    if method == "our_method":
        repo = METHODS / "FSGS"
        do3_env = Path("/data/lilinwei/.conda/envs/do3")
        env.update({"PYTHONNOUSERSITE": "1",
                    # The copied 0930jointR config contains a stale /data1 host path.
                    "JOINT_BASE_CKPT_DIR": "/data/lilinwei/3D/D23E"})
        env["PYTHONPATH"] = os.pathsep.join(map(str, [
            do3_env / "clean_site", do3_env / "lib/python3.10/site-packages",
            Path("/data/lilinwei/3D/1006difix/1007mast3r/envs/mast3r/lib/python3.10/site-packages"),
            CODE / "r4_support", repo / "submodules/diff-gaussian-rasterization-confidence",
            repo / "submodules/simple-knn", repo]))
        return env, CODE, PY_FSGS

    repo = METHODS / "Difix3D"
    cfg = repo / "configs/difix10k"
    fused = Path("/data/lilinwei/3D/0918pland/code/third_party/gaussian_splatting_planD/submodules/fused-ssim")
    env.update({"PYTHONNOUSERSITE": "1", "HF_HOME": str(TEMP / "huggingface"),
                "HF_HUB_CACHE": str(TEMP / "huggingface/hub"),
                "DIFIX_MODEL_DIR": "/data/lilinwei/3D/baselines/Difix3D/weights/difix_ref",
                "TORCH_EXTENSIONS_DIR": str(TEMP / "difix_torch_extensions"),
                "MAX_JOBS": "4", "TORCH_CUDA_ARCH_LIST": "8.6", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": os.pathsep.join(map(str, [cfg, cfg / "adapter_support", cfg / "datasets",
                                                           TEMP / "fused_ssim_src", repo / "examples/gsplat",
                                                           repo, fused, METHODS / "CoR-GS"]))})
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env, repo, PY_DIFIX


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log) -> None:
    log.write("\nCOMMAND " + " ".join(map(str, cmd)) + "\n")
    log.flush()
    subprocess.run(list(map(str, cmd)), cwd=str(cwd), env=env, stdout=log,
                   stderr=subprocess.STDOUT, check=True)


def common_3dgs_args(method: str, source: Path, model: Path) -> list[str]:
    args = ["--source_path", str(source), "--model_path", str(model), "--images", "images",
            "--resolution", "4", "--eval", "--n_views", "3", "--iterations", "10000",
            "--test_iterations", "10000", "--save_iterations", "10000", "--checkpoint_iterations", "10000"]
    if method == "FSGS":
        return args + ["--sample_pseudo_interval", "1"]
    return args + ["--random_background", "--position_lr_max_steps", "10000",
                   "--densify_until_iter", "10000", "--densify_grad_threshold", "0.0005",
                   "--gaussiansN", "2", "--coprune", "--coprune_threshold", "10", "--coreg",
                   "--sample_pseudo_interval", "1", "--start_sample_pseudo", "2000"]


def train_command(method: str, python: Path, repo: Path, source: Path, model: Path,
                  mapping: dict) -> tuple[list[str], list[str] | None]:
    if method in ("FSGS", "CoR-GS"):
        train = [str(python), str(CODE / "mipnerf_reference/fsgs_corgs/compat_run.py"),
                 "--repo", str(repo), "--script", "train.py", "--", *common_3dgs_args(method, source, model)]
        render_args = ["--source_path", str(source), "--model_path", str(model), "--iteration", "10000",
                       "--skip_train", "--render_depth"]
        render = [str(python), str(CODE / "mipnerf_reference/simple_run.py"),
                  "--repo", str(repo), "--script", "render.py", "--", *render_args]
        return train, render

    if method in ("DropoutGS", "DNGaussian"):
        args = ["--source_path", str(source), "--model_path", str(model), "--images", "images",
                "--resolution", "4", "--dataset", "DTU", "--eval", "--n_sparse", "3", "--mvs_pcd",
                "--iterations", "10000", "--test_iterations", "10000", "--save_iterations", "10000",
                # DNGaussian/DropoutGS 官方 DTU 脚本参数；仅将迭代相关终点按本轮 10k 延长。
                "--checkpoint_iterations", "10000", "--lambda_dssim", "0.6",
                "--densify_grad_threshold", "0.001", "--prune_threshold", "0.01",
                "--densify_until_iter", "10000", "--percent_dense", "0.1",
                "--position_lr_init", "0.0016", "--position_lr_final", "0.000016",
                "--position_lr_max_steps", "9500", "--position_lr_start", "500",
                "--error_tolerance", "0.01", "--opacity_lr", "0.05",
                "--scaling_lr", "0.003", "--shape_pena", "0.005",
                "--opa_pena", "0.001", "--scale_pena", "0.005"]
        if method == "DropoutGS":
            args += ["--dropout", "--dropout_rate", "0.3", "--consistency_lambda", "0.5",
                     "--edge_threshold", "0.001", "--size_threshold_rate", "1"]
            script = "train_dtu.py"
        else:
            script = "train_dtu.py"
        train = [str(python), str(CODE / "mipnerf_reference/simple_run.py"),
                 "--repo", str(repo), "--script", script, "--", *args]
        render_args = ["--source_path", str(source), "--model_path", str(model), "--images", "images",
                       "--resolution", "4", "--dataset", "DTU", "--eval", "--n_sparse", "3", "--mvs_pcd",
                       "--iteration", "10000", "--skip_train", "--skip_test"]
        render = [str(python), str(CODE / "mipnerf_reference/simple_run.py"),
                  "--repo", str(repo), "--script", "render.py", "--", *render_args]
        return train, render

    if method == "our_method":
        pose_file = PROJECT / "data/pseudo_pose_targets" / f"{mapping['scan']}.json"
        train = [str(python), str(CODE / "train_r4_memory_dtu.py"), "--source_path", str(source),
                 "--model_path", str(model), "--images", "images", "--resolution", "4", "--eval",
                 "--n_views", "3", "--iterations", "10000", "--test_iterations", "10000",
                 "--save_iterations", "10000", "--checkpoint_iterations", "10000", "--seed", "2024",
                 "--pseudo_training_mode", "difix",
                 "--pseudo_target_source", "external", "--pseudo_target_pose_file", str(pose_file),
                 "--pseudo_target_count", "8", "--pseudo_target_index", "0",
                 "--joint_code_root", "/data/lilinwei/3D/0930jointR"]
        render_args = ["--source_path", str(source), "--model_path", str(model), "--iteration", "10000",
                       "--skip_train", "--render_depth"]
        repo = METHODS / "FSGS"
        render = [str(PY_FSGS), str(CODE / "mipnerf_reference/simple_run.py"), "--repo", str(repo),
                  "--script", "render.py", "--", *render_args]
        return train, render

    cfg = METHODS / "Difix3D/configs/difix10k"
    # Difix3D 官方 60k 日程在前 10k 的修复时点；按用户要求只跑 10k。
    fix_steps = ["3000", "6000", "8000", "10000"]
    train = [str(python), str(cfg / "simple_trainer_difix3d.py"), "default", "--data-dir", str(source),
             "--result-dir", str(model), "--strict-split-json", mapping["strict_split"],
             "--mvs-ply", mapping["initial_cloud"], "--data-factor", "4", "--strict-no-undistort",
             "--test-every", "1", "--seed", "2024", "--max-steps", "10000", "--eval-steps", "10000",
             "--save-steps", "10000", "--fix-steps", *fix_steps]
    return train, None


def paired_pngs(directory: Path, expected_names: list[str]) -> dict[str, Path]:
    files = sorted(p for p in directory.glob("*.png") if not p.name.endswith("_depth.png")
                   and not p.name.startswith("alpha_") and not p.name.startswith("color_"))
    stems = {p.stem: p for p in files}
    expected_stems = [Path(name).stem for name in expected_names]
    if set(expected_stems).issubset(stems):
        return {name: stems[Path(name).stem] for name in expected_names}
    if len(files) != len(expected_names):
        raise ValueError(f"{directory}: 有 {len(files)} 张 PNG，期望 {len(expected_names)} 张")
    return {name: path for name, path in zip(expected_names, files)}


def archive(method: str, scan: str, model: Path, mapping: dict, log_path: Path, gpu: int) -> None:
    dest = PROJECT / "output" / method / scan
    names = mapping["test_names"]
    if method == "Difix3D_10k":
        native = model / "renders/eval/ours_10000"
        if not native.exists():
            native = model / "renders/eval/ours_010000"
        render_dir, gt_dir = native / "renders", native / "gt"
        render_map, gt_map = paired_pngs(render_dir, names), paired_pngs(gt_dir, names)
        plys = sorted(model.glob("point_cloud/iteration_10000/*.ply")) or sorted(model.glob("*.ply"))
        if not plys:
            raise FileNotFoundError(f"{model}: 缺少 Difix 10k 点云")
        final_ply = plys[0]
        source_hash_verified_by = "command --mvs-ply; run.log records shared input hash"
    else:
        result_root = model / ("eval/ours_10000" if method in ("DropoutGS", "DNGaussian") else "test/ours_10000")
        render_dir, gt_dir = result_root / "renders", result_root / "gt"
        render_map, gt_map = paired_pngs(render_dir, names), paired_pngs(gt_dir, names)
        final_ply = model / "point_cloud/iteration_10000/point_cloud.ply"
        if not final_ply.is_file():
            raise FileNotFoundError(final_ply)
        input_ply = model / "input.ply"
        if not input_ply.is_file():
            raise FileNotFoundError(f"{method}/{scan}: missing input.ply; cannot verify shared MVS initialization")
        if sha256(input_ply) != mapping["initial_cloud_sha256"]:
            raise ValueError(f"{method}/{scan}: input.ply 与共享 MVS 初始化 hash 不符")
        source_hash_verified_by = "input.ply SHA-256"

    dest.mkdir(parents=True, exist_ok=True)
    render_dest = dest / "renders"
    if render_dest.exists():
        shutil.rmtree(render_dest)
    render_dest.mkdir(parents=True)
    gt_dest = Path(mapping["canonical_gt_dir"])
    if not gt_dest.is_dir():
        raise FileNotFoundError(f"{method}/{scan}: canonical GT not prepared at {gt_dest}")
    expected_gt_hashes = mapping["canonical_gt_sha256"]
    for name in names:
        stem = Path(name).stem + ".png"
        shutil.copy2(render_map[name], render_dest / stem)
        gt_path = gt_dest / stem
        if not gt_path.is_file() or sha256(gt_path) != expected_gt_hashes[stem]:
            raise ValueError(f"{scan}/{stem}: canonical GT hash mismatch")
        with Image.open(gt_path) as gt_image, Image.open(gt_map[name]) as method_gt, Image.open(render_map[name]) as render_image:
            if method_gt.size != gt_image.size or render_image.size != gt_image.size:
                raise ValueError(f"{method}/{scan}/{stem}: render/GT resolution differs from canonical GT")
            if method_gt.convert("RGB").tobytes() != gt_image.convert("RGB").tobytes():
                raise ValueError(f"{method}/{scan}/{stem}: 原生 GT 像素或相机顺序与 canonical GT 不一致")

    compressed = dest / "final_point_cloud.ply.gz"
    with final_ply.open("rb") as source, gzip.open(compressed, "wb", compresslevel=1) as target:
        shutil.copyfileobj(source, target)
    with gzip.open(compressed, "rb") as decoded:
        decoded_hash = hashlib.sha256()
        for block in iter(lambda: decoded.read(1 << 20), b""):
            decoded_hash.update(block)
    final_hash = sha256(final_ply)
    if decoded_hash.hexdigest() != final_hash:
        raise ValueError(f"{method}/{scan}: compressed point-cloud archive failed hash verification")
    checkpoint_archive = None
    if method in ("DNGaussian", "DropoutGS"):
        # 这两个方法的 PLY 不包含神经颜色网络，必须同时保留检查点才能复渲染。
        checkpoint = model / "chkpnt_latest.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{method}/{scan}: 神经颜色检查点缺失")
        inference_checkpoint = model / "chkpnt_latest_inference.pth"
        subprocess.run([str(PY_GS), str(CODE / "slim_neural_checkpoint.py"),
                        str(checkpoint), str(inference_checkpoint)], check=True)
        checkpoint_archive = dest / "chkpnt_latest_inference.pth.gz"
        with inference_checkpoint.open("rb") as source, gzip.open(checkpoint_archive, "wb", compresslevel=1) as target:
            shutil.copyfileobj(source, target)
    shutil.copy2(log_path, dest / "run.log")
    manifest = {"method": method, "scan": scan, "seed": 2024, "iterations": 10000, "gpu": gpu,
                "train_names": mapping["train_names"], "test_names": names,
                "initial_cloud": mapping["initial_cloud"], "initial_cloud_sha256": mapping["initial_cloud_sha256"],
                "initial_cloud_point_count": mapping["initial_cloud_point_count"],
                "input_hash_verification": source_hash_verified_by,
                "final_point_cloud": "final_point_cloud.ply.gz",
                "final_point_cloud_sha256": final_hash, "compressed_point_cloud_sha256": sha256(compressed),
                "render_count": len(render_map), "ground_truth_dir": str(gt_dest),
                "render_names": sorted(p.name for p in render_dest.glob("*.png")),
                "resolution_factor": 4, "checkpoint_archived": checkpoint_archive is not None}
    if mapping.get("config_profile"):
        manifest["config_profile"] = mapping["config_profile"]
    if checkpoint_archive is not None:
        manifest.update({"checkpoint": checkpoint_archive.name,
                         "checkpoint_sha256": sha256(inference_checkpoint),
                         "compressed_checkpoint_sha256": sha256(checkpoint_archive),
                         "checkpoint_inference_only": True})
    for config_name in ("cfg_args", "cfg.yml"):
        config_path = model / config_name
        if config_path.is_file():
            shutil.copy2(config_path, dest / config_name)
            manifest[config_name.replace(".", "_") + "_sha256"] = sha256(dest / config_name)
    (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (dest / "SUCCESS").write_text(f"method={method}\nscan={scan}\ngpu={gpu}\n")
    shutil.rmtree(model, ignore_errors=True)


def run(method: str, scan: str, gpu: int) -> None:
    if method not in ALLOWED:
        raise ValueError(f"不支持方法 {method}")
    dest = PROJECT / "output" / method / scan
    if (dest / "SUCCESS").is_file():
        return
    mapping = load_json(dest / "data_mapping.json")
    cloud = Path(mapping["initial_cloud"])
    if sha256(cloud) != mapping["initial_cloud_sha256"]:
        raise ValueError(f"{method}/{scan}: MVS 输入 hash 不匹配")
    source = TEMP / "overlay/DTU" / scan
    model = TEMP / "training" / method / scan
    if model.exists():
        shutil.rmtree(model)
    model.mkdir(parents=True, exist_ok=True)
    env, repo, python = env_for(method, gpu)
    if method in ("DropoutGS", "DNGaussian"):
        depth_dir = source / "depth_maps"
        expected = {"depth_" + Path(name).stem + ".png" for name in load_json(PROJECT / "data/splits/cor_gs_3view" / f"{scan}.json")["camera_names"]}
        if {p.name for p in depth_dir.glob("*.png")} != expected:
            raise FileNotFoundError(f"{scan}: DPT 深度先验不完整")
    log_path = TEMP / "logs/training" / f"{method}_{scan}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train, render = train_command(method, python, repo, source, model, mapping)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"START method={method} scan={scan} gpu={gpu} seed=2024 iterations=10000 time={time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
        log.write(f"source={source}\ninitial_cloud={cloud}\ninitial_cloud_sha256={sha256(cloud)}\n")
        run_command(train, repo, env, log)
        if render:
            run_command(render, repo, env, log)
        log.write(f"TRAIN_RENDER_COMPLETE method={method} scan={scan} gpu={gpu}\n")
    archive(method, scan, model, mapping, log_path, gpu)
    print(f"SUCCESS {method}/{scan} gpu={gpu}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(ALLOWED))
    parser.add_argument("--scan", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    run(args.method, args.scan, args.gpu)
