#!/usr/bin/env python3
"""FSGS official three-view initialization followed by fixed R4 evaluation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = Path("/data/lilinwei/3D/D23E/DO3/Scenedata/nerf_llff_data")
DATA = Path("/data/lilinwei/3D/D23E/Aref/FSGS_LLFF_data")
BASELINE = PROJECT / "output/r4_all_scenes"
TEMP = Path("/tmp/lilinwei/1007fsgsmvs")
OUTPUT = ROOT / "output"
INITIALIZATION_ARCHIVE = OUTPUT / "initialization" / "fsgs_official_mvs"
PYTHON = Path("/data/lilinwei/3D/1006difix/1007mast3r/envs/mast3r/bin/python")
TEACHER = Path("/data/lilinwei/3D/A_our_denoiser/20261002_164657_train_rgbdepth_teacher_perimg/checkpoint/iter_014000")
BASE_MODEL = Path("/data/lilinwei/3D/D23E")
COMPAT_SITE = Path("/data/lilinwei/.conda/envs/do3/clean_site")
DO3_SITE = Path("/data/lilinwei/3D/1006difix/1007mast3r/envs/mast3r/lib/python3.10/site-packages")
SCENES = ("fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex")
GPUS = tuple(int(x) for x in os.environ.get("MVS_R4_GPUS", "0,1,2,3,4,5,7").split(",") if x.strip())
if not GPUS or 6 in GPUS or len(set(GPUS)) != len(GPUS) or any(g not in range(8) for g in GPUS):
    raise ValueError("MVS_R4_GPUS must be unique GPU ids in 0-7 and exclude GPU 6")
METRICS = ("PSNR", "SSIM", "LPIPS")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ply_vertices(path: Path) -> int:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"PLY header has no vertex count: {path}")


def validate() -> dict[str, list[str]]:
    needed = (SOURCE, DATA, BASELINE / "metrics_summary.json", PYTHON, TEACHER / "unet",
              BASE_MODEL / "checkpoints/marigold-depth-v1-0", COMPAT_SITE, DO3_SITE,
              PROJECT / "submodules/simple-knn",
              PROJECT / "submodules/diff-gaussian-rasterization-confidence")
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    counts = {}
    for scene in SCENES:
        raw, canonical = SOURCE / scene, DATA / scene
        for p in (raw / "images", raw / "sparse/0/cameras.bin", raw / "sparse/0/images.bin",
                  canonical / "images_4", canonical / "sparse"):
            if not p.exists():
                raise FileNotFoundError(p)
        report = json.loads((canonical / "fsgs_do3_prepare_report.json").read_text())
        names = report.get("fsgs_train_names", [])
        if len(names) != 3 or not report.get("fsgs_test_names"):
            raise ValueError(f"invalid 3-view split in {scene}")
        if any(not (raw / "images" / n).is_file() for n in names):
            raise FileNotFoundError(f"selected train image missing for {scene}")
        counts[scene] = [Path(n).stem + ".png" for n in report["fsgs_test_names"]]
    if sum(len(x) for x in counts.values()) != 43:
        raise ValueError("expected exactly 43 evaluation images")
    return counts


def run_stage(name: str, scenes: tuple[str, ...], worker) -> None:
    errors = []
    available_gpus: Queue[int] = Queue()
    for gpu in GPUS:
        available_gpus.put(gpu)

    def invoke(scene: str) -> None:
        gpu = available_gpus.get()
        try:
            worker(scene, gpu)
        finally:
            available_gpus.put(gpu)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {pool.submit(invoke, scene): scene for scene in scenes}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                future.result()
                print(f"{name} complete: {scene}", flush=True)
            except Exception as e:
                errors.append({"scene": scene, "error": repr(e)})
                print(f"{name} FAILED: {scene}: {e}", flush=True)
    (TEMP / f"{name.lower()}_summary.json").write_text(json.dumps({
        "completed": [s for s in scenes if s not in {e['scene'] for e in errors}],
        "failed": errors, "updated_at": time.strftime("%F %T %z")}, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"{name} failed: {errors}")


def build_overlay(scene: str, mvs_ply: Path) -> None:
    source, overlay = DATA / scene, TEMP / "data_overlay" / scene
    if overlay.exists() and any(overlay.iterdir()):
        manifest = overlay / "mvs_overlay_manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("mvs_sha256") == sha256(mvs_ply):
            return
        raise FileExistsError(f"refusing to overwrite overlay {overlay}")
    overlay.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name != "3_views":
            (overlay / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
    dense = overlay / "3_views/dense"
    dense.mkdir(parents=True, exist_ok=True)
    (dense / "fused.ply").symlink_to(mvs_ply.resolve())
    report = json.loads((source / "fsgs_do3_prepare_report.json").read_text())
    (overlay / "mvs_overlay_manifest.json").write_text(json.dumps({
        "scene": scene, "mvs_ply": str(mvs_ply.resolve()), "mvs_sha256": sha256(mvs_ply),
        "point_count": ply_vertices(mvs_ply), "training_images": report["fsgs_train_names"],
        "images_4": str((overlay / "images_4").resolve()),
        "sparse_model": str((overlay / "sparse").resolve())}, indent=2) + "\n")


def archive_initialization_clouds(config: dict) -> None:
    """Keep reusable three-view FSGS MVS clouds locally before R4 training."""
    INITIALIZATION_ARCHIVE.mkdir(parents=True, exist_ok=True)
    method_manifest = {
        "method": "FSGS official three-view COLMAP MVS",
        "source_script": str(ROOT / "fsgs_official_colmap_llff_reference.py"),
        "per_scene_cloud": "<scene>/initial_point_cloud.ply",
        "point_clouds_are_reusable_by_other_methods": True,
        "initialization_protocol": config["initialization"],
        "scenes": {},
    }
    for scene in SCENES:
        source = TEMP / "colmap" / scene
        scene_archive = INITIALIZATION_ARCHIVE / scene
        scene_archive.mkdir(parents=True, exist_ok=True)
        source_manifest = json.loads((source / "fsgs_mvs_manifest.json").read_text())
        cloud_source = source / "fsgs_mvs_fused.ply"
        cloud_archive = scene_archive / "initial_point_cloud.ply"
        shutil.copy2(cloud_source, cloud_archive)
        archived_manifest = dict(source_manifest)
        archived_manifest["archived_point_cloud"] = str(cloud_archive)
        archived_manifest["archived_point_cloud_sha256"] = sha256(cloud_archive)
        archived_manifest["reuse_note"] = "This is the fused initialization PLY before R4 training; use it directly as another method's initialization."
        source_log = source / "fsgs_colmap.log"
        log_text = source_log.read_text(errors="replace") if source_log.is_file() else ""
        if "Creating SIFT CPU feature matcher" in log_text:
            archived_manifest["matching_backend"] = "CPU"
        elif "Creating SIFT GPU feature matcher" in log_text:
            archived_manifest["matching_backend"] = "GPU"
        archived_manifest["matching_settings"] = {
            "matcher": "exhaustive", "guided_matching": True, "max_num_matches": 32768,
            "use_gpu": archived_manifest.get("matching_backend") == "GPU",
        }
        (scene_archive / "manifest.json").write_text(json.dumps(archived_manifest, indent=2) + "\n")
        if source_log.is_file():
            shutil.copy2(source_log, scene_archive / "colmap.log")
        if (TEMP / "data_overlay" / scene / "mvs_overlay_manifest.json").is_file():
            shutil.copy2(TEMP / "data_overlay" / scene / "mvs_overlay_manifest.json",
                         scene_archive / "overlay_manifest.json")
        if sha256(cloud_archive) != source_manifest["point_cloud_sha256"]:
            raise ValueError(f"archived initial cloud hash mismatch for {scene}")
        method_manifest["scenes"][scene] = {
            "path": str(cloud_archive), "point_count": source_manifest["point_count"],
            "sha256": source_manifest["point_cloud_sha256"],
            "training_images": source_manifest["training_images"],
        }
    (INITIALIZATION_ARCHIVE / "method_manifest.json").write_text(
        json.dumps(method_manifest, indent=2) + "\n")
    (INITIALIZATION_ARCHIVE / "README.md").write_text(
        "# Reusable FSGS official MVS initialization clouds\n\n"
        "Each scene's `initial_point_cloud.ply` is the fused COLMAP MVS cloud generated "
        "from that scene's three selected training images, before R4 optimization. The "
        "supplied sparse model contributes camera parameters and selected poses; its "
        "existing 3D points are not used as the triangulator input. These PLY files are "
        "retained locally for reuse when rerunning other baselines. See `manifest.json`, "
        "`overlay_manifest.json`, and `colmap.log` per scene for provenance.\n")


def train_command(scene: str, model: Path, target_count: int) -> list[str]:
    return [str(PYTHON), "-u", str(ROOT / "train_r4_memory.py"),
            "--source_path", str(TEMP / "data_overlay" / scene), "--model_path", str(model),
            "--eval", "--n_views", "3", "--images", "images_4", "--resolution", "1",
            "--iterations", "10000", "--start_sample_pseudo", "2000", "--end_sample_pseudo", "9500",
            "--pseudo_update_interval", "200", "--pseudo_target_index", "0", "--depth_weight", "0.05",
            "--depth_pseudo_weight", "0.01", "--pseudo_rgb_weight", "0.35",
            "--joint_checkpoint", str(TEACHER), "--joint_code_root", str(PROJECT),
            "--joint_processing_res", "768", "--joint_denoise_steps", "4",
            "--pseudo_alpha_threshold", "0.01", "--pseudo_rgb_alpha_threshold", "0.01",
            "--seed", "2024", "--test_iterations", "5000", "10000",
            "--save_iterations", "5000", "10000", "--checkpoint_iterations", "5000", "10000",
            "--pseudo_training_mode", "difix", "--pseudo_target_source", "test",
            "--pseudo_target_count", str(target_count), "--pseudo_pose_step", "0.5",
            "--pseudo_real_probability", "0.7", "--pseudo_history_mode", "accumulate"]


def train_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join(map(str, [COMPAT_SITE, DO3_SITE,
                 PROJECT / "submodules/simple-knn", PROJECT / "submodules/diff-gaussian-rasterization-confidence"])),
                "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                "JOINT_BASE_CKPT_DIR": str(BASE_MODEL), "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return env


def main() -> int:
    counts = validate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT.iterdir())
    if any(path.name != "initialization" for path in existing):
        raise FileExistsError(f"final output contains prior results; refusing overwrite: {OUTPUT}")
    initialization_root = OUTPUT / "initialization"
    if initialization_root.exists() and any(
            path.name != "fsgs_official_mvs" for path in initialization_root.iterdir()):
        raise FileExistsError(f"initialization contains unrelated prior results: {initialization_root}")
    TEMP.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text())
    config["actual_gpu_pool"] = list(GPUS)
    config["pytorch_cuda_allocator"] = {
        "first_attempt": "default",
        "oom_retry": "expandable_segments:True (runtime memory fragmentation mitigation; training hyperparameters unchanged)",
    }
    config["teacher_memory_mode"] = (
        "experiment-local adapter: keep frozen joint RGBD teacher on CPU between pseudo-label refreshes; "
        "move to training GPU only for prediction, then offload before Gaussian loss backward"
    )
    config["knn_memory_mode"] = (
        "experiment-local CPU cKDTree implementation of the same distCUDA2 contract: "
        "mean squared distance and indices of each point's three nearest other points; "
        "verified against CUDA simple-knn on the leaves initialization cloud"
    )
    config["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    config["source_sparse_sha256"] = {s: {n: sha256(SOURCE / s / "sparse/0" / n)
        for n in ("cameras.bin", "images.bin", "points3D.bin")} for s in SCENES}
    (TEMP / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def mvs_worker(scene: str, gpu: int) -> None:
        root = TEMP / "colmap" / scene
        log = TEMP / "logs" / f"fsgs_mvs_{scene}.launcher.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(PYTHON), "-u", str(ROOT / "run_scene_fsgs_mvs.py"), "--scene", scene, "--gpu", str(gpu)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log.open("a") as f:
            f.write(f"\nphysical_gpu={gpu}; command={' '.join(cmd)}\n")
            f.flush()
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"see {log}")
        manifest = json.loads((root / "fsgs_mvs_manifest.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("point_count", 0) <= 0:
            raise RuntimeError(f"invalid MVS result for {scene}")

    run_stage("FSGS_MVS", SCENES, mvs_worker)
    for scene in SCENES:
        build_overlay(scene, TEMP / "colmap" / scene / "fsgs_mvs_fused.ply")

    def training_worker(scene: str, gpu: int) -> None:
        model = TEMP / "r4_training" / scene
        model.mkdir(parents=True, exist_ok=True)
        log = model / "train.log"
        if log.is_file():
            old = log.read_text(errors="replace")
            if "Training complete." in old and (model / "point_cloud/iteration_10000/point_cloud.ply").is_file():
                return
            if "out of memory" not in old.lower() and "oom" not in old.lower():
                raise FileExistsError(f"incomplete run without a recorded OOM: {model}")
            failed_root = TEMP / "failed_attempts" / scene
            failed_root.mkdir(parents=True, exist_ok=True)
            failed_model = failed_root / time.strftime("attempt_%Y%m%d_%H%M%S")
            shutil.move(str(model), str(failed_model))
            print(f"preserved OOM attempt at {failed_model}; retrying with unchanged R4 arguments", flush=True)
            model.mkdir(parents=True, exist_ok=True)
        cmd = train_command(scene, model, len(counts[scene]))
        with log.open("w") as f:
            f.write(f"physical_gpu={gpu}\n$ {' '.join(cmd)}\n")
            f.flush()
            result = subprocess.run(cmd, cwd=PROJECT, env=train_env(gpu), stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"training exit={result.returncode}; see {log}")
        if "Training complete." not in log.read_text(errors="replace") or not \
                (model / "point_cloud/iteration_10000/point_cloud.ply").is_file():
            raise RuntimeError(f"training output missing: {model}")
        summary = json.loads((model / "joint_summary.json").read_text())
        required = {"iterations": 10000, "pseudo_update_interval": 200, "pseudo_rgb_weight": 0.35,
                    "pseudo_depth_weight": 0.01, "pseudo_history_mode": "accumulate",
                    "pseudo_pose_step": 0.5, "pseudo_real_probability": 0.7,
                    "target_camera_source": "test", "target_camera_count": len(counts[scene])}
        if any(summary.get(k) != v for k, v in required.items()):
            raise ValueError(f"R4 config mismatch for {scene}: {summary}")

    run_stage("R4_TRAINING", SCENES, training_worker)

    def render_worker(scene: str, gpu: int) -> None:
        model = TEMP / "r4_training" / scene
        expected = set(counts[scene])
        renders, gt = model / "test/ours_10000/renders", model / "test/ours_10000/gt"
        expected_depth_png = {Path(n).stem + "_depth.png" for n in expected}
        expected_depth_npy = {Path(n).stem + "_depth.npy" for n in expected}
        rgb_names = {p.name for p in renders.glob("*.png") if not p.name.endswith("_depth.png")}
        if (rgb_names == expected and expected_depth_png <= {p.name for p in renders.glob("*.png")}
                and expected_depth_npy <= {p.name for p in renders.glob("*.npy")}
                and set(p.name for p in gt.glob("*.png")) == expected):
            return
        cmd = [str(PYTHON), "-u", str(PROJECT / "render.py"), "--source_path",
               str(TEMP / "data_overlay" / scene), "--model_path", str(model),
               "--iteration", "10000", "--skip_train", "--render_depth"]
        log = model / "render.log"
        with log.open("w") as f:
            f.write(f"physical_gpu={gpu}\n$ {' '.join(cmd)}\n")
            f.flush()
            result = subprocess.run(cmd, cwd=PROJECT, env=train_env(gpu), stdout=f, stderr=subprocess.STDOUT)
        rgb_names = {p.name for p in renders.glob("*.png") if not p.name.endswith("_depth.png")}
        if result.returncode or rgb_names != expected or \
                not expected_depth_png <= {p.name for p in renders.glob("*.png")} or \
                not expected_depth_npy <= {p.name for p in renders.glob("*.npy")} or \
                set(p.name for p in gt.glob("*.png")) != expected:
            raise RuntimeError(f"render incomplete; see {log}")

    run_stage("RENDER", SCENES, render_worker)
    metric_cmd = [str(PYTHON), "-u", str(PROJECT / "metrics.py"), "--source_paths",
                  *[str(TEMP / "data_overlay" / s) for s in SCENES], "--model_paths",
                  *[str(TEMP / "r4_training" / s) for s in SCENES]]
    metric_log = TEMP / "metrics.log"
    with metric_log.open("w") as f:
        result = subprocess.run(metric_cmd, cwd=PROJECT, env=train_env(GPUS[0]), stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"metrics failed; see {metric_log}")
    archive(counts, config)
    cleanup_intermediates()
    print(f"pipeline complete: {OUTPUT / 'summary.md'}", flush=True)
    return 0


def archive(counts: dict[str, list[str]], config: dict) -> None:
    archive_initialization_clouds(config)
    baseline_summary = json.loads((BASELINE / "metrics_summary.json").read_text())
    baseline_summary = baseline_summary["results_by_method"]["R4 accumulated pseudo views"]
    rows, wins = {}, {m: 0 for m in METRICS}
    total = 0
    for scene in SCENES:
        model, base = TEMP / "r4_training" / scene, BASELINE / scene
        new_mean = json.loads((model / "results.json").read_text())["ours_10000"]
        new_views = json.loads((model / "per_view.json").read_text())["ours_10000"]
        old_mean = json.loads((base / "results.json").read_text())["ours_10000"]
        old_views = json.loads((base / "per_view.json").read_text())["ours_10000"]
        expected = set(counts[scene])
        if any(set(x["PSNR"]) != expected for x in (new_views, old_views)):
            raise ValueError(f"paired evaluation view mismatch for {scene}")
        for m in METRICS:
            if abs(float(old_mean[m]) - float(baseline_summary[scene][m])) > 1e-5:
                raise ValueError(f"baseline {scene}/{m} does not match stored summary")
            wins[m] += int(float(new_mean[m]) < float(old_mean[m]) if m == "LPIPS"
                           else float(new_mean[m]) > float(old_mean[m]))
        manifest = json.loads((TEMP / "colmap" / scene / "fsgs_mvs_manifest.json").read_text())
        rows[scene] = {"heldout_views": len(expected), "baseline_r4": {m: float(old_mean[m]) for m in METRICS},
            "mvs_init_r4": {m: float(new_mean[m]) for m in METRICS},
            "delta_mvs_minus_baseline": {m: float(new_mean[m]) - float(old_mean[m]) for m in METRICS},
            "mvs_initial_points": manifest["point_count"], "initial_cloud_sha256": manifest["point_cloud_sha256"],
            "per_view": {n: {"baseline_r4": {m: float(old_views[m][n]) for m in METRICS},
                             "mvs_init_r4": {m: float(new_views[m][n]) for m in METRICS}} for n in sorted(expected)}}
        total += len(expected)
        dst = OUTPUT / scene
        dst.mkdir()
        for name in ("joint_summary.json", "pseudo_stats.jsonl", "cfg_args", "cameras.json", "results.json", "per_view.json"):
            if (model / name).is_file():
                shutil.copy2(model / name, dst / name)
        for rel in ("point_cloud/iteration_10000/point_cloud.ply", "test/ours_10000/renders", "test/ours_10000/gt"):
            src, target = model / rel, dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, target) if src.is_dir() else shutil.copy2(src, target)
    if len(rows) != 8 or total != 43:
        raise ValueError(f"expected 8 scenes and 43 views; got {len(rows)} and {total}")
    macro = {method: {m: sum(rows[s][method][m] for s in SCENES) / len(SCENES) for m in METRICS}
             for method in ("baseline_r4", "mvs_init_r4")}
    delta = {m: macro["mvs_init_r4"][m] - macro["baseline_r4"][m] for m in METRICS}
    report = {"protocol": {"scenes": list(SCENES), "heldout_images": total,
               "aggregation": "within-scene image mean; equal-weight mean over eight scenes",
               "mvs_inputs": "exactly three selected training images per scene",
               "camera_poses": "supplied COLMAP sparse model", "r4_seed": 2024},
              "scene_results": rows, "macro_average": macro,
              "macro_delta_mvs_minus_baseline": delta, "scene_wins_by_metric": wins}
    (OUTPUT / "metrics_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")
    fmt = lambda d: f"{d['PSNR']:.4f} / {d['SSIM']:.4f} / {d['LPIPS']:.4f}"
    lines = ["# FSGS official COLMAP initialization × R4", "", "## Protocol", "",
        "Initialization uses a copied FSGS `tools/colmap_llff.py` with only an isolated input root and per-scene command-line entry. Its COLMAP feature extraction, matching, three-view triangulation, undistortion, PatchMatch, and fusion commands are kept intact. R4 uses images_4 training resolution, the same parameters, target-camera protocol, held-out images, and metric evaluator as the existing R4 reference. Eight scene means are equally weighted; the 43 held-out views are paired by filename. Camera poses are inherited from the supplied COLMAP sparse model; the triangulated and dense points are generated from the selected three views.", "",
        "## Results", "", "| Scene | Test views | Baseline R4 PSNR / SSIM / LPIPS | MVS-init R4 PSNR / SSIM / LPIPS | Δ PSNR / SSIM / LPIPS | MVS points |", "|---|---:|---:|---:|---:|---:|"]
    for s in SCENES:
        r, d = rows[s], rows[s]["delta_mvs_minus_baseline"]
        lines.append(f"| {s} | {r['heldout_views']} | {fmt(r['baseline_r4'])} | {fmt(r['mvs_init_r4'])} | {d['PSNR']:+.4f} / {d['SSIM']:+.4f} / {d['LPIPS']:+.4f} | {r['mvs_initial_points']:,} |")
    lines += [f"| **Eight-scene mean** | **{total}** | **{fmt(macro['baseline_r4'])}** | **{fmt(macro['mvs_init_r4'])}** | **{delta['PSNR']:+.4f} / {delta['SSIM']:+.4f} / {delta['LPIPS']:+.4f}** | — |", "",
        f"FSGS initialization wins in PSNR/SSIM/LPIPS on {wins['PSNR']}/8, {wins['SSIM']}/8, and {wins['LPIPS']}/8 scenes. Image-paired scores are in `metrics_comparison.json`. Initial clouds, final clouds, RGB renders, rendered depth PNG/NPY maps, and ground truth images are archived here. Intermediate COLMAP data are removed after each successful scene; training checkpoints and duplicate temporary renders are removed after archival, while logs and manifests remain in `/tmp/lilinwei/1007fsgsmvs`.", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def cleanup_intermediates() -> None:
    """Free duplicate clouds, render trees, and checkpoints after final archive."""
    for scene in SCENES:
        model = TEMP / "r4_training" / scene
        for rel in ("checkpoint", "point_cloud", "test"):
            path = model / rel
            if path.exists():
                shutil.rmtree(path)
        temporary_cloud = TEMP / "colmap" / scene / "fsgs_mvs_fused.ply"
        if temporary_cloud.exists():
            temporary_cloud.unlink()
        overlay = TEMP / "data_overlay" / scene
        if overlay.exists():
            shutil.rmtree(overlay)


if __name__ == "__main__":
    raise SystemExit(main())
