#!/usr/bin/env python3
"""Compute shared held-out metrics for the official-FSGS-MVS baseline reruns.

Run with the d23e environment and CoR-GS on PYTHONPATH (for lpipsPyTorch).
This intentionally reads only archived outputs, never temporary scratch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf

ROOT = Path("/data/lilinwei/3D/baselines/output/1007_fsgs_mvs_baselines")
INPUT_MANIFEST = ROOT / "configs/input_manifest.json"
CANONICAL_MVS_GT = Path("/data/lilinwei/3D/1006difix/1007fsgsmvs/output")
METHOD_STEPS = {
    "FSGS": 10000,
    "CoR-GS": 10000,
    "DNGaussian": 10000,
    "DropoutGS": 10000,
    "Difix3D_10k": 10000,
}


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3, :, :].to(device)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_common_gt(scene: str, test_names: list[str]) -> dict[str, Path]:
    """Copy the exact canonical 1007mvs GT files used by our method."""
    source = CANONICAL_MVS_GT / scene / "test/ours_10000/gt"
    target = ROOT / "common_gt" / scene
    target.mkdir(parents=True, exist_ok=True)
    result = {}
    for test_name in test_names:
        name = Path(test_name).stem + ".png"
        src, dst = source / name, target / name
        if not src.is_file():
            raise FileNotFoundError(f"canonical 1007mvs GT is missing: {src}")
        if dst.exists() and sha256(dst) != sha256(src):
            raise ValueError(f"archived common GT differs from 1007mvs: {dst}")
        if not dst.exists():
            shutil.copy2(src, dst)
        result[test_name] = dst
    return result


def pair_renders(render_dir: Path, test_names: list[str]) -> list[tuple[str, str, Path]]:
    """Map each renderer filename to the common COLMAP test image name."""
    renders = {p.name.lower(): p for p in render_dir.glob("*.png")}
    pairs = []
    used = set()
    for idx, test_name in enumerate(test_names):
        stem = Path(test_name).stem.lower()
        candidates = [
            f"{stem}.png",
            f"{idx:05d}.png",
            f"{idx:04d}.png",
            f"{idx}.png",
        ]
        matches = [renders[name] for name in candidates if name in renders]
        matches = list({p.name: p for p in matches}.values())
        if len(matches) != 1 or matches[0].name in used:
            raise ValueError(f"cannot uniquely map render to test view {test_name} in {render_dir}")
        render = matches[0]
        used.add(render.name)
        pairs.append((render.name, test_name, render))
    if used != {p.name for p in renders.values()}:
        raise ValueError(f"unexpected or extra renders in {render_dir}: {set(p.name for p in renders.values()) - used}")
    return pairs


def score_views(pairs, common_gt: dict[str, Path], device: torch.device, psnr_fn, ssim_fn, lpips_fn):
    psnr_values, ssim_values, lpips_values = [], [], []
    mapping = []
    for render_name, test_name, render_path in pairs:
        pred = image_tensor(render_path, device)
        gt = image_tensor(common_gt[test_name], device)
        if pred.shape != gt.shape:
            raise ValueError(f"dimension mismatch for {render_name} / {test_name}: {pred.shape} vs {gt.shape}")
        psnr_values.append(float(psnr_fn(pred, gt).mean()))
        ssim_values.append(float(ssim_fn(pred, gt)))
        lpips_values.append(float(lpips_fn(pred, gt, net_type="vgg")))
    return {
        "n_views": len(pairs),
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
        "view_mapping": [{"render": render_name, "canonical_test_image": test_name} for render_name, test_name, _ in pairs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", nargs="*", default=list(METHOD_STEPS))
    parser.add_argument("--require-all", action="store_true", help="fail if any requested scene is missing")
    args = parser.parse_args()

    project_repo = Path("/data/lilinwei/3D/1006difix")
    sys.path.insert(0, str(project_repo))
    from utils.image_utils import psnr as psnr_fn
    from utils.loss_utils import ssim as ssim_fn
    from lpipsPyTorch import lpips

    device = torch.device(args.device)
    manifest = json.loads(INPUT_MANIFEST.read_text())
    scenes = list(manifest["scenes"])
    combined = {
        "protocol": {
            "input_manifest": str(INPUT_MANIFEST),
            "canonical_gt_source": str(CANONICAL_MVS_GT / "<scene>/test/ours_10000/gt"),
            "scenes": scenes,
            "per_scene_average": "mean over all paired held-out views",
            "macro_average": "unweighted mean over scenes",
            "rgb_range": "[0,1] from 8-bit RGB PNG",
            "metrics_implementation": "/data/lilinwei/3D/1006difix/metrics.py",
            "PSNR": "utils.image_utils.psnr from 1006difix (20*log10(1/sqrt(MSE)))",
            "SSIM": "utils.loss_utils.ssim from 1006difix (Graphdeco 11x11 Gaussian window)",
            "LPIPS": "lpipsPyTorch VGG from 1006difix",
            "device": str(device),
        },
        "results": {},
    }
    missing = []
    for method in args.methods:
        step = METHOD_STEPS[method]
        scene_results = {}
        for scene in scenes:
            base = ROOT / method / "outputs" / scene / "eval" / f"ours_{step}"
            render_dir = base / "renders"
            expected = int(manifest["scenes"][scene]["eval_count"])
            mapping_path = ROOT / method / "data_mapping" / f"{scene}.json"
            if not render_dir.is_dir() or not mapping_path.is_file():
                missing.append(f"{method}/{scene}: missing render or data mapping")
                continue
            try:
                mapping = json.loads(mapping_path.read_text())
                test_names = mapping["test_names"]
                if len(test_names) != expected:
                    raise ValueError(f"data mapping has {len(test_names)} test names; expected {expected}")
                common_gt = prepare_common_gt(scene, test_names)
                pairs = pair_renders(render_dir, test_names)
                scene_results[scene] = score_views(pairs, common_gt, device, psnr_fn, ssim_fn, lpips)
                values = scene_results[scene]
                print(f"{method}\t{scene}\t{values['n_views']}\t{values['PSNR']:.6f}\t{values['SSIM']:.6f}\t{values['LPIPS']:.6f}", flush=True)
            except Exception as exc:
                missing.append(f"{method}/{scene}: {exc}")
        summary = {"results": scene_results}
        if scene_results:
            summary["macro"] = {
                metric: float(np.mean([record[metric] for record in scene_results.values()]))
                for metric in ("PSNR", "SSIM", "LPIPS")
            }
        combined["results"][method] = summary
        method_dir = ROOT / method / "metrics"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "public_common_eval.json").write_text(json.dumps({**combined["protocol"], **summary, "method": method}, indent=2) + "\n")
        if "macro" in summary:
            (method_dir / "public_common_macro.json").write_text(json.dumps(summary["macro"], indent=2) + "\n")

    combined["missing_or_invalid"] = missing
    (ROOT / "UNIFIED_METRICS.json").write_text(json.dumps(combined, indent=2) + "\n")
    with (ROOT / "UNIFIED_METRICS.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_scenes", "n_views", "PSNR", "SSIM", "LPIPS"])
        for method, summary in combined["results"].items():
            macro = summary.get("macro", {})
            records = summary["results"]
            writer.writerow([
                method,
                len(records),
                sum(v["n_views"] for v in records.values()),
                macro.get("PSNR", ""),
                macro.get("SSIM", ""),
                macro.get("LPIPS", ""),
            ])
    if missing:
        print("INCOMPLETE:", *missing, sep="\n- ", file=sys.stderr)
        if args.require_all:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
