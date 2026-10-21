#!/usr/bin/env python3
"""Score Difix3D raw and repaired renders against the shared Mip-NeRF 360 GT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf

ROOT = Path("/data/lilinwei/3D/baselines/output/1009_fsgs_mvs_mipnerf")
REPO = Path("/data/lilinwei/3D/1006difix")
SCENES = ("bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump")
METRICS = ("PSNR", "SSIM", "LPIPS")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


def check_shared_inputs(scene: str, expected: list[str], scene_root: Path) -> None:
    difix_map = json.loads((scene_root / "data_mapping.json").read_text())
    fsgs_map = json.loads((ROOT / "FSGS" / scene / "data_mapping.json").read_text())
    if difix_map["test_names"] != fsgs_map["test_names"]:
        raise ValueError(f"{scene}: Difix and FSGS test-view order differs")
    if difix_map["initial_cloud_sha256"] != fsgs_map["initial_cloud_sha256"]:
        raise ValueError(f"{scene}: Difix and FSGS MVS hashes differ")
    if [Path(x).stem + ".png" for x in difix_map["test_names"]] != expected:
        raise ValueError(f"{scene}: archived test names differ from data mapping")


def score_folder(folder: Path, expected: list[str], gt_dir: Path, device: torch.device,
                 psnr_fn, ssim_fn, lpips_fn) -> tuple[dict, dict]:
    actual = sorted(p.name for p in folder.glob("*.png"))
    if actual != expected:
        raise ValueError(f"{folder}: PNG names differ; got {len(actual)} expected {len(expected)}")
    per_view = {metric: {} for metric in METRICS}
    for name in expected:
        pred_path, gt_path = folder / name, gt_dir / name
        with Image.open(pred_path) as pred_img, Image.open(gt_path) as gt_img:
            if pred_img.size != gt_img.size:
                raise ValueError(f"{folder.parent.name}/{name}: pred/GT sizes differ")
        pred = image_tensor(pred_path, device)
        gt = image_tensor(gt_path, device)
        with torch.no_grad():
            per_view["PSNR"][name] = float(psnr_fn(pred, gt).mean().item())
            per_view["SSIM"][name] = float(ssim_fn(pred, gt).item())
            per_view["LPIPS"][name] = float(lpips_fn(pred, gt, net_type="vgg").mean().item())
    means = {metric: float(np.mean(list(values.values()))) for metric, values in per_view.items()}
    return means, per_view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Use CPU by default; GPU 6 is intentionally not used.")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Difix3D_10k",
                        help="Override output location for partial metric checks.")
    args = parser.parse_args()

    if any(scene not in SCENES for scene in args.scenes):
        raise ValueError(f"Unknown scene in {args.scenes}")
    sys.path.insert(0, str(REPO))
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    results = {"raw": {}, "repaired": {}}
    per_view_all = {"raw": {}, "repaired": {}}
    for scene in args.scenes:
        scene_root = ROOT / "Difix3D_10k" / scene
        if not (scene_root / "SUCCESS").is_file():
            raise FileNotFoundError(f"Missing SUCCESS marker for Difix3D_10k/{scene}")
        mapping = json.loads((scene_root / "data_mapping.json").read_text())
        expected = sorted(Path(name).stem + ".png" for name in mapping["test_names"])
        check_shared_inputs(scene, expected, scene_root)
        shared_gt = ROOT / "our_method" / scene / "ground_truth"
        archived_gt = scene_root / "ground_truth"
        if sorted(p.name for p in shared_gt.glob("*.png")) != expected:
            raise ValueError(f"{scene}: canonical shared GT set differs")
        if sorted(p.name for p in archived_gt.glob("*.png")) != expected:
            raise ValueError(f"{scene}: Difix archived GT set differs")
        for name in expected:
            with Image.open(shared_gt / name) as shared, Image.open(archived_gt / name) as archived:
                if shared.convert("RGB").tobytes() != archived.convert("RGB").tobytes():
                    raise ValueError(f"{scene}/{name}: archived GT pixels differ from shared R4 GT")

        for label, folder in (("raw", scene_root / "renders"),
                              ("repaired", scene_root / "difix_renders")):
            means, per_view = score_folder(folder, expected, shared_gt, device, psnr, ssim, lpips)
            results[label][scene] = {"views": len(expected), "metrics": means,
                                     "initial_cloud_sha256": mapping["initial_cloud_sha256"],
                                     "final_point_cloud_sha256": sha256(scene_root / "final_point_cloud.ply")}
            per_view_all[label][scene] = per_view
            print(f"{label}\t{scene}\t{len(expected)}\t{means['PSNR']:.6f}\t"
                  f"{means['SSIM']:.6f}\t{means['LPIPS']:.6f}", flush=True)

    total_views = sum(results["raw"][scene]["views"] for scene in args.scenes)
    for label in results:
        results[label]["seven_scene_macro_average"] = {
            metric: float(np.mean([results[label][scene]["metrics"][metric] for scene in args.scenes]))
            for metric in METRICS
        }
        results[label]["view_weighted_average"] = {
            metric: float(np.mean([value for scene in args.scenes
                                   for value in per_view_all[label][scene][metric].values()]))
            for metric in METRICS
        }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "Shared seven-scene FSGS hold8/12-view split, seed 2024; shared R4 ground-truth pixels",
               "scenes_scored": args.scenes, "total_test_views": total_views,
               "results": results, "per_view": per_view_all}
    (output / "metrics_shared_gt.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Difix3D 10k on Mip-NeRF 360", "",
             "All scores use the same held-out image names and pixel-identical canonical GT as R4. "
             "The `raw` rows are the Gaussian renderer output; `repaired` rows are the diffusion-refined images.",
             "", "| Scene | Views | Raw PSNR | Raw SSIM | Raw LPIPS | Repaired PSNR | Repaired SSIM | Repaired LPIPS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scene in args.scenes:
        raw, repaired = results["raw"][scene], results["repaired"][scene]
        rm, fm = raw["metrics"], repaired["metrics"]
        lines.append(f"| {scene} | {raw['views']} | {rm['PSNR']:.4f} | {rm['SSIM']:.4f} | {rm['LPIPS']:.4f} | "
                     f"{fm['PSNR']:.4f} | {fm['SSIM']:.4f} | {fm['LPIPS']:.4f} |")
    for label, title in (("raw", "Raw macro mean"), ("repaired", "Repaired macro mean")):
        m = results[label]["seven_scene_macro_average"]
        lines.append(f"| **{title}** | **{total_views}** | **{m['PSNR']:.4f}** | **{m['SSIM']:.4f}** | **{m['LPIPS']:.4f}** | | | |")
    lines += ["", "## View-weighted averages", "",
              "| Render set | PSNR | SSIM | LPIPS |", "|---|---:|---:|---:|"]
    for label in ("raw", "repaired"):
        m = results[label]["view_weighted_average"]
        lines.append(f"| {label} | {m['PSNR']:.4f} | {m['SSIM']:.4f} | {m['LPIPS']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {output / 'metrics_shared_gt.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit("此文件是 1009 MipNeRF 指标脚本副本；DTU 实验请使用 DTU 专用评测入口。")
