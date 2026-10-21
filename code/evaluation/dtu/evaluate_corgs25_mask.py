#!/usr/bin/env python3
"""用 CoR-GS 的 DTU 25 视角与物体 mask 公式评估已保存渲染。"""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
CONFIG = json.loads((CODE / "experiment_config.json").read_text())
MASK_ZIP = PROJECT / "data/eval_masks/corgs_25view_masks.zip"
MASK_SHA256 = "988f1a1957083191b45f7d4ea55a0b74566fc4286f1c224be9d19aa2a143ffbd"
SOURCE_MASK_SHA256 = "cd65d10cc0126e3f266a0f30e407c26fda3ecf61ce13318613261d9dfeef8b23"
RESERVED_NINE = {25, 22, 28, 40, 44, 48, 0, 8, 13}
EXCLUDED_FIFTEEN = set(CONFIG["excluded_indices"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def official_names(scan: str) -> list[str]:
    split = json.loads((PROJECT / "data/splits/cor_gs_3view" / f"{scan}.json").read_text())
    names = sorted(record["name"] for record in split["image_records"])
    if len(names) != 49 or len(set(names)) != 49:
        raise ValueError(f"{scan}: expected 49 unique camera names")
    if {names[i] for i in (25, 22, 28)} != set(split["train_names"]):
        raise ValueError(f"{scan}: three training names differ from CoR-GS")
    test = [names[i] for i in range(49) if i not in RESERVED_NINE | EXCLUDED_FIFTEEN]
    if len(test) != 25 or not set(test) <= set(split["test_names"]):
        raise ValueError(f"{scan}: upstream 25-view split does not match saved renders")
    return test


def mask_path(archive: ZipFile, scan: str, name: str) -> str:
    frame = int(re.fullmatch(r"rect_(\d+)_3_r5000\.png", name).group(1)) - 1
    prefix = f"submission_data/idrmasks/{scan}/"
    candidates = (prefix + f"{frame:03d}.png", prefix + "mask/" + f"{frame:03d}.png")
    found = [candidate for candidate in candidates if candidate in archive.NameToInfo]
    if len(found) != 1:
        raise ValueError(f"{scan}/{name}: expected exactly one matching mask, got {found}")
    return found[0]


def load_mask(archive: ZipFile, scan: str, name: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(BytesIO(archive.read(mask_path(archive, scan, name)))) as im:
        if im.mode not in ("RGB", "RGBA") or im.size != (1600, 1200):
            raise ValueError(f"{scan}/{name}: unexpected mask mode/size: {im.mode}/{im.size}")
        # CoR-GS metrics_dtu.py calls PIL Image.resize(gt.size) with no resample argument.
        # It then keeps only the first three channels, including for RGBA masks.
        resized = np.asarray(im.resize(size), dtype=np.float32)[:, :, :3] / 255.0
    return resized


def aggregate(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    result = {}
    for method in CONFIG["methods"]:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(CONFIG["scans"]):
            raise ValueError(f"{method}: incomplete scan coverage")
        result[method] = {metric: {"mean": mean(row[metric] for row in selected),
                                    "scene_sample_std": stdev(row[metric] for row in selected)}
                          for metric in metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psnr-only", action="store_true", help="CPU 诊断；仅计算 CoR-GS mask 内 PSNR")
    args = parser.parse_args()
    if sha256(MASK_ZIP) != MASK_SHA256:
        raise ValueError("mask ZIP SHA-256 mismatch")

    torch = None
    if not args.psnr_only:
        import torch as torch_module
        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("完整 CoR-GS 指标评估需要空闲的非 6 号 GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("请通过 CUDA_VISIBLE_DEVICES 只暴露一张允许使用的 GPU")
        sys.path.insert(0, str(CODE / "methods/CoR-GS"))
        from utils.loss_utils import ssim as corgs_ssim
        from lpipsPyTorch.modules.lpips import LPIPS
        lpips_model = LPIPS("vgg", "0.1").cuda().eval()

    rows = []
    outdir = PROJECT / "output/metrics"
    outdir.mkdir(exist_ok=True)
    with ZipFile(MASK_ZIP) as archive:
        for scan in CONFIG["scans"]:
            test_names = official_names(scan)
            masks = {}
            gts = {}
            for name in test_names:
                gt = load_rgb(PROJECT / "output/ground_truth" / scan / name)
                mask = load_mask(archive, scan, name, (gt.shape[1], gt.shape[0]))
                masks[name] = mask
                gts[name] = gt
            for method in CONFIG["methods"]:
                values = []
                for name in test_names:
                    gt, mask = gts[name], masks[name]
                    render = load_rgb(PROJECT / "output" / method / scan / "renders" / name)
                    if render.shape != gt.shape:
                        raise ValueError(f"{method}/{scan}/{name}: render/GT size mismatch")
                    # Upstream uses (mask == 1.) on all three RGB channels,
                    # including masks with slightly different channel edges.
                    mask_bin = mask == 1.0
                    if not mask_bin.any():
                        raise ValueError(f"{scan}/{name}: empty strict foreground mask")
                    mse = float(np.mean((render[mask_bin] - gt[mask_bin]) ** 2))
                    if mse <= 0:
                        raise ValueError(f"{method}/{scan}/{name}: invalid masked MSE {mse}")
                    record = {"view": name, "mask_member": mask_path(archive, scan, name),
                              "mask_fraction": float(mask_bin.mean()),
                              "PSNR": -10.0 * math.log10(mse)}
                    if torch is not None:
                        # The upstream metrics_dtu.py first whitens both images outside
                        # the mask, then uses its 3DGS SSIM and VGG LPIPS implementation.
                        masked_render = render * mask + (1.0 - mask)
                        masked_gt = gt * mask + (1.0 - mask)
                        a = torch.from_numpy(masked_render.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                        b = torch.from_numpy(masked_gt.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                        with torch.no_grad():
                            record["SSIM"] = float(corgs_ssim(a, b).item())
                            # Original CoR-GS passes [0,1] masked tensors to this LPIPS.
                            record["LPIPS"] = float(lpips_model(a, b).mean().item())
                    values.append(record)
                metrics = ("PSNR",) if args.psnr_only else ("PSNR", "SSIM", "LPIPS")
                row = {"method": method, "scan": scan, "view_count": 25,
                       **{key: mean(v[key] for v in values) for key in metrics}}
                rows.append(row)
                if not args.psnr_only:
                    out = PROJECT / "output" / method / scan / "metrics_corgs25_mask.json"
                    out.write_text(json.dumps({"protocol": "CoR-GS upstream DTU 25-view + object mask",
                                               "mask_archive_sha256": MASK_SHA256,
                                               "source_mask_archive_sha256": SOURCE_MASK_SHA256,
                                               "mean": {key: row[key] for key in metrics},
                                               "per_view": values}, ensure_ascii=False, indent=2) + "\n")
                print(method, scan, " ".join(f"{key}={row[key]:.4f}" for key in metrics), flush=True)

    suffix = "corgs25_mask_psnr" if args.psnr_only else "corgs25_mask"
    metrics = ("PSNR",) if args.psnr_only else ("PSNR", "SSIM", "LPIPS")
    with (outdir / f"per_scene_{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "scan", "view_count", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"protocol": "CoR-GS source-code DTU split: 25 test views per scan",
               "metric_code": "CoR-GS metrics_dtu.py: foreground PSNR; white-background 3DGS SSIM and VGG LPIPS",
               "mask_source": "DNGaussian official RegNeRF mask backup: submission_data.zip",
               "mask_archive_sha256": MASK_SHA256,
               "source_mask_archive_sha256": SOURCE_MASK_SHA256,
               "scene_count": len(CONFIG["scans"]), "views_per_scene": 25,
               "method_count": len(CONFIG["methods"]), "summary": aggregate(rows, metrics)}
    (outdir / f"summary_{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("EVALUATION_COMPLETE", suffix, flush=True)


if __name__ == "__main__":
    main()
