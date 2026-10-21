#!/usr/bin/env python3
"""Evaluate Difix3D_60k outputs with the exact shared 1007fsgsmvs protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/lilinwei/3D/baselines/output/1007_fsgs_mvs_baselines")
sys.path.insert(0, str(ROOT / "configs"))
from compute_unified_metrics import (  # noqa: E402
    INPUT_MANIFEST,
    CANONICAL_MVS_GT,
    pair_renders,
    prepare_common_gt,
    score_views,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", nargs="+", type=int, default=[30000, 60000])
    args = parser.parse_args()

    sys.path.insert(0, "/data/lilinwei/3D/1006difix")
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips

    device = torch.device(args.device)
    manifest = json.loads(INPUT_MANIFEST.read_text())
    scene_list = list(manifest["scenes"])
    all_results = {
        "protocol": {
            "method": "Difix3D_60k",
            "steps": args.steps,
            "scenes": scene_list,
            "canonical_gt": str(CANONICAL_MVS_GT / "<scene>/test/ours_10000/gt"),
            "aggregation": "mean over test views within scene, then equal-weight mean over 8 scenes",
            "metrics": "same utils.image_utils.psnr, utils.loss_utils.ssim and lpipsPyTorch VGG as the 1007fsgsmvs baseline comparison",
            "device": str(device),
        },
        "results": {},
    }
    summary_rows = []
    for step in args.steps:
        scene_results = {}
        for scene in scene_list:
            info = manifest["scenes"][scene]
            mapping = json.loads((ROOT / "Difix3D_60k/data_mapping" / f"{scene}.json").read_text())
            test_names = mapping["test_names"]
            if len(test_names) != info["eval_count"]:
                raise ValueError(f"unexpected view count for {scene}: {len(test_names)}")
            gt = prepare_common_gt(scene, test_names)
            render_dir = ROOT / "Difix3D_60k/outputs" / scene / "eval" / f"ours_{step}" / "renders"
            pairs = pair_renders(render_dir, test_names)
            metrics = score_views(pairs, gt, device, psnr, ssim, lpips)
            scene_results[scene] = metrics
            print(f"{step}\t{scene}\t{metrics['n_views']}\t{metrics['PSNR']:.6f}\t{metrics['SSIM']:.6f}\t{metrics['LPIPS']:.6f}", flush=True)
        macro = {
            key: float(np.mean([record[key] for record in scene_results.values()]))
            for key in ("PSNR", "SSIM", "LPIPS")
        }
        all_results["results"][str(step)] = {"scenes": scene_results, "macro": macro}
        summary_rows.append({"step": step, "n_scenes": len(scene_results), "n_views": sum(v["n_views"] for v in scene_results.values()), **macro})

        method_metrics = ROOT / "Difix3D_60k/metrics"
        method_metrics.mkdir(parents=True, exist_ok=True)
        (method_metrics / f"public_common_eval_{step}.json").write_text(
            json.dumps({**all_results["protocol"], "step": step, "results": scene_results, "macro": macro}, indent=2) + "\n"
        )
        (method_metrics / f"public_common_macro_{step}.json").write_text(json.dumps(macro, indent=2) + "\n")
        if step == 60000:
            (method_metrics / "public_common_eval.json").write_text(
                json.dumps({**all_results["protocol"], "step": step, "results": scene_results, "macro": macro}, indent=2) + "\n"
            )
            (method_metrics / "public_common_macro.json").write_text(json.dumps(macro, indent=2) + "\n")

    (ROOT / "DIFIX3D_60K_METRICS.json").write_text(json.dumps({**all_results, "summary": summary_rows}, indent=2) + "\n")
    with (ROOT / "DIFIX3D_60K_METRICS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "n_scenes", "n_views", "PSNR", "SSIM", "LPIPS"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
