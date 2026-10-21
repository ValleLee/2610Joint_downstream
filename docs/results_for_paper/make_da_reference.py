#!/usr/bin/env python3
"""Run Depth Anything V2 on the exact copied RGB GT views used in figures.

The output is a relative monocular depth proxy (larger values are generally
nearer), never metric depth ground truth. Keep raw float32 TIFFs for rerendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from build_figures import ASSETS, CASES, HERE, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da-root", type=Path,
                        default=Path("/data/lilinwei/3D/D23E/Depth-Anything-V2"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data/lilinwei/3D/D23E/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sys.path.insert(0, str(args.da_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vitl", features=256,
                            out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(args.device)
    records = []
    for ds, cases in CASES.items():
        for case in cases:
            scene, view = case["scene"], case["view"]
            source = ASSETS / ds / scene / "GT.png"
            target = ASSETS / ds / scene / "GT_da_depth.tiff"
            rgb = np.asarray(Image.open(source).convert("RGB"))
            bgr = rgb[:, :, ::-1].copy()
            with torch.inference_mode():
                depth = model.infer_image(bgr, input_size=518).astype(np.float32)
            if depth.shape != rgb.shape[:2] or not np.all(np.isfinite(depth)):
                raise ValueError((source, depth.shape, rgb.shape))
            tifffile.imwrite(target, depth, compression="zlib")
            records.append(dict(dataset=ds, scene=scene, view=view,
                                rgb_gt=source.relative_to(HERE).as_posix(),
                                rgb_sha256=digest(source),
                                asset=target.relative_to(HERE).as_posix(),
                                sha256=digest(target), width=depth.shape[1], height=depth.shape[0],
                                min=float(depth.min()), max=float(depth.max())))
            print(ds, scene, view, "DA raw range", float(depth.min()), float(depth.max()), flush=True)
    (HERE / "da_reference_manifest.json").write_text(
        json.dumps(dict(model="Depth Anything V2 relative ViT-L", input_size=518,
                        checkpoint_sha256=digest(args.checkpoint),
                        note="Predictions from the exact figure RGB GT; not measured depth; values are relative/inverse-like.",
                        assets=records), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
