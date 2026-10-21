#!/usr/bin/env python3
"""Render comparable camera-space depth from archived Gaussian PLYs.

Use a CUDA Python environment with the confidence rasterizer named in
README.md. The resulting float TIFFs are small enough to publish with figures.
They are model-rendered depth, not measured ground-truth depth.
"""
import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
import torch
from plyfile import PlyData

from scene.cameras import PseudoCamera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

from build_figures import ASSETS, CASES, METHODS, ROOT


def ply_for(ds, method, scene):
    folder = ROOT / "output" / ds / method / scene / "point_cloud"
    matches = list(folder.glob("point_cloud.ply")) or list(folder.glob("final_point_cloud.ply"))
    matches = matches or list(folder.glob("final_point_cloud.ply.gz"))
    if len(matches) != 1:
        raise ValueError((folder, matches))
    return matches[0]


def selected_camera(ds, scene, view, width, height):
    entries = json.loads((ASSETS.parent / "selected_cameras.json").read_text())
    entry = next(x for x in entries if (x["dataset"], x["scene"], x["view"]) == (ds, scene, view))
    if (entry["width"], entry["height"]) != (width, height):
        raise ValueError((ds, scene, view, width, height))
    return PseudoCamera(np.asarray(entry["R"], dtype=np.float32),
                        np.asarray(entry["T"], dtype=np.float32),
                        entry["fovx"], entry["fovy"], width, height)


def main():
    torch.set_grad_enabled(False)
    records = []
    for ds, cases in CASES.items():
        for case in cases:
            scene, view = case["scene"], case["view"]
            for method in METHODS[ds]:
                if method == "GT":
                    continue
                source = ply_for(ds, method, scene)
                target = ASSETS / ds / scene / f"{method}_depth.tiff"
                with tempfile.TemporaryDirectory(prefix="paper_depth_") as temp:
                    if source.suffix == ".gz":
                        plain = Path(temp) / "model.ply"
                        with gzip.open(source, "rb") as src, plain.open("wb") as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
                    else:
                        plain = source
                    props = PlyData.read(str(plain))["vertex"].data.dtype.names
                    n_rest = sum(x.startswith("f_rest_") for x in props)
                    degree = round(math.sqrt(n_rest / 3 + 1) - 1)
                    if 3 * (degree + 1) ** 2 - 3 != n_rest:
                        raise ValueError((source, n_rest))
                    model = GaussianModel(SimpleNamespace(sh_degree=degree))
                    if "f_dc_0" in props:
                        model.load_ply(str(plain))
                    else:
                        # DNGaussian/DropoutGS archive geometry-only PLYs.
                        # The rasterized depth does not depend on SH color.
                        vertices = PlyData.read(str(plain))["vertex"]
                        from torch import nn
                        def fields(names):
                            return torch.tensor(np.stack([vertices[n] for n in names], axis=1),
                                                dtype=torch.float32, device="cuda")
                        model._xyz = nn.Parameter(fields(["x", "y", "z"]))
                        model._opacity = nn.Parameter(fields(["opacity"]))
                        model._scaling = nn.Parameter(fields([f"scale_{i}" for i in range(3)]))
                        model._rotation = nn.Parameter(fields([f"rot_{i}" for i in range(4)]))
                        model._features_dc = nn.Parameter(torch.zeros((len(vertices), 1, 3), device="cuda"))
                        model._features_rest = nn.Parameter(torch.zeros((len(vertices), 0, 3), device="cuda"))
                        model.active_sh_degree = 0
                    model.confidence = torch.ones_like(model.get_opacity)
                    from PIL import Image
                    width, height = Image.open(ASSETS / ds / scene / f"{method}.png").size
                    camera = selected_camera(ds, scene, view, width, height)
                    pipe = SimpleNamespace(use_confidence=False, debug=False,
                                           compute_cov3D_python=False, convert_SHs_python=False)
                    packet = render(camera, model, pipe, torch.zeros(3, device="cuda"))
                    # The FSGS confidence rasterizer returns sum(z_i * w_i)
                    # and alpha=sum(w_i), not normalized expected camera-z.
                    weighted_depth = packet["depth"].squeeze().cpu().numpy().astype(np.float32)
                    alpha = packet["alpha"].squeeze().cpu().numpy()
                    depth = weighted_depth / np.maximum(alpha, 1e-6)
                    depth[(alpha < 0.5) | ~np.isfinite(depth) | (depth <= 0)] = np.nan
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(target), depth, compression="zlib")
                    records.append(dict(dataset=ds, scene=scene, view=view, method=method,
                                        point_cloud=source.relative_to(ROOT).as_posix(),
                                        asset=target.relative_to(ASSETS.parent).as_posix(),
                                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                        valid_fraction=float(np.isfinite(depth).mean()),
                                        width=width, height=height))
                    print(ds, scene, view, method, "depth", np.nanpercentile(depth, [2, 50, 98]), flush=True)
                    del packet, model, camera
                    torch.cuda.empty_cache()
    (ASSETS.parent / "depth_manifest.json").write_text(
        json.dumps(dict(method="Expected camera-z = rasterizer weighted depth / alpha; alpha >= 0.5; "
                        "camera-space scene units; NaN for invalid pixels", assets=records),
                   indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
