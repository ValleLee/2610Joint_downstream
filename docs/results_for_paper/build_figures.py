#!/usr/bin/env python3
"""Copy exact evaluated RGB views and build RGB/depth comparison plates.

Run once with --collect while the original GT folders are mounted. The copied
assets then make ordinary runs self-contained inside this downstream package.
RGB is never sharpened, color corrected, or cropped per method. Depth TIFFs are
rendered separately from archived Gaussian models by render_depth.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / "assets"
FIGURES = HERE / "figures"
ORIGINAL = Path("/data/lilinwei/3D/1006difix/1007fsgsmvs")
LLFF_GT = ORIGINAL / "output"
DTU_GT = ORIGINAL / "dtu/output/ground_truth"

CASES = {
    "llff": [
        # Crop coordinates are in the original 1008 x 756 pixel image.
        dict(scene="trex", view="DJI_20200223_163604_605.png", roi=[210, 175, 620, 565]),
        dict(scene="room", view="DJI_20200226_143912_008.png", roi=[345, 215, 680, 535]),
    ],
    "dtu": [
        # Coordinates are in the original ~396 x 297 pixel image.
        dict(scene="scan30", view="rect_010_3_r5000.png", roi=[143, 68, 270, 178]),
        dict(scene="scan21", view="rect_010_3_r5000.png", roi=[235, 76, 390, 215]),
    ],
}
METHODS = {
    "llff": ["vanilla_3DGS", "FSGS", "CoR-GS",
             "DropoutGS", "Difix3D_60k", "our_R4", "GT"],
    "dtu": ["vanilla_3DGS", "CoR-GS", "DNGaussian",
            "DropoutGS", "our_R4", "GT"],
}
FOCUSED = {
    "llff": ["vanilla_3DGS", "FSGS", "DropoutGS", "our_R4", "GT"],
    "dtu": ["vanilla_3DGS", "CoR-GS", "DNGaussian", "DropoutGS", "our_R4", "GT"],
}
LABELS = {
    "vanilla_3DGS": "3DGS", "FSGS": "FSGS",
    "CoR-GS": "CoR-GS", "DNGaussian": "DNGaussian",
    "DropoutGS": "DropoutGS", "Difix3D_60k": "Difix3D+ (60k)",
    "our_R4": "Ours", "GT": "Ground Truth",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mapped_render(ds: str, method: str, case: dict) -> Path:
    scene, view = case["scene"], case["view"]
    if ds == "llff" and method in {"DNGaussian", "DropoutGS", "Difix3D_60k"}:
        if method == "Difix3D_60k":
            audit = json.loads((ROOT / "output/llff/source_metrics/DIFIX3D_60K_METRICS.json").read_text())
            mapping = audit["results"]["60000"]["scenes"][scene]["view_mapping"]
        else:
            audit = json.loads((ROOT / "output/llff/source_metrics/UNIFIED_METRICS.json").read_text())
            mapping = audit["results"][method]["results"][scene]["view_mapping"]
        matches = [x["render"] for x in mapping
                   if Path(x["canonical_test_image"]).stem == Path(view).stem]
        if len(matches) != 1:
            raise ValueError(f"No unique LLFF view mapping: {method}/{scene}/{view}")
        name = matches[0]
    else:
        name = view
    base = ROOT / "output" / ds / method
    return base / scene / "renders" / name


def collect() -> None:
    records = []
    for ds, cases in CASES.items():
        for case in cases:
            scene, view = case["scene"], case["view"]
            target_dir = ASSETS / ds / scene
            target_dir.mkdir(parents=True, exist_ok=True)
            for method in METHODS[ds]:
                if method == "GT":
                    source = ((LLFF_GT / scene / "test/ours_10000/gt" / view)
                              if ds == "llff" else DTU_GT / scene / view)
                else:
                    source = mapped_render(ds, method, case)
                if not source.is_file():
                    raise FileNotFoundError(source)
                target = target_dir / f"{method}.png"
                shutil.copyfile(source, target)
                try:
                    source_id = source.resolve().relative_to(ROOT).as_posix()
                except ValueError:
                    source_id = str(source.resolve())
                records.append(dict(dataset=ds, scene=scene, view=view, method=method,
                                    source=source_id, asset=target.relative_to(HERE).as_posix(),
                                    sha256=digest(target), bytes=target.stat().st_size,
                                    width=Image.open(target).width,
                                    height=Image.open(target).height))
    (HERE / "asset_manifest.json").write_text(
        json.dumps(dict(note="Unmodified source PNG copies; absolute GT paths are provenance only.",
                        assets=records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Copied {len(records)} source PNGs to assets/")


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_plate(ds: str, methods: list[str], suffix: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    cell_w, image_w, image_h = 337, 320, 240
    left, top, row_gap = 14, 50, 43
    width = left * 2 + len(methods) * cell_w - (cell_w - image_w)
    height = top + len(CASES[ds]) * (image_h + row_gap) + 10
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font, scene_font = font(27), font(23)
    for col, method in enumerate(methods):
        x = left + col * cell_w
        label = LABELS[method]
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + (image_w - (box[2] - box[0])) / 2, 8),
                  label, fill=(15, 15, 15), font=label_font)
    for row, case in enumerate(CASES[ds]):
        y = top + row * (image_h + row_gap)
        scene, view = case["scene"], case["view"]
        for col, method in enumerate(methods):
            x = left + col * cell_w
            source = ASSETS / ds / scene / f"{method}.png"
            with Image.open(source) as image:
                original = image.convert("RGB")
            w, h = original.size
            bx = case["roi"]
            if not (0 <= bx[0] < bx[2] <= w and 0 <= bx[1] < bx[3] <= h):
                raise ValueError(f"ROI outside {source}: {bx}, image={w}x{h}")
            panel = original.resize((image_w, image_h), Image.Resampling.LANCZOS)
            pd = ImageDraw.Draw(panel)
            rect = tuple(round(v * scale) for v, scale in
                         zip(bx, (image_w / w, image_h / h,
                                  image_w / w, image_h / h)))
            pd.rectangle(rect, outline=(225, 34, 46), width=3)
            crop = original.crop(tuple(bx)).resize((108, 92), Image.Resampling.LANCZOS)
            panel.paste(crop, (5, image_h - 98))
            pd.rectangle((4, image_h - 99, 114, image_h - 4),
                         outline=(225, 34, 46), width=3)
            canvas.paste(panel, (x, y))
        text = f"{scene} | {view}"
        draw.text((left, y + image_h + 5), text, fill=(45, 45, 45), font=scene_font)
    png = FIGURES / f"{ds}_{suffix}.png"
    pdf = FIGURES / f"{ds}_{suffix}.pdf"
    canvas.save(png, optimize=True)
    canvas.save(pdf, "PDF", resolution=300.0)
    print(f"Wrote {png.relative_to(HERE)} ({width}x{height}) and PDF")


def draw_rgb_depth_plate(ds: str, methods: list[str], suffix: str) -> None:
    """Pair each camera's RGB row with its model-rendered depth row."""
    cell_w, image_w, image_h = 337, 320, 240
    left, top, row_gap = 14, 50, 43
    width = left * 2 + len(methods) * cell_w - (cell_w - image_w)
    case_h = image_h * 2 + row_gap
    height = top + len(CASES[ds]) * case_h + 10
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font, scene_font = font(27), font(22)
    gs_cmap = colormaps["turbo_r"]  # camera-z: near warm, far cool
    da_cmap = colormaps["turbo"]    # DA-V2 inverse-like output: high/near warm
    ranges = {}
    for col, method in enumerate(methods):
        x = left + col * cell_w
        label = "Reference" if method == "GT" else LABELS[method]
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + (image_w - (box[2] - box[0])) / 2, 8),
                  label, fill=(15, 15, 15), font=label_font)
    for row, case in enumerate(CASES[ds]):
        scene, view, bx = case["scene"], case["view"], case["roi"]
        y = top + row * case_h
        arrays = {}
        for method in METHODS[ds]:
            if method != "GT":
                path = ASSETS / ds / scene / f"{method}_depth.tiff"
                if not path.is_file():
                    raise FileNotFoundError(path)
                arrays[method] = tifffile.imread(path)
        # One model-depth scale for all methods at a camera. DTU uses the shared
        # RGB crop to expose object-surface variation instead of letting the
        # large background dominate; the complete depth map is still drawn.
        region = "shared ROI" if ds == "dtu" else "full frame"
        depth_arrays = ([v[bx[1]:bx[3], bx[0]:bx[2]] for v in arrays.values()]
                        if ds == "dtu" else list(arrays.values()))
        pooled = np.concatenate([v[np.isfinite(v)] for v in depth_arrays])
        lo, hi = np.percentile(pooled, [2, 98])
        da = tifffile.imread(ASSETS / ds / scene / "GT_da_depth.tiff")
        da_lo, da_hi = np.percentile(da[np.isfinite(da)], [1, 99])
        ranges[f"{ds}/{scene}/{view}"] = {
            "gs_camera_depth_units": [float(lo), float(hi)],
            "gs_scale_region": region,
            "da_v2_relative_values": [float(da_lo), float(da_hi)],
            "polarity": "near=warm, far=cool; DA-V2 independently normalized",
        }
        for col, method in enumerate(methods):
            x = left + col * cell_w
            with Image.open(ASSETS / ds / scene / f"{method}.png") as src:
                original = src.convert("RGB")
            w, h = original.size
            if not (0 <= bx[0] < bx[2] <= w and 0 <= bx[1] < bx[3] <= h):
                raise ValueError((scene, view, bx, original.size))
            rect = tuple(round(v * scale) for v, scale in
                         zip(bx, (image_w / w, image_h / h,
                                  image_w / w, image_h / h)))
            panel = original.resize((image_w, image_h), Image.Resampling.LANCZOS)
            pd = ImageDraw.Draw(panel)
            pd.rectangle(rect, outline=(225, 34, 46), width=3)
            crop = original.crop(tuple(bx)).resize((108, 92), Image.Resampling.LANCZOS)
            panel.paste(crop, (5, image_h - 98))
            pd.rectangle((4, image_h - 99, 114, image_h - 4),
                         outline=(225, 34, 46), width=3)
            canvas.paste(panel, (x, y))
            dy = y + image_h + 5
            if method == "GT":
                if da.shape != (h, w):
                    raise ValueError((scene, view, "DA shape", da.shape, (h, w)))
                normalized = np.clip((da - da_lo) / (da_hi - da_lo), 0, 1)
                rgb = (da_cmap(normalized)[..., :3] * 255).astype(np.uint8)
                dep = Image.fromarray(rgb, "RGB")
                panel = dep.resize((image_w, image_h), Image.Resampling.LANCZOS)
                pd = ImageDraw.Draw(panel)
                pd.rectangle(rect, outline=(225, 34, 46), width=3)
                crop = dep.crop(tuple(bx)).resize((108, 92), Image.Resampling.LANCZOS)
                panel.paste(crop, (5, image_h - 98))
                pd.rectangle((4, image_h - 99, 114, image_h - 4),
                             outline=(225, 34, 46), width=3)
            else:
                data = arrays[method]
                valid = np.isfinite(data)
                normalized = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0, 1)
                rgb = (gs_cmap(normalized)[..., :3] * 255).astype(np.uint8)
                rgb[~valid] = (245, 245, 245)
                dep = Image.fromarray(rgb, "RGB")
                panel = dep.resize((image_w, image_h), Image.Resampling.LANCZOS)
                pd = ImageDraw.Draw(panel)
                pd.rectangle(rect, outline=(225, 34, 46), width=3)
                crop = dep.crop(tuple(bx)).resize((108, 92), Image.Resampling.LANCZOS)
                panel.paste(crop, (5, image_h - 98))
                pd.rectangle((4, image_h - 99, 114, image_h - 4),
                             outline=(225, 34, 46), width=3)
            canvas.paste(panel, (x, dy))
        draw.text((left, dy + image_h + 4), f"{scene} | {view}",
                  fill=(45, 45, 45), font=scene_font)
    ranges_file = HERE / "depth_display_ranges.json"
    previous = json.loads(ranges_file.read_text()) if ranges_file.exists() else {}
    previous.update(ranges)
    ranges_file.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    FIGURES.mkdir(exist_ok=True)
    png = FIGURES / f"{ds}_rgb_depth_{suffix}.png"
    pdf = FIGURES / f"{ds}_rgb_depth_{suffix}.pdf"
    canvas.save(png, optimize=True)
    canvas.save(pdf, "PDF", resolution=300.0)
    print(f"Wrote {png.relative_to(HERE)} ({width}x{height}) and PDF")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true",
                        help="Copy selected render PNGs and GT from original machine.")
    args = parser.parse_args()
    if args.collect:
        collect()
    manifest = HERE / "asset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Run once with --collect to copy the source images")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for record in data["assets"]:
        asset = HERE / record["asset"]
        if not asset.is_file() or digest(asset) != record["sha256"]:
            raise ValueError(f"Missing or modified source asset: {asset}")
    depth_manifest = json.loads((HERE / "depth_manifest.json").read_text())
    for record in depth_manifest["assets"]:
        asset = HERE / record["asset"]
        if not asset.is_file() or digest(asset) != record["sha256"]:
            raise ValueError(f"Missing or modified depth asset: {asset}")
    da_manifest = json.loads((HERE / "da_reference_manifest.json").read_text())
    for record in da_manifest["assets"]:
        asset = HERE / record["asset"]
        rgb_gt = HERE / record["rgb_gt"]
        if (not asset.is_file() or digest(asset) != record["sha256"]
                or not rgb_gt.is_file() or digest(rgb_gt) != record["rgb_sha256"]):
            raise ValueError(f"Missing or modified DA reference: {asset}")
    (HERE / "depth_display_ranges.json").write_text("{}\n", encoding="utf-8")
    for ds in CASES:
        draw_plate(ds, FOCUSED[ds], "main")
        draw_rgb_depth_plate(ds, FOCUSED[ds], "main")
        if FOCUSED[ds] != METHODS[ds]:
            draw_plate(ds, METHODS[ds], "expanded")
            draw_rgb_depth_plate(ds, METHODS[ds], "expanded")


if __name__ == "__main__":
    main()
