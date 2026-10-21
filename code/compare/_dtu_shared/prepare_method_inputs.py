#!/usr/bin/env python3
"""Prepare CPU-side per-method mappings after the shared MVS clouds exist."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from PIL import Image

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
TEMP = Path("/tmp/lilinwei/1007fsgsmvs_dtu")
DATA = TEMP / "data_cache/dtu_corgs"
METHODS = ("FSGS", "CoR-GS", "DropoutGS", "DNGaussian", "Difix3D_10k", "our_method")
SCANS = json.loads((CODE / "experiment_config.json").read_text())["scans"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    difix_splits = PROJECT / "data/splits/difix_strict"
    difix_splits.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    for scan in SCANS:
        split_path = PROJECT / "data/splits/cor_gs_3view" / f"{scan}.json"
        split = json.loads(split_path.read_text())
        init_dir = PROJECT / "output/initialization" / scan
        cloud = init_dir / "initial_point_cloud.ply"
        init = json.loads((init_dir / "manifest.json").read_text())
        cloud_hash = sha256(cloud)
        if init.get("status") != "complete" or init.get("sha256") != cloud_hash:
            raise ValueError(f"{scan}: shared MVS hash differs from its complete manifest")
        if init.get("training_names") != split["train_names"]:
            raise ValueError(f"{scan}: MVS input names differ from the frozen three-view split")
        overlay = TEMP / "overlay/DTU" / scan
        overlay_manifest = json.loads((overlay / "dtu_overlay_manifest.json").read_text())
        if overlay_manifest.get("shared_mvs_sha256") != cloud_hash:
            raise ValueError(f"{scan}: method overlay does not resolve to the shared point cloud")

        # Difix's Nerfstudio parser expects a factor-suffixed folder; materialize
        # the same rounded 1/4-size RGBs used by the other loaders' -r 4 path.
        difix_images = overlay / "images_4"
        difix_images.mkdir(exist_ok=True)
        for name in split["camera_names"]:
            source_image = DATA / scan / "images" / name
            target_image = difix_images / name
            with Image.open(source_image) as image:
                image = image.convert("RGB")
                resolution = (round(image.width / 4), round(image.height / 4))
                resized = image.resize(resolution)
                payload = BytesIO()
                resized.save(payload, format="PNG")
                encoded = payload.getvalue()
                if target_image.exists():
                    if target_image.read_bytes() != encoded:
                        raise ValueError(f"{scan}/{target_image.name}: existing Difix 1/4 image differs")
                else:
                    target_image.write_bytes(encoded)
        if {p.name for p in difix_images.iterdir() if p.is_file()} != set(split["camera_names"]):
            raise ValueError(f"{scan}: Difix images_4 folder is not exactly the frozen 49 images")

        strict = {"scan": scan, "expected_train_count": 3,
                  "train_names": split["train_names"], "test_names": split["test_names"],
                  "fsgs_train_names": split["train_names"], "fsgs_test_names": split["test_names"],
                  "split_sha256": sha256(split_path), "initial_cloud_sha256": cloud_hash}
        strict_path = difix_splits / f"{scan}.json"
        strict_path.write_text(json.dumps(strict, indent=2, ensure_ascii=False) + "\n")

        gt_dir = PROJECT / "output/ground_truth" / scan
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_hashes = {}
        for name in split["test_names"]:
            source_image = DATA / scan / "images" / name
            target_image = gt_dir / f"{Path(name).stem}.png"
            with Image.open(source_image) as image:
                image = image.convert("RGB")
                resolution = (round(image.width / 4), round(image.height / 4))
                resized = image.resize(resolution)
                payload = BytesIO()
                resized.save(payload, format="PNG")
                encoded = payload.getvalue()
                if target_image.exists():
                    if target_image.read_bytes() != encoded:
                        raise ValueError(f"{scan}/{target_image.name}: existing canonical GT differs")
                else:
                    target_image.write_bytes(encoded)
            gt_hashes[target_image.name] = sha256(target_image)

        for method in METHODS:
            dest = PROJECT / "output" / method / scan
            dest.mkdir(parents=True, exist_ok=True)
            mapping = {"protocol": "CoR-GS DTU three-view 2024 seed",
                       "method": method, "scan": scan, "seed": 2024, "iterations": 10000,
                       "camera_order": split["camera_order"], "train_indices": split["train_indices"],
                       "excluded_indices": split["excluded_indices"], "train_names": split["train_names"],
                       "test_names": split["test_names"], "initial_cloud": str(cloud),
                       "initial_cloud_sha256": cloud_hash, "initial_cloud_point_count": init["point_count"],
                       "split_manifest": str(split_path), "split_sha256": sha256(split_path),
                       "strict_split": str(strict_path) if method == "Difix3D_10k" else None,
                       "data_overlay": str(overlay), "resolution_factor": 4,
                       "difix_images_4": str(difix_images) if method == "Difix3D_10k" else None,
                       "canonical_gt_dir": str(gt_dir), "canonical_gt_sha256": gt_hashes}
            (dest / "data_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
        scene_rows.append({"scan": scan, "train_names": split["train_names"],
                           "test_count": len(split["test_names"]), "initial_cloud_sha256": cloud_hash,
                           "point_count": init["point_count"]})

    manifest = {"dataset": "DTU CoR-GS selected 15 scans", "methods": list(METHODS),
                "seed": 2024, "iterations": 10000, "resolution_factor": 4,
                "shared_initialization": "one recomputed three-view COLMAP MVS PLY per scene",
                "scenes": scene_rows}
    (PROJECT / "data/method_input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"PREPARED_METHOD_INPUTS scans={len(scene_rows)} methods={len(METHODS)}")


if __name__ == "__main__":
    main()
