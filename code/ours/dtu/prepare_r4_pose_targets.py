#!/usr/bin/env python3
"""Freeze eight pose-only R4 pseudo targets from held-out DTU camera poses."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SCANS = json.loads((PROJECT / "code/experiment_config.json").read_text())["scans"]
OUT = PROJECT / "data/pseudo_pose_targets"


def rotation_from_q(q: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for scan in SCANS:
        split = json.loads((PROJECT / "data/splits/cor_gs_3view" / f"{scan}.json").read_text())
        test_records = [r for r in split["image_records"] if r["name"] in set(split["test_names"])]
        test_records.sort(key=lambda r: r["name"])
        chosen_idx = np.rint(np.linspace(0, len(test_records) - 1, 8)).astype(int).tolist()
        chosen = [test_records[i] for i in chosen_idx]
        poses = []
        for record in chosen:
            w2c_r = rotation_from_q(record["qvec_wxyz"])
            w2c_t = np.asarray(record["tvec"], dtype=np.float64)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = w2c_r.T
            c2w[:3, 3] = -w2c_r.T @ w2c_t
            poses.append(c2w.tolist())
        payload = {"scan": scan, "pose_convention": "c2w", "coordinate_frame": "dataset_world",
                   "source_note": "Eight evenly spaced COLMAP camera poses selected from the 31 held-out pose records; RGB is never loaded as a pseudo target.",
                   "selection": {"candidate_names": split["test_names"], "selected_names": [r["name"] for r in chosen],
                                 "selected_test_list_indices": chosen_idx, "selection_rule": "round(linspace(0,30,8))"},
                   "poses": poses}
        (OUT / f"{scan}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"FROZEN_POSE_ONLY_TARGETS scans={len(SCANS)} poses_per_scan=8")


if __name__ == "__main__":
    main()
