"""Load camera targets supplied independently of the evaluation split."""

import json
from pathlib import Path

import numpy as np


def load_external_target_poses(path: str | Path) -> list[np.ndarray]:
    """Read dataset-world camera-to-world poses from a declared trajectory."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Target trajectory must be a JSON object")
    if payload.get("pose_convention") != "c2w":
        raise ValueError("Target trajectory must declare pose_convention='c2w'")
    if payload.get("coordinate_frame") != "dataset_world":
        raise ValueError("Target trajectory must declare coordinate_frame='dataset_world'")
    if not isinstance(payload.get("source_note"), str) or not payload["source_note"].strip():
        raise ValueError("Target trajectory needs a nonempty source_note")
    raw_poses = payload.get("poses")
    if not isinstance(raw_poses, list) or not raw_poses:
        raise ValueError("Target trajectory needs a nonempty poses list")

    poses = []
    for index, raw_pose in enumerate(raw_poses):
        try:
            pose = np.asarray(raw_pose, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pose {index} is not numeric") from exc
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Pose {index} must be a finite 4x4 matrix")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"Pose {index} has an invalid homogeneous row")
        rotation = pose[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
            raise ValueError(f"Pose {index} has an invalid rotation")
        poses.append(pose)
    return poses
