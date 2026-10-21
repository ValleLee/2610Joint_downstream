"""Compatibility shim for the legacy SceneManager API used by this checkout.

The current PyPI pycolmap releases no longer expose SceneManager. This adapter
uses COLMAP's standard binary/text readers to present the small subset of the
legacy interface consumed by examples/gsplat/datasets/colmap.py.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from colmap_read_write_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
)


_CAMERA_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def _camera_adapter(camera):
    p = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx, cx, cy = p[:3]
        fy = fx
    else:
        fx, fy, cx, cy = p[:4]
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "sx1", "sy1", "theta")
    distortion = p[3:] if model not in ("SIMPLE_PINHOLE", "PINHOLE") else np.empty(0)
    fields = {name: float(distortion[i]) if i < len(distortion) else 0.0 for i, name in enumerate(names)}
    return SimpleNamespace(
        id=camera.id,
        model=model,
        camera_type=_CAMERA_IDS[model],
        width=int(camera.width),
        height=int(camera.height),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        **fields,
    )


class _ImageAdapter:
    def __init__(self, image):
        self.id = int(image.id)
        self.qvec = np.asarray(image.qvec, dtype=np.float64)
        self.tvec = np.asarray(image.tvec, dtype=np.float64)
        self.camera_id = int(image.camera_id)
        self.name = image.name
        self.point3D_ids = np.asarray(image.point3D_ids, dtype=np.int64)

    def R(self):
        return qvec2rotmat(self.qvec)


class SceneManager:
    def __init__(self, path):
        self.path = Path(path)
        self.cameras = {}
        self.images = {}
        self.points3D = np.empty((0, 3), dtype=np.float64)
        self.point3D_errors = np.empty((0,), dtype=np.float64)
        self.point3D_colors = np.empty((0, 3), dtype=np.uint8)
        self.name_to_image_id = {}
        self.point3D_id_to_point3D_idx = {}
        self.point3D_id_to_images = {}

    def _suffix(self):
        return ".bin" if (self.path / "cameras.bin").is_file() else ".txt"

    def load_cameras(self):
        suffix = self._suffix()
        reader = read_cameras_binary if suffix == ".bin" else read_cameras_text
        self.cameras = {int(k): _camera_adapter(v) for k, v in reader(str(self.path / f"cameras{suffix}")).items()}

    def load_images(self):
        suffix = self._suffix()
        reader = read_images_binary if suffix == ".bin" else read_images_text
        raw = reader(str(self.path / f"images{suffix}"))
        self.images = {int(k): _ImageAdapter(v) for k, v in raw.items()}
        self.name_to_image_id = {v.name: k for k, v in self.images.items()}

    def load_points3D(self):
        suffix = self._suffix()
        reader = read_points3D_binary if suffix == ".bin" else read_points3D_text
        raw = reader(str(self.path / f"points3D{suffix}"))
        point_ids = list(raw.keys())
        self.point3D_id_to_point3D_idx = {int(pid): i for i, pid in enumerate(point_ids)}
        self.points3D = np.asarray([raw[pid].xyz for pid in point_ids], dtype=np.float64).reshape(-1, 3)
        self.point3D_errors = np.asarray([raw[pid].error for pid in point_ids], dtype=np.float64)
        self.point3D_colors = np.asarray([raw[pid].rgb for pid in point_ids], dtype=np.uint8).reshape(-1, 3)
        self.point3D_id_to_images = {
            int(pid): list(zip(raw[pid].image_ids.tolist(), raw[pid].point2D_idxs.tolist()))
            for pid in point_ids
        }
