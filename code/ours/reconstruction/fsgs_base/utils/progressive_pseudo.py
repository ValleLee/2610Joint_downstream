"""Update timing, rigid camera interpolation, and current-stage pseudo labels.

FSGS stores camera ``R`` transposed: the world-to-camera rotation is ``R.T``.
All interpolation is performed in camera-to-world coordinates, then converted
back to the representation expected by ``scene.cameras.PseudoCamera``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from utils.joint_pseudo_loss import masked_pearson_loss, masked_rgb_loss_stats
from utils.joint_pseudo_loss import valid_render_mask
from utils.pseudo_label_utils import align_teacher_depth_orientation


@dataclass(frozen=True)
class ProgressiveSchedule:
    interval: int
    first: int = 2001
    stop: int = 9500
    updates: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.interval < 1 or self.first >= self.stop:
            raise ValueError("Expected a positive interval and a nonempty pseudo phase")
        object.__setattr__(self, "updates", tuple(range(self.first, self.stop, self.interval)))

    def is_active(self, iteration: int) -> bool:
        return self.first <= iteration < self.stop

    def is_update(self, iteration: int) -> bool:
        return self.is_active(iteration) and (iteration - self.first) % self.interval == 0

    def alpha(self, iteration: int) -> float:
        if not self.is_update(iteration):
            raise ValueError(f"Iteration {iteration} is not a pseudo update")
        update_index = (iteration - self.first) // self.interval + 1
        return update_index / len(self.updates)


def camera_to_c2w(camera) -> np.ndarray:
    """Convert FSGS ``Camera``/``PseudoCamera`` R,T to a 4x4 C2W pose."""
    if isinstance(camera, np.ndarray) and camera.shape == (4, 4):
        return camera.astype(np.float64, copy=True)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(camera.R, dtype=np.float64).T
    w2c[:3, 3] = np.asarray(camera.T, dtype=np.float64).reshape(3)
    return np.linalg.inv(w2c)


def pose_to_camera_rt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 C2W pose to FSGS's transposed R and W2C T."""
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    return w2c[:3, :3].T.copy(), w2c[:3, 3].copy()


def nearest_anchor_index(train_cameras, target_camera) -> int:
    """Nearest true camera by translation plus rotation angle, as in Difix3D."""
    if not train_cameras:
        raise ValueError("At least one real training camera is required")
    target = camera_to_c2w(target_camera)
    distances = []
    for camera in train_cameras:
        source = camera_to_c2w(camera)
        translation = np.linalg.norm(source[:3, 3] - target[:3, 3])
        relative = source[:3, :3].T @ target[:3, :3]
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        distances.append(translation + math.acos(float(cosine)))
    return int(np.argmin(distances))


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return a unit quaternion in [w, x, y, z] order."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * scale, (m[2, 1] - m[1, 2]) / scale,
                      (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = math.sqrt(max(0.0, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2.0
        q = np.empty(4, dtype=np.float64)
        q[0] = (m[k, j] - m[j, k]) / scale
        q[i + 1] = 0.25 * scale
        q[j + 1] = (m[j, i] + m[i, j]) / scale
        q[k + 1] = (m[k, i] + m[i, k]) / scale
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_c2w(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate camera center and SLERP the C2W rotation."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    q0 = _matrix_to_quaternion(start[:3, :3])
    q1 = _matrix_to_quaternion(target[:3, :3])
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
    else:
        theta = math.acos(dot)
        q = (math.sin((1.0 - alpha) * theta) * q0
             + math.sin(alpha * theta) * q1) / math.sin(theta)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quaternion_to_matrix(q)
    pose[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * target[:3, 3]
    return pose


def shift_c2w_toward(start: np.ndarray, target: np.ndarray, distance: float) -> np.ndarray:
    """Take a Difix-style bounded translation step toward a target pose."""
    if distance <= 0:
        raise ValueError("Pose step distance must be positive")
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target[:3, 3] - start[:3, 3]
    translation = float(np.linalg.norm(delta))
    relative = start[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    if translation + math.acos(float(cosine)) <= distance:
        return target.copy()
    if translation <= 1e-6:
        return target.copy()
    progress = min(distance / translation, 1.0)
    pose = interpolate_c2w(start, target, progress)
    if np.linalg.norm(pose[:3, 3] - target[:3, 3]) <= distance:
        pose[:3, 3] = target[:3, 3]
    return pose


@dataclass(frozen=True)
class PseudoLabel:
    """The single current-stage label, replaced at the next camera update."""

    camera: object
    rgb: torch.Tensor
    depth: torch.Tensor
    depth_mask: torch.Tensor
    rgb_mask: torch.Tensor
    generation_iteration: int
    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rgb", self.rgb.detach())
        object.__setattr__(self, "depth", self.depth.detach())
        object.__setattr__(self, "depth_mask", self.depth_mask.detach().bool())
        object.__setattr__(self, "rgb_mask", self.rgb_mask.detach().bool())


class PseudoLabelPool:
    """Keep the latest label or sample one label from all successful updates."""

    def __init__(self, *, mode: str, seed: int, camera_factory) -> None:
        if mode not in ("replace", "accumulate"):
            raise ValueError(f"Unsupported pseudo history mode: {mode}")
        self.mode = mode
        self.camera_factory = camera_factory
        self.labels: list[PseudoLabel] = []
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def count(self) -> int:
        return len(self.labels)

    def add(self, label: PseudoLabel) -> None:
        self.add_batch([label])

    def add_batch(self, labels: list[PseudoLabel]) -> None:
        if self.mode == "replace":
            self.labels = list(labels)
            return
        for label in labels:
            self.labels.append(PseudoLabel(
                camera=label.camera, rgb=label.rgb.cpu(), depth=label.depth.cpu(),
                depth_mask=label.depth_mask.cpu(), rgb_mask=label.rgb_mask.cpu(),
                generation_iteration=label.generation_iteration, alpha=label.alpha,
            ))

    def sample(self, *, device: torch.device) -> PseudoLabel:
        if not self.labels:
            raise RuntimeError("Cannot sample an empty pseudo label pool")
        if self.mode == "replace" and self.count == 1 and self.labels[0].rgb.device == device:
            return self.labels[0]
        index = int(torch.randint(self.count, (1,), generator=self.generator))
        label = self.labels[index]
        return PseudoLabel(
            camera=label.camera, rgb=label.rgb.to(device), depth=label.depth.to(device),
            depth_mask=label.depth_mask.to(device), rgb_mask=label.rgb_mask.to(device),
            generation_iteration=label.generation_iteration, alpha=label.alpha,
        )

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "rng_state": self.generator.get_state(),
            "labels": [{
                "camera_c2w": camera_to_c2w(label.camera).tolist(),
                "rgb": label.rgb.cpu(), "depth": label.depth.cpu(),
                "depth_mask": label.depth_mask.cpu(), "rgb_mask": label.rgb_mask.cpu(),
                "generation_iteration": label.generation_iteration,
                "alpha": label.alpha,
            } for label in self.labels],
        }

    def load_state_dict(self, state: dict) -> None:
        if state["mode"] != self.mode:
            raise ValueError("Pseudo history mode differs from saved pool")
        self.generator.set_state(state["rng_state"])
        self.labels = [PseudoLabel(
            camera=self.camera_factory(np.asarray(saved["camera_c2w"], dtype=np.float64)),
            rgb=saved["rgb"], depth=saved["depth"],
            depth_mask=saved["depth_mask"], rgb_mask=saved["rgb_mask"],
            generation_iteration=int(saved["generation_iteration"]),
            alpha=float(saved["alpha"]),
        ) for saved in state["labels"]]


class ProgressivePseudoController:
    """Maintain exactly one pseudo label while its camera advances to a fixed target."""

    def __init__(self, *, schedule: ProgressiveSchedule, start_c2w: np.ndarray,
                 target_c2w: np.ndarray, camera_factory, predictor, seed: int,
                 alpha_threshold: float = 0.01, rgb_alpha_threshold: float = 0.01) -> None:
        self.schedule = schedule
        self.start_c2w = np.asarray(start_c2w, dtype=np.float64)
        self.target_c2w = np.asarray(target_c2w, dtype=np.float64)
        self.camera_factory = camera_factory
        self.predictor = predictor
        self.seed = seed
        self.alpha_threshold = alpha_threshold
        self.rgb_alpha_threshold = rgb_alpha_threshold
        self.current: PseudoLabel | None = None
        self.last_refresh: dict | None = None

    def state_dict(self) -> dict:
        """Persist the one current label beside a Gaussian checkpoint."""
        state = {"interval": self.schedule.interval,
                 "start_c2w": self.start_c2w.tolist(),
                 "target_c2w": self.target_c2w.tolist(),
                 "current": None}
        if self.current is not None:
            stage = self.current
            state["current"] = {
                "camera_c2w": camera_to_c2w(stage.camera).tolist(),
                "rgb": stage.rgb.cpu(), "depth": stage.depth.cpu(),
                "depth_mask": stage.depth_mask.cpu(), "rgb_mask": stage.rgb_mask.cpu(),
                "generation_iteration": stage.generation_iteration,
                "alpha": stage.alpha,
            }
        return state

    def load_state_dict(self, state: dict, *, device: torch.device) -> None:
        if state["interval"] != self.schedule.interval:
            raise ValueError("Pseudo interval differs from saved stage")
        if (not np.allclose(state["start_c2w"], self.start_c2w)
                or not np.allclose(state["target_c2w"], self.target_c2w)):
            raise ValueError("Pseudo camera path differs from saved stage")
        saved = state["current"]
        if saved is None:
            self.current = None
            return
        self.current = PseudoLabel(
            camera=self.camera_factory(np.asarray(saved["camera_c2w"], dtype=np.float64)),
            rgb=saved["rgb"].to(device), depth=saved["depth"].to(device),
            depth_mask=saved["depth_mask"].to(device),
            rgb_mask=saved["rgb_mask"].to(device),
            generation_iteration=int(saved["generation_iteration"]),
            alpha=float(saved["alpha"]),
        )

    def prepare_iteration(self, iteration: int, render_for_teacher) -> PseudoLabel | None:
        """Refresh only at schedule boundaries; otherwise reuse the current label."""
        self.last_refresh = None
        if not self.schedule.is_active(iteration):
            self.current = None
            return None
        if not self.schedule.is_update(iteration):
            return self.current

        progress = self.schedule.alpha(iteration)
        pose = interpolate_c2w(self.start_c2w, self.target_c2w, progress)
        camera = self.camera_factory(pose)
        self.current = None  # a failed refresh must not keep stale supervision
        record = {"iteration": iteration, "alpha": progress,
                  "camera_c2w": pose.tolist(), "predictor_called": False,
                  "valid_pixels": 0, "status": "skipped"}
        self.last_refresh = record
        with torch.no_grad():
            rendered = render_for_teacher(camera)
            rgb = rendered["render"]
            depth = rendered["depth"][0]
            alpha = rendered["alpha"][0]
            valid = valid_render_mask(alpha, depth, self.alpha_threshold)
            record["valid_pixels"] = int(valid.sum())
            if record["valid_pixels"] < 32:
                record["reason"] = "fewer_than_32_valid_pixels"
                return None
            try:
                prediction = self.predictor.predict(rgb, depth, valid, seed=self.seed + iteration)
            except RuntimeError as exc:
                if "Invalid rendered depth range" not in str(exc):
                    raise
                record["reason"] = str(exc)
                return None
            record["predictor_called"] = True
            aligned_depth = align_teacher_depth_orientation(
                prediction["depth01"], inverse_representation=True,
            )
            rgb_valid = valid_render_mask(alpha, depth, self.rgb_alpha_threshold)
            native_loss = masked_pearson_loss(depth, prediction["depth01"], valid)
            aligned_loss = masked_pearson_loss(depth, aligned_depth, valid)
            record["native_depth_corr"] = (None if native_loss is None else 1.0 - float(native_loss))
            record["aligned_depth_corr"] = (None if aligned_loss is None else 1.0 - float(aligned_loss))
            record["status"] = "generated"
            self.current = PseudoLabel(
                camera=camera, rgb=prediction["rgb01"], depth=aligned_depth,
                depth_mask=valid, rgb_mask=rgb_valid,
                generation_iteration=iteration, alpha=progress,
            )
        return self.current


class DifixStageController:
    """Refresh one repaired view per target at each scheduled update."""

    def __init__(self, *, schedule: ProgressiveSchedule, initial_c2ws,
                 target_c2ws, camera_factory, predictor, seed: int,
                 pose_step: float, alpha_threshold: float = 0.01,
                 rgb_alpha_threshold: float = 0.01) -> None:
        if not initial_c2ws or not target_c2ws:
            raise ValueError("Difix stages need initial and target camera poses")
        if pose_step <= 0:
            raise ValueError("Pose step distance must be positive")
        self.schedule = schedule
        self.current_poses = [np.asarray(p, dtype=np.float64).copy() for p in initial_c2ws]
        self.target_c2ws = [np.asarray(p, dtype=np.float64).copy() for p in target_c2ws]
        self.camera_factory = camera_factory
        self.predictor = predictor
        self.seed = seed
        self.pose_step = pose_step
        self.alpha_threshold = alpha_threshold
        self.rgb_alpha_threshold = rgb_alpha_threshold
        self.current: list[PseudoLabel] | None = None
        self.last_refresh: dict | None = None

    def state_dict(self) -> dict:
        return {
            "current_poses": [pose.tolist() for pose in self.current_poses],
            "target_c2ws": [pose.tolist() for pose in self.target_c2ws],
            "pose_step": self.pose_step,
            "interval": self.schedule.interval,
            "current": None if self.current is None else [{
                "camera_c2w": camera_to_c2w(label.camera).tolist(),
                "rgb": label.rgb.cpu(), "depth": label.depth.cpu(),
                "depth_mask": label.depth_mask.cpu(), "rgb_mask": label.rgb_mask.cpu(),
                "generation_iteration": label.generation_iteration,
                "alpha": label.alpha,
            } for label in self.current],
        }

    def load_state_dict(self, state: dict, *, device: torch.device) -> None:
        if state["interval"] != self.schedule.interval or state["pose_step"] != self.pose_step:
            raise ValueError("Difix update schedule differs from saved stage")
        if len(state["target_c2ws"]) != len(self.target_c2ws) or any(
            not np.allclose(saved, target)
            for saved, target in zip(state["target_c2ws"], self.target_c2ws)
        ):
            raise ValueError("Difix targets differ from saved stage")
        self.current_poses = [np.asarray(p, dtype=np.float64) for p in state["current_poses"]]
        saved_labels = state["current"]
        self.current = None if saved_labels is None else [PseudoLabel(
            camera=self.camera_factory(np.asarray(saved["camera_c2w"], dtype=np.float64)),
            rgb=saved["rgb"].cpu(), depth=saved["depth"].cpu(),
            depth_mask=saved["depth_mask"].cpu(),
            rgb_mask=saved["rgb_mask"].cpu(),
            generation_iteration=int(saved["generation_iteration"]),
            alpha=float(saved["alpha"]),
        ) for saved in saved_labels]

    def prepare_iteration(self, iteration: int, render_for_teacher) -> list[PseudoLabel] | None:
        self.last_refresh = None
        if not self.schedule.is_active(iteration):
            self.current = None
            return None
        if not self.schedule.is_update(iteration):
            return self.current

        next_poses = []
        labels = []
        views = []
        for target_index, target in enumerate(self.target_c2ws):
            anchor_index = nearest_anchor_index(self.current_poses, target)
            pose = shift_c2w_toward(self.current_poses[anchor_index], target, self.pose_step)
            next_poses.append(pose)
            camera = self.camera_factory(pose)
            view_record = {"target_index": target_index, "anchor_index": anchor_index,
                           "camera_c2w": pose.tolist(), "status": "skipped",
                           "valid_pixels": 0}
            views.append(view_record)
            with torch.no_grad():
                rendered = render_for_teacher(camera)
                rgb = rendered["render"]
                depth = rendered["depth"][0]
                alpha = rendered["alpha"][0]
                valid = valid_render_mask(alpha, depth, self.alpha_threshold)
                view_record["valid_pixels"] = int(valid.sum())
                if view_record["valid_pixels"] < 32:
                    view_record["reason"] = "fewer_than_32_valid_pixels"
                    continue
                try:
                    prediction = self.predictor.predict(
                        rgb, depth, valid, seed=self.seed + iteration * 1000 + target_index,
                    )
                except RuntimeError as exc:
                    if "Invalid rendered depth range" not in str(exc):
                        raise
                    view_record["reason"] = str(exc)
                    continue
                aligned_depth = align_teacher_depth_orientation(
                    prediction["depth01"], inverse_representation=True,
                )
                labels.append(PseudoLabel(
                    camera=camera, rgb=prediction["rgb01"].cpu(), depth=aligned_depth.cpu(),
                    depth_mask=valid.cpu(),
                    rgb_mask=valid_render_mask(alpha, depth, self.rgb_alpha_threshold).cpu(),
                    generation_iteration=iteration, alpha=self.schedule.alpha(iteration),
                ))
                view_record["status"] = "generated"
        self.current_poses = next_poses
        self.current = labels or None
        self.last_refresh = {
            "iteration": iteration, "target_count": len(self.target_c2ws),
            "generated_count": len(labels), "predictor_called": bool(labels),
            "status": "generated" if labels else "skipped", "views": views,
        }
        return self.current


def prepare_training_label(controller: ProgressivePseudoController | DifixStageController,
                           pool: PseudoLabelPool, iteration: int,
                           render_for_teacher, device: torch.device) -> PseudoLabel | None:
    """Refresh a label when scheduled, then select one active pseudo view."""
    current = controller.prepare_iteration(iteration, render_for_teacher)
    if current is None:
        return None
    if controller.last_refresh is not None and controller.last_refresh["status"] == "generated":
        pool.add_batch(current if isinstance(current, list) else [current])
    return pool.sample(device=device)


def compute_stage_loss(
    rendered_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    label: PseudoLabel,
    *,
    rgb_weight: float,
    depth_weight: float,
    alpha: torch.Tensor | None = None,
    alpha_threshold: float = 0.01,
    rgb_full_valid_windows: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compare a *fresh* Gaussian render to cached, detached stage labels."""
    current_valid = torch.isfinite(rendered_depth) & (rendered_depth > 0)
    if alpha is not None:
        current_valid &= alpha.detach() > alpha_threshold
    depth_mask = label.depth_mask & current_valid
    rgb_mask = label.rgb_mask & current_valid
    rgb_details = masked_rgb_loss_stats(
        rendered_rgb, label.rgb, rgb_mask, dssim_weight=0.2,
        full_valid_windows=rgb_full_valid_windows,
    )
    depth_loss = masked_pearson_loss(rendered_depth, label.depth, depth_mask)
    total = rendered_rgb.new_zeros(())
    if rgb_details["loss"] is not None:
        total = total + rgb_weight * rgb_details["loss"]
    if depth_loss is not None:
        total = total + depth_weight * depth_loss
    return total, {"rgb": rgb_details, "depth": depth_loss,
                   "depth_valid_pixels": int(depth_mask.sum())}
