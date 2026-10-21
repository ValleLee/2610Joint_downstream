# The experiment-local entrypoint imports all unmodified FSGS modules from the
# original checkout while keeping this memory adapter isolated in 1007fsgsmvs.
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path("/data/lilinwei/3D/1006difix")
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import os
import torch
import numpy as np
from torchmetrics.functional.regression import pearson_corrcoef
from fsgs_r4_mipnerf_depth import stable_pearson_depth_loss
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
import scene.gaussian_model as _gaussian_model
from scipy.spatial import cKDTree as _cKDTree
from utils.general_utils import safe_state
import uuid
import json
import time
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from fsgs_r4_memory_teacher import JointRgbdTeacher
from utils.progressive_pseudo import (
    ProgressiveSchedule, ProgressivePseudoController, DifixStageController,
    PseudoLabelPool,
    camera_to_c2w, nearest_anchor_index, pose_to_camera_rt,
    prepare_training_label, compute_stage_loss,
)
from scene.cameras import PseudoCamera
from utils.target_trajectory import load_external_target_poses

# Mip-NeRF 360 images are COLMAP scenes and do not ship LLFF poses_bounds.npy.
# The inherited loader reads that file only to fill CameraInfo.bounds, which is
# not used by R4's external target-pose path. Supply a neutral placeholder in
# this isolated Mip-NeRF entrypoint, and skip constructing the unused random
# LLFF pseudo trajectory. No camera pose, image, or training split is changed.
import scene.dataset_readers as _dataset_readers
import scene as _scene_package
_read_colmap_cameras = _dataset_readers.readColmapCameras


def _read_colmap_cameras_mipnerf(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping):
    if (_Path(path) / "poses_bounds.npy").is_file():
        return _read_colmap_cameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping)
    original_numpy_load = np.load
    bounds_path = (_Path(path) / "poses_bounds.npy").resolve()

    def numpy_load_with_unused_bounds(file, *args, **kwargs):
        try:
            return original_numpy_load(file, *args, **kwargs)
        except FileNotFoundError:
            if _Path(file).resolve() != bounds_path:
                raise
            placeholder = np.ones((len(cam_extrinsics), 17), dtype=np.float32)
            placeholder[:, -2:] = (1.0, 2.0)
            return placeholder

    np.load = numpy_load_with_unused_bounds
    try:
        return _read_colmap_cameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping)
    finally:
        np.load = original_numpy_load


_dataset_readers.readColmapCameras = _read_colmap_cameras_mipnerf
_scene_package.generate_random_poses_llff = lambda views: []


def _dist3_cpu(points):
    """Memory-safe equivalent of simple-knn distCUDA2 for this run only.

    Returns the mean squared distance to the three nearest *other* points and
    their indices, matching the CUDA operator's contract. The surrounding
    Gaussian model/training implementation remains imported from the original
    checkout.
    """
    xyz = points.detach().to(device="cpu", dtype=torch.float32).numpy()
    distances, indices = _cKDTree(xyz).query(xyz, k=4, workers=4)
    row_ids = np.arange(len(xyz))[:, None]
    nonself = indices != row_ids
    nearest_d = np.stack([d[m][:3] for d, m in zip(distances, nonself)])
    nearest_i = np.stack([i[m][:3] for i, m in zip(indices, nonself)])
    mean_squared = np.square(nearest_d, dtype=np.float64).mean(axis=1).astype(np.float32)
    return (
        torch.as_tensor(mean_squared, device=points.device, dtype=torch.float32),
        torch.as_tensor(nearest_i, device=points.device, dtype=torch.int32),
    )


# This only replaces the memory-heavy implementation of the same 3-NN metric
# in the copied experiment entrypoint; the original FSGS repository is untouched.
_gaussian_model.distCUDA2 = _dist3_cpu


def save_pseudo_snapshot(path, iteration, stage, rendered_rgb, rendered_depth,
                         anchor_id, target_id):
    """Save paired stage labels and fresh Gaussian renders for later inspection."""
    from torchvision.utils import save_image

    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        save_image(rendered_rgb.detach().clamp(0, 1), os.path.join(path, "render_rgb.png"))
        save_image(stage.rgb.clamp(0, 1), os.path.join(path, "restored_rgb.png"))
        for name, depth in (("render_depth.png", rendered_depth.detach()),
                            ("restored_depth_aligned.png", stage.depth)):
            valid = stage.depth_mask & torch.isfinite(depth)
            values = depth[valid].float()
            if values.numel():
                low, high = torch.quantile(values, torch.tensor([0.01, 0.99], device=values.device))
                preview = ((depth - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)
                preview = torch.where(valid, preview, torch.zeros_like(preview))
                save_image(preview[None], os.path.join(path, name))
        torch.save({
            "rendered_depth": rendered_depth.detach().cpu(),
            "restored_depth_aligned": stage.depth.cpu(),
            "depth_mask": stage.depth_mask.cpu(),
            "rgb_mask": stage.rgb_mask.cpu(),
        }, os.path.join(path, "depth_and_masks.pt"))
    with open(os.path.join(path, "camera.json"), "w", encoding="utf-8") as file:
        json.dump({"iteration": iteration, "generation_iteration": stage.generation_iteration,
                   "alpha": stage.alpha, "anchor_camera_id": anchor_id,
                   "target_camera_id": target_id,
                   "camera_c2w": camera_to_c2w(stage.camera).tolist()}, file, indent=2)


def training(dataset, opt, pipe, args):
    schedule = ProgressiveSchedule(
        interval=args.pseudo_update_interval,
        first=args.start_sample_pseudo + 1,
        stop=args.end_sample_pseudo,
    )
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    joint_teacher = JointRgbdTeacher(
        checkpoint=args.joint_checkpoint, code_root=args.joint_code_root,
        device=background.device, processing_res=args.joint_processing_res,
        denoise_steps=args.joint_denoise_steps, fp16=args.joint_fp16,
    )
    joint_teacher.offload_to_cpu()
    pseudo_stats_path = os.path.join(scene.model_path, "pseudo_stats.jsonl")
    open(pseudo_stats_path, "a", encoding="utf-8").close()
    loss_components_path = os.path.join(scene.model_path, "loss_components.jsonl")
    open(loss_components_path, "a", encoding="utf-8").close()
    pseudo_snapshot_root = os.path.join(scene.model_path, "diagnostics", "pseudo_views")
    train_cameras = scene.getTrainCameras()
    if args.pseudo_target_source == "external":
        if not args.pseudo_target_pose_file:
            raise ValueError("External targets require --pseudo_target_pose_file")
        external_poses = load_external_target_poses(args.pseudo_target_pose_file)
        if args.pseudo_target_index != 0 or args.pseudo_target_count != len(external_poses):
            raise ValueError("External trajectory must contain exactly the configured target count")
        reference = train_cameras[0]
        pseudo_cameras = []
        for pose in external_poses:
            rotation, translation = pose_to_camera_rt(pose)
            pseudo_cameras.append(PseudoCamera(
                R=rotation, T=translation, FoVx=reference.FoVx,
                FoVy=reference.FoVy, width=reference.image_width,
                height=reference.image_height,
            ))
    else:
        pseudo_cameras = (scene.getTestCameras() if args.pseudo_target_source == "test"
                          else scene.getPseudoCameras())
        if args.pseudo_target_source == "test" and (
            args.pseudo_target_index != 0 or args.pseudo_target_count != len(pseudo_cameras)
        ):
            raise ValueError("Test-target mode must use every held-out test camera")
    target_end = args.pseudo_target_index + args.pseudo_target_count
    if not 0 <= args.pseudo_target_index < target_end <= len(pseudo_cameras):
        raise ValueError("Requested pseudo targets are outside the selected camera set")
    if args.pseudo_training_mode == "joint" and args.pseudo_target_count != 1:
        raise ValueError("Joint pseudo training requires exactly one target")
    target_cameras = pseudo_cameras[args.pseudo_target_index:target_end]
    if len({(camera.image_width, camera.image_height, camera.FoVx, camera.FoVy)
            for camera in target_cameras}) != 1:
        raise ValueError("Multi-target pseudo cameras must share image size and intrinsics")
    target_camera = target_cameras[0]
    anchor_ids = [nearest_anchor_index(train_cameras, target) for target in target_cameras]
    anchor_id = anchor_ids[0]
    target_ids = list(range(args.pseudo_target_index, target_end))

    def make_progressive_camera(c2w):
        rotation, translation = pose_to_camera_rt(c2w)
        return PseudoCamera(
            R=rotation, T=translation, FoVx=target_camera.FoVx,
            FoVy=target_camera.FoVy, width=target_camera.image_width,
            height=target_camera.image_height,
        )

    if args.pseudo_training_mode == "difix":
        controller = DifixStageController(
            schedule=schedule,
            initial_c2ws=[camera_to_c2w(camera) for camera in train_cameras],
            target_c2ws=[camera_to_c2w(camera) for camera in target_cameras],
            camera_factory=make_progressive_camera, predictor=joint_teacher,
            seed=args.seed, pose_step=args.pseudo_pose_step,
            alpha_threshold=args.pseudo_alpha_threshold,
            rgb_alpha_threshold=args.pseudo_rgb_alpha_threshold,
        )
    else:
        controller = ProgressivePseudoController(
            schedule=schedule, start_c2w=camera_to_c2w(train_cameras[anchor_id]),
            target_c2w=camera_to_c2w(target_camera),
            camera_factory=make_progressive_camera, predictor=joint_teacher,
            seed=args.seed, alpha_threshold=args.pseudo_alpha_threshold,
            rgb_alpha_threshold=args.pseudo_rgb_alpha_threshold,
        )
    pseudo_pool = PseudoLabelPool(
        mode=args.pseudo_history_mode, seed=args.seed,
        camera_factory=make_progressive_camera,
    )
    train_choice_generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    prior_elapsed = 0.0
    pseudo_train_steps = 0
    pseudo_rgb_steps = 0
    pseudo_depth_steps = 0
    if checkpoint:
        stage_path = checkpoint + ".pseudo.pt"
        if not os.path.isfile(stage_path):
            raise FileNotFoundError(
                f"Missing progressive state for resumed checkpoint: {stage_path}"
            )
        resumed = torch.load(stage_path, map_location="cpu", weights_only=True)
        controller.load_state_dict(resumed["controller"], device=background.device)
        if args.pseudo_history_mode == "accumulate":
            if resumed.get("pseudo_pool") is None:
                raise ValueError("Accumulated pseudo history is missing from checkpoint")
            pseudo_pool.load_state_dict(resumed["pseudo_pool"])
        elif controller.current is not None:
            pseudo_pool.add_batch(controller.current if isinstance(controller.current, list)
                                  else [controller.current])
            if resumed.get("pseudo_pool_rng_state") is not None:
                pseudo_pool.generator.set_state(resumed["pseudo_pool_rng_state"])
        if args.pseudo_training_mode == "difix":
            if resumed.get("train_choice_state") is None:
                raise ValueError("Difix sampling state is missing from checkpoint")
            train_choice_generator.set_state(resumed["train_choice_state"])
        prior_elapsed = float(resumed["elapsed_training_seconds"])
        pseudo_train_steps = int(resumed["pseudo_train_steps"])
        pseudo_rgb_steps = int(resumed["pseudo_rgb_steps"])
        pseudo_depth_steps = int(resumed["pseudo_depth_steps"])
    print(f"[ProgressivePath] anchors={anchor_ids} targets={target_ids} "
          f"updates={len(schedule.updates)} interval={schedule.interval} "
          f"history_mode={args.pseudo_history_mode} training_mode={args.pseudo_training_mode} "
          f"target_source={args.pseudo_target_source}",
          flush=True)
    training_start = time.monotonic()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss_mask(image, gt_image)
        real_rgb_dssim = 1.0 - ssim(image, gt_image)
        real_rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * real_rgb_dssim
        loss = real_rgb_loss


        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = stable_pearson_depth_loss(midas_depth, rendered_depth)
        real_depth_weight = args.depth_weight
        real_depth_weighted = real_depth_weight * depth_loss
        loss += real_depth_weighted

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001



        pseudo_depth_raw = None
        pseudo_rgb_raw = None
        pseudo_depth_weighted = None
        pseudo_rgb_weighted = None
        pseudo_depth_weight = 0.0
        pseudo_rgb_weight = 0.0
        if schedule.is_update(iteration):
            torch.cuda.synchronize()
        refresh_start = time.monotonic()
        stage = prepare_training_label(
            controller, pseudo_pool, iteration,
            lambda camera: render(camera, gaussians, pipe, background),
            background.device,
        )
        # The teacher is needed only for refreshing labels. Move it off-device
        # before constructing/backpropagating the Gaussian training loss.
        joint_teacher.offload_to_cpu()
        use_pseudo = stage is not None and (
            args.pseudo_training_mode == "joint"
            or float(torch.rand((), generator=train_choice_generator)) >= args.pseudo_real_probability
        )
        refresh_record = controller.last_refresh
        if refresh_record is not None:
            torch.cuda.synchronize()
            refresh_record.update(
                anchor_camera_id=(anchor_ids if args.pseudo_training_mode == "difix" else anchor_id),
                target_camera_id=(target_ids if args.pseudo_training_mode == "difix"
                                  else args.pseudo_target_index),
                refresh_seconds=time.monotonic() - refresh_start,
                predictor_call_count=joint_teacher.call_count,
                history_size=pseudo_pool.count,
                selected_generation_iteration=(stage.generation_iteration if use_pseudo else None),
            )
        if use_pseudo:
            render_pkg_pseudo = render(stage.camera, gaussians, pipe, background)
            rendered_rgb_pseudo = render_pkg_pseudo["render"]
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            pseudo_total, pseudo_details = compute_stage_loss(
                rendered_rgb_pseudo, rendered_depth_pseudo, stage,
                rgb_weight=args.pseudo_rgb_weight,
                depth_weight=args.depth_pseudo_weight,
                alpha=render_pkg_pseudo["alpha"][0],
                alpha_threshold=args.pseudo_alpha_threshold,
                rgb_full_valid_windows=args.pseudo_rgb_full_valid_windows,
            )
            if args.pseudo_training_mode == "difix" and not pseudo_total.requires_grad:
                use_pseudo = False
            elif args.pseudo_training_mode == "difix":
                loss = pseudo_total
                viewspace_point_tensor = render_pkg_pseudo["viewspace_points"]
                visibility_filter = render_pkg_pseudo["visibility_filter"]
                radii = render_pkg_pseudo["radii"]
            else:
                loss = loss + pseudo_total
            if use_pseudo:
                pseudo_train_steps += 1
            rgb_details = pseudo_details["rgb"]
            pseudo_depth_raw = pseudo_details["depth"]
            pseudo_rgb_raw = rgb_details["loss"]
            if pseudo_depth_raw is not None:
                pseudo_depth_steps += 1
                pseudo_depth_weight = args.depth_pseudo_weight
                pseudo_depth_weighted = pseudo_depth_weight * pseudo_depth_raw
            if pseudo_rgb_raw is not None:
                pseudo_rgb_steps += 1
                pseudo_rgb_weight = args.pseudo_rgb_weight
                pseudo_rgb_weighted = pseudo_rgb_weight * pseudo_rgb_raw
            if refresh_record is not None:
                refresh_record.update(
                    depth_loss=(None if pseudo_depth_raw is None else float(pseudo_depth_raw.detach())),
                    rgb_loss=(None if pseudo_rgb_raw is None else float(pseudo_rgb_raw.detach())),
                    rgb_valid_pixels=rgb_details["valid_pixels"],
                    depth_valid_pixels=pseudo_details["depth_valid_pixels"],
                    rgb_skip_reason=rgb_details["skip_reason"],
                )
        if stage is not None and iteration in args.pseudo_snapshot_iterations:
            if not use_pseudo:
                with torch.no_grad():
                    snapshot_render = render(stage.camera, gaussians, pipe, background)
                    rendered_rgb_pseudo = snapshot_render["render"]
                    rendered_depth_pseudo = snapshot_render["depth"][0]
            snapshot_dir = os.path.join(pseudo_snapshot_root, f"iter_{iteration:05d}")
            save_pseudo_snapshot(snapshot_dir, iteration, stage,
                                 rendered_rgb_pseudo, rendered_depth_pseudo,
                                 (anchor_ids if args.pseudo_training_mode == "difix" else anchor_id),
                                 (target_ids if args.pseudo_training_mode == "difix"
                                  else args.pseudo_target_index))
        if refresh_record is not None:
            refresh_record["selected_generation_iteration"] = (
                stage.generation_iteration if use_pseudo else None
            )
            with open(pseudo_stats_path, "a", encoding="utf-8") as stats_file:
                stats_file.write(json.dumps(refresh_record) + "\n")
            if iteration == schedule.first or iteration % 100 == 0:
                print(f"[ProgressivePseudo] {json.dumps(refresh_record)}", flush=True)

        if iteration % 100 == 0:
            component_record = {
                "iteration": iteration,
                "real_rgb_l1": float(Ll1.detach()),
                "real_rgb_dssim": float(real_rgb_dssim.detach()),
                "real_rgb_raw": float(real_rgb_loss.detach()),
                "real_rgb_weight": 0.0 if use_pseudo and args.pseudo_training_mode == "difix" else 1.0,
                "real_rgb_weighted": (0.0 if use_pseudo and args.pseudo_training_mode == "difix"
                                      else float(real_rgb_loss.detach())),
                "real_depth_raw": float(depth_loss.detach()),
                "real_depth_weight": (0.0 if use_pseudo and args.pseudo_training_mode == "difix"
                                      else float(real_depth_weight)),
                "real_depth_weighted": (0.0 if use_pseudo and args.pseudo_training_mode == "difix"
                                         else float(real_depth_weighted.detach())),
                "pseudo_depth_raw": None if pseudo_depth_raw is None else float(pseudo_depth_raw.detach()),
                "pseudo_depth_weight": float(pseudo_depth_weight),
                "pseudo_depth_weighted": (None if pseudo_depth_weighted is None
                                           else float(pseudo_depth_weighted.detach())),
                "pseudo_rgb_raw": None if pseudo_rgb_raw is None else float(pseudo_rgb_raw.detach()),
                "pseudo_rgb_weight": float(pseudo_rgb_weight),
                "pseudo_rgb_weighted": (None if pseudo_rgb_weighted is None
                                         else float(pseudo_rgb_weighted.detach())),
                "total_loss": float(loss.detach()),
                "pseudo_history_size": pseudo_pool.count,
                "pseudo_selected_generation_iteration": (stage.generation_iteration
                                                         if use_pseudo else None),
            }
            with open(loss_components_path, "a", encoding="utf-8") as loss_file:
                loss_file.write(json.dumps(component_record) + "\n")


        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save({
                    "controller": controller.state_dict(),
                    "pseudo_pool": (pseudo_pool.state_dict()
                                    if args.pseudo_history_mode == "accumulate" else None),
                    "pseudo_pool_rng_state": (pseudo_pool.generator.get_state()
                                              if args.pseudo_history_mode == "replace" else None),
                    "train_choice_state": (train_choice_generator.get_state()
                                           if args.pseudo_training_mode == "difix" else None),
                    "elapsed_training_seconds": prior_elapsed + time.monotonic() - training_start,
                    "pseudo_train_steps": pseudo_train_steps,
                    "pseudo_rgb_steps": pseudo_rgb_steps,
                    "pseudo_depth_steps": pseudo_depth_steps,
                },
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth.pseudo.pt")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()

    with open(pseudo_stats_path, encoding="utf-8") as stats_file:
        # Iteration IDs make summary counters reliable after an interrupted run is resumed.
        records = {row["iteration"]: row for line in stats_file if (row := json.loads(line))}
    elapsed = prior_elapsed + time.monotonic() - training_start
    summary = {
        "iterations": opt.iterations,
        "pseudo_update_interval": schedule.interval,
        "scheduled_updates": len(schedule.updates),
        "pseudo_update_attempts": len(records),
        "pseudo_labels_generated": sum(row.get("generated_count", int(row.get("status") == "generated"))
                                       for row in records.values()),
        "predictor_calls": sum(row.get("generated_count", int(bool(row.get("predictor_called"))))
                               for row in records.values()),
        "pseudo_training_steps": pseudo_train_steps,
        "pseudo_rgb_used_steps": pseudo_rgb_steps,
        "pseudo_depth_used_steps": pseudo_depth_steps,
        "anchor_camera_id": anchor_id,
        "target_camera_id": args.pseudo_target_index,
        "target_camera_ids": target_ids,
        "target_camera_count": args.pseudo_target_count,
        "target_camera_source": args.pseudo_target_source,
        "target_pose_file": args.pseudo_target_pose_file,
        "target_camera_names": [getattr(camera, "image_name", None) for camera in target_cameras],
        "pseudo_rgb_weight": args.pseudo_rgb_weight,
        "pseudo_depth_weight": args.depth_pseudo_weight,
        "pseudo_history_mode": args.pseudo_history_mode,
        "pseudo_history_size": pseudo_pool.count,
        "pseudo_training_mode": args.pseudo_training_mode,
        "pseudo_real_probability": args.pseudo_real_probability,
        "pseudo_pose_step": args.pseudo_pose_step,
        "depth_conversion": "1 - predictor_depth01",
        "training_seconds": elapsed,
        "mean_iteration_seconds": elapsed / max(1, opt.iterations),
        "checkpoint": args.joint_checkpoint,
    }
    with open(os.path.join(scene.model_path, "joint_summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"[JointSummary] {json.dumps(summary)}", flush=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--joint_checkpoint', type=str, default='/data/lilinwei/3D/A_our_denoiser/20261002_164657_train_rgbdepth_teacher_perimg/checkpoint/iter_014000')
    parser.add_argument('--joint_code_root', type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--joint_processing_res', type=int, default=768)
    parser.add_argument('--joint_denoise_steps', type=int, default=4)
    parser.add_argument('--joint_fp16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pseudo_rgb_weight', type=float, default=0.25)
    parser.add_argument('--pseudo_target_index', type=int, default=0,
                        help='Fixed target from the seeded FSGS pseudo-camera list')
    parser.add_argument('--pseudo_history_mode', choices=('replace', 'accumulate'),
                        default='replace',
                        help='Use only the newest repaired label, or sample one from all past labels')
    parser.add_argument('--pseudo_training_mode', choices=('joint', 'difix'), default='joint')
    parser.add_argument('--pseudo_target_count', type=int, default=1)
    parser.add_argument('--pseudo_target_source', choices=('random', 'test', 'external'), default='random')
    parser.add_argument('--pseudo_target_pose_file', type=str, default=None,
                        help='JSON camera-to-world target trajectory for external target mode')
    parser.add_argument('--pseudo_pose_step', type=float, default=0.5)
    parser.add_argument('--pseudo_real_probability', type=float, default=0.7)
    parser.add_argument('--pseudo_rgb_alpha_threshold', type=float, default=0.01)
    parser.add_argument('--pseudo_rgb_full_valid_windows', action='store_true')
    parser.add_argument('--pseudo_snapshot_iterations', nargs='*', type=int,
                        default=[2001, 4000, 6000, 8000])
    parser.add_argument('--pseudo_alpha_threshold', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=2024)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if not 0.0 <= args.pseudo_real_probability <= 1.0:
        parser.error('--pseudo_real_probability must be in [0, 1]')
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
