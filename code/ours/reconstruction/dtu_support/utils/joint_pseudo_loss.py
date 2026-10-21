"""Losses and fixed schedules for the joint teacher's pseudo-camera labels."""

import torch
import torch.nn.functional as F

from utils.loss_utils import ssim, ssim_map


def pseudo_active(iteration, start=2000, end=9500):
    return start < iteration < end


def pseudo_weights(iteration, start=2000, end=9500, depth_max=0.5, rgb_max=0.5,
                   rgb_start=3000, rgb_ramp_iters=2000,
                   rgb_schedule="standard", rgb_taper_start=5000,
                   rgb_taper_end=6000):
    if not pseudo_active(iteration, start, end):
        return 0.0, 0.0
    depth = depth_max * min(max((iteration - start) / 500.0, 0.0), 1.0)
    if rgb_schedule == "shared_depth_ramp":
        rgb = rgb_max * min(max((iteration - start) / 500.0, 0.0), 1.0)
    elif rgb_schedule == "standard":
        rgb = rgb_max * min(max((iteration - rgb_start) / float(rgb_ramp_iters), 0.0), 1.0)
    elif rgb_schedule == "taper_5k_6k":
        if iteration <= rgb_taper_start:
            rgb = rgb_max * min(max((iteration - rgb_start) / float(rgb_ramp_iters), 0.0), 1.0)
        elif iteration < rgb_taper_end:
            rgb = rgb_max * (rgb_taper_end - iteration) / float(rgb_taper_end - rgb_taper_start)
        else:
            rgb = 0.0
    else:
        raise ValueError(f"Unsupported pseudo RGB schedule: {rgb_schedule}")
    return depth, rgb


def valid_render_mask(alpha, depth, threshold=0.01):
    return (alpha > threshold) & torch.isfinite(depth) & (depth > 0)


def masked_pearson_loss(rendered_depth, target_depth, mask, min_pixels=32):
    valid = mask.bool() & torch.isfinite(rendered_depth) & torch.isfinite(target_depth)
    if int(valid.sum()) < min_pixels:
        return None
    x = rendered_depth[valid].float()
    y = target_depth.detach()[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    norm = torch.sqrt(x.square().sum() * y.square().sum())
    if not bool(torch.isfinite(norm.detach())) or float(norm.detach()) <= 1e-8:
        return None
    return 1.0 - (x * y).sum().div(norm).clamp(-1.0, 1.0)


def masked_rgb_loss_stats(rendered_rgb, target_rgb, mask, dssim_weight=0.2,
                          full_valid_windows=False, min_pixels=32, min_windows=32):
    valid = mask.bool() & torch.isfinite(rendered_rgb).all(dim=0) & torch.isfinite(target_rgb).all(dim=0)
    valid_pixels = int(valid.sum())
    coverage = valid.float().mean()
    if valid_pixels < min_pixels:
        return {"loss": None, "l1": None, "dssim": None, "mask_fraction": coverage,
                "window_fraction": None, "valid_pixels": valid_pixels, "valid_windows": 0,
                "skip_reason": "fewer_than_min_valid_pixels"}
    target = torch.nan_to_num(target_rgb.detach(), nan=0.0, posinf=1.0, neginf=0.0)
    rendered = torch.nan_to_num(rendered_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    mask_ch = valid[None].to(rendered.dtype)
    l1 = ((rendered - target).abs() * mask_ch).sum() / (valid_pixels * rendered.shape[0])
    if full_valid_windows:
        window = torch.ones((1, 1, 11, 11), device=valid.device, dtype=torch.float32)
        counts = F.conv2d(valid[None, None].float(), window, padding=5)
        valid_window = counts >= 121.0
        valid_windows = int(valid_window.sum())
        window_fraction = valid_window.float().mean()
        if valid_windows < min_windows:
            return {"loss": None, "l1": None, "dssim": None, "mask_fraction": coverage,
                    "window_fraction": window_fraction, "valid_pixels": valid_pixels,
                    "valid_windows": valid_windows, "skip_reason": "fewer_than_min_valid_ssim_windows"}
        ssim_values = ssim_map(rendered[None], target[None])
        selected = valid_window.expand(-1, ssim_values.shape[1], -1, -1)
        dssim = 1.0 - ssim_values[selected].mean()
    else:
        # Preserve the original white-fill masked SSIM convention.
        dssim = 1.0 - ssim(rendered[None], target[None], mask=mask_ch[None])
        valid_windows = 0
        window_fraction = None
    total = (1.0 - dssim_weight) * l1 + dssim_weight * dssim
    return {"loss": total, "l1": l1, "dssim": dssim, "mask_fraction": coverage,
            "window_fraction": window_fraction, "valid_pixels": valid_pixels,
            "valid_windows": valid_windows, "skip_reason": None}


def masked_rgb_loss(rendered_rgb, target_rgb, mask, dssim_weight=0.2):
    """Backward-compatible masked RGB loss used by the original pseudo-RGB path."""
    result = masked_rgb_loss_stats(rendered_rgb, target_rgb, mask, dssim_weight,
                                   full_valid_windows=False, min_pixels=1)
    if result["loss"] is None:
        return None
    return result["loss"], result["l1"], result["dssim"], result["mask_fraction"]
