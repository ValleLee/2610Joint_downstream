"""Numerically safe real-depth Pearson loss for the isolated Mip-NeRF R4 run."""

import math

import torch
from torchmetrics.functional.regression import pearson_corrcoef


def _pearson_input_is_valid(values: torch.Tensor) -> bool:
    """Match TorchMetrics' scale-normalized near-zero-variance guard."""
    values = values.reshape(-1).float()
    if values.numel() < 2 or not bool(torch.isfinite(values).all().item()):
        return False

    centered = values - values.mean()
    max_abs_deviation = centered.abs().max()
    if not bool(torch.isfinite(max_abs_deviation).item()) or max_abs_deviation.item() == 0:
        return False

    sample_variance = centered.square().sum() / (values.numel() - 1)
    normalized_variance = sample_variance / max_abs_deviation.square()
    threshold = math.sqrt(torch.finfo(values.dtype).eps)
    return bool((torch.isfinite(normalized_variance) & (normalized_variance >= threshold)).item())


def stable_pearson_depth_loss(target_depth: torch.Tensor, rendered_depth: torch.Tensor) -> torch.Tensor:
    """Use the original min-of-two Pearson losses, skipping degenerate inputs.

    Pearson correlation is undefined when either depth vector has negligible
    normalized variance. Such a candidate is omitted. If both transformed
    targets are degenerate, the depth term contributes zero for this step.
    """
    rendered_depth = rendered_depth.reshape(-1, 1)
    targets = (
        -target_depth.reshape(-1, 1),
        1.0 / (target_depth.reshape(-1, 1) + 200.0),
    )

    if not _pearson_input_is_valid(rendered_depth):
        return rendered_depth.sum() * 0.0

    losses = []
    for target in targets:
        if _pearson_input_is_valid(target):
            corr = pearson_corrcoef(target.float(), rendered_depth.float())
            if bool(torch.isfinite(corr.detach()).all().item()):
                losses.append(1.0 - corr)

    if not losses:
        return rendered_depth.sum() * 0.0
    return torch.stack(losses).min()
