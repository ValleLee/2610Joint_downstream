import torch


def align_teacher_depth_orientation(
    depth01: torch.Tensor, *, inverse_representation: bool
) -> torch.Tensor:
    """Map teacher depth to the renderer's near-to-far z-depth orientation."""
    if inverse_representation:
        return 1.0 - depth01
    return depth01
