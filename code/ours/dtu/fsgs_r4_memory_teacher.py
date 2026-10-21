"""Frozen joint RGB-depth diffusion teacher for FSGS pseudo cameras."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from omegaconf import OmegaConf


class JointRgbdTeacher:
    """Run the trained 0930jointR model and return detached RGB/depth labels."""

    def __init__(
        self,
        checkpoint: str,
        code_root: str,
        device: torch.device,
        processing_res: int = 768,
        denoise_steps: int = 4,
        fp16: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.processing_res = int(processing_res)
        self.denoise_steps = int(denoise_steps)
        self.dtype = torch.float16 if fp16 else torch.float32
        root = Path(code_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Joint diffusion code root not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        # The shared xformers wheel is not ABI compatible with the active
        # torch build. Diffusers safely falls back to regular attention.
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        checkpoint_path = Path(checkpoint).resolve()
        if not (checkpoint_path / "unet").is_dir():
            raise FileNotFoundError(f"Joint checkpoint has no unet/: {checkpoint_path}")
        run_config = OmegaConf.load(checkpoint_path.parents[1] / "config.yaml")
        # Checkpoints moved from the original /data1 host may retain that
        # machine-specific base path. Allow an explicit local base-model root
        # without changing the archived training config.
        base_root = os.environ.get("JOINT_BASE_CKPT_DIR", run_config.runtime.base_ckpt_dir)
        base = Path(base_root) / run_config.model.pretrained_path
        if not base.is_dir():
            raise FileNotFoundError(f"Marigold base model not found: {base}")

        from marigold import MarigoldDepthPipeline  # imported from code_root

        pipe = MarigoldDepthPipeline.from_pretrained(
            str(base),
            variant="fp16" if fp16 else None,
            torch_dtype=self.dtype,
        )
        pipe.unet = UNet2DConditionModel.from_pretrained(
            str(checkpoint_path / "unet"), torch_dtype=self.dtype
        )
        scheduler_dir = checkpoint_path / "scheduler"
        if scheduler_dir.is_dir():
            pipe.scheduler = DDIMScheduler.from_pretrained(str(scheduler_dir))
        if int(pipe.unet.config.in_channels) != 16 or int(pipe.unet.config.out_channels) != 8:
            raise RuntimeError(
                "Expected joint diffusion UNet 16->8 channels, got "
                f"{pipe.unet.config.in_channels}->{pipe.unet.config.out_channels}"
            )
        self.pipe = pipe.to(device=self.device)
        self.pipe.unet.eval().requires_grad_(False)
        self.pipe.vae.eval().requires_grad_(False)
        self.pipe.text_encoder.eval().requires_grad_(False)
        self.pipe.encode_empty_text()
        self.call_count = 0
        print(
            f"[JointRgbdTeacher] checkpoint={checkpoint_path} device={self.device} "
            f"processing_res={self.processing_res} steps={self.denoise_steps} "
            f"dtype={self.dtype}",
            flush=True,
        )

    def offload_to_cpu(self) -> None:
        """Free GPU memory between label generation and Gaussian training."""
        if next(self.pipe.unet.parameters()).device.type == "cpu":
            return
        self.pipe.unet.to(device="cpu")
        self.pipe.vae.to(device="cpu")
        self.pipe.text_encoder.to(device="cpu")
        if self.pipe.empty_text_embed is not None:
            self.pipe.empty_text_embed = self.pipe.empty_text_embed.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _depth_condition(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = depth.detach()[valid]
        if values.numel() < 32:
            raise RuntimeError(f"Need at least 32 rendered depth pixels; got {values.numel()}")
        lo, hi = torch.quantile(values.float(), torch.tensor([0.01, 0.99], device=values.device))
        if not torch.isfinite(lo) or not torch.isfinite(hi) or float(hi - lo) <= 1e-8:
            raise RuntimeError(f"Invalid rendered depth range q01={lo}, q99={hi}")
        normalized = ((depth - lo.to(depth.dtype)) / (hi - lo).to(depth.dtype)).clamp(0.0, 1.0)
        # 0930jointR was trained with invalid depth placed at the far plane.
        return torch.where(valid, normalized, torch.ones_like(normalized))

    def _processing_size(self, h: int, w: int) -> tuple[int, int]:
        scale = min(1.0, self.processing_res / max(h, w))
        out_h = max(8, int(round(h * scale / 8.0)) * 8)
        out_w = max(8, int(round(w * scale / 8.0)) * 8)
        return out_h, out_w

    @torch.no_grad()
    def predict(
        self,
        rendered_rgb: torch.Tensor,
        rendered_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        if next(self.pipe.unet.parameters()).device != self.device:
            self.pipe.to(device=self.device)
        rgb = rendered_rgb.detach().float().clamp(0.0, 1.0)
        depth = rendered_depth.squeeze().float()
        valid = valid_mask.squeeze().bool() & torch.isfinite(depth) & (depth > 0)
        depth01 = self._depth_condition(depth, valid)
        original_size = tuple(depth.shape)
        process_size = self._processing_size(*original_size)
        if process_size != original_size:
            rgb_in = F.interpolate(rgb[None], size=process_size, mode="bilinear", align_corners=False)
            depth_in = F.interpolate(depth01[None, None], size=process_size, mode="bilinear", align_corners=False)
        else:
            rgb_in, depth_in = rgb[None], depth01[None, None]

        rgb_norm = rgb_in.mul(2.0).sub(1.0).to(self.device, self.dtype)
        depth_norm = depth_in.mul(2.0).sub(1.0).to(self.device, self.dtype)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        pred_rgb_norm, pred_depth_norm = self.pipe.predict_joint(
            rgb_norm,
            depth_norm,
            denoising_steps=self.denoise_steps,
            generator=generator,
        )
        pred_rgb = pred_rgb_norm.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        pred_depth = pred_depth_norm.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        if process_size != original_size:
            pred_rgb = F.interpolate(pred_rgb, size=original_size, mode="bilinear", align_corners=False)
            pred_depth = F.interpolate(pred_depth, size=original_size, mode="bilinear", align_corners=False)
        self.call_count += 1
        return {
            "rgb01": pred_rgb[0].detach().to(rendered_rgb.device, rendered_rgb.dtype),
            "depth01": pred_depth[0, 0].detach().to(rendered_depth.device, rendered_depth.dtype),
        }
