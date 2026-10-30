"""Training losses."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


class RestorationLoss(nn.Module):
    """``pixel`` loss (l1 / l2 / charbonnier) optionally combined with an LPIPS
    (VGG) perceptual term: ``loss = pixel + lpips_weight * LPIPS``."""

    def __init__(self, pixel: str = "l1", lpips_weight: float = 0.0):
        super().__init__()
        self.pixel = {"l1": nn.L1Loss(), "l2": nn.MSELoss(), "charbonnier": CharbonnierLoss()}[pixel]
        self.lpips_weight = lpips_weight
        self.lpips = None
        if lpips_weight > 0:
            import warnings

            import lpips

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.lpips = lpips.LPIPS(net="vgg", verbose=False)
            for p in self.lpips.parameters():
                p.requires_grad_(False)
            self.lpips.eval()

    def forward(self, pred, target) -> tuple[torch.Tensor, dict[str, float]]:
        pix = self.pixel(pred, target)
        logs = {"pixel": pix.item()}
        total = pix
        if self.lpips is not None:
            with torch.autocast(device_type=pred.device.type, enabled=False):
                p = self.lpips(pred.float().clamp(0, 1) * 2 - 1, target.float() * 2 - 1).mean()
            logs["lpips"] = p.item()
            total = total + self.lpips_weight * p
        return total, logs
