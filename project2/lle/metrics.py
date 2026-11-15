"""Evaluation metrics: MSE, PSNR, SSIM (scikit-image) and LPIPS (AlexNet).

All metrics operate on float RGB images in [0, 1]. PSNR/SSIM are computed with
``data_range=1.0``; SSIM over the colour channels (``channel_axis=-1``).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from skimage.metrics import structural_similarity

METRIC_NAMES = ("mse", "psnr", "ssim", "lpips")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


PSNR_MAX = 100.0  # cap for identical images (MSE = 0), e.g. perfectly flat sky patches


def psnr_from_mse(m: float) -> float:
    return PSNR_MAX if m <= 10 ** (-PSNR_MAX / 10) else float(10 * np.log10(1.0 / m))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB for data_range = 1, capped at PSNR_MAX (otherwise a single
    identical image makes the dataset mean infinite)."""
    return psnr_from_mse(mse(a, b))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=-1))


def _quiet_lpips(lpips_module, net: str):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return lpips_module.LPIPS(net=net, verbose=False)


class LPIPS:
    """Wrapper around the ``lpips`` package (AlexNet backbone, the standard
    choice for *evaluation*; VGG is used for the training loss)."""

    def __init__(self, device: torch.device, net: str = "alex"):
        import warnings

        import lpips

        self.model = _quiet_lpips(lpips, net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """pred/target: Bx3xHxW in [0, 1] -> per-image LPIPS, shape (B,)."""
        pred = pred.to(self.device).float().clamp(0, 1) * 2 - 1
        target = target.to(self.device).float().clamp(0, 1) * 2 - 1
        return self.model(pred, target).flatten().cpu().numpy()


class MetricAccumulator:
    """Collects per-image metrics for several methods and reports means."""

    def __init__(self, lpips_fn: LPIPS | None):
        self.lpips_fn = lpips_fn
        self.values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.rows: list[dict] = []

    @torch.no_grad()
    def update(self, method: str, pred: torch.Tensor, target: torch.Tensor, ids=None) -> None:
        """pred/target: Bx3xHxW tensors in [0, 1]."""
        pred = pred.detach().float().clamp(0, 1)
        target = target.detach().float().clamp(0, 1)
        lp = self.lpips_fn(pred, target) if self.lpips_fn is not None else [float("nan")] * pred.shape[0]
        p = pred.cpu().numpy().transpose(0, 2, 3, 1)
        t = target.cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(p.shape[0]):
            row = {
                "method": method,
                "idx": int(ids[i]) if ids is not None else len(self.values[method]["mse"]),
                "mse": mse(p[i], t[i]),
                "psnr": psnr(p[i], t[i]),
                "ssim": ssim(p[i], t[i]),
                "lpips": float(lp[i]),
            }
            self.rows.append(row)
            for k in METRIC_NAMES:
                self.values[method][k].append(row[k])

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for method, d in self.values.items():
            out[method] = {k: float(np.mean(v)) for k, v in d.items()}
            out[method].update({f"{k}_std": float(np.std(v)) for k, v in d.items()})
            out[method]["n"] = len(d["mse"])
        return out
