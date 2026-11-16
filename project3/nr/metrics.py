"""FLIP, LPIPS, SSIM and the Hausdorff distance between Canny edge maps.

All functions take float images in [0, 1], HxWx3 numpy (LPIPS: batched torch).
"""
from __future__ import annotations

import cv2
import flip_evaluator as flip
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity

CANNY_LOW, CANNY_HIGH = 50, 150


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def flip_error(ref: np.ndarray, test: np.ndarray, return_map: bool = False):
    """Mean LDR-FLIP (NVlabs, default 67 ppd). Lower is better."""
    err_map, mean, _ = flip.evaluate(ref.astype(np.float32), test.astype(np.float32), 'LDR',
                                     applyMagma=return_map)
    return (float(mean), err_map) if return_map else float(mean)


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(ref, test, channel_axis=2, data_range=1.0))


def canny(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0


def hausdorff(ref: np.ndarray, test: np.ndarray) -> float:
    """Symmetric Hausdorff distance [px] between the Canny edge pixels.

    Undefined if one of the edge maps is empty: both empty -> 0, only one empty
    -> image diagonal (the largest possible distance) as a penalty."""
    a, b = np.argwhere(canny(ref)), np.argwhere(canny(test))
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float(np.hypot(*ref.shape[:2]))
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


class LPIPS:
    """LPIPS (AlexNet, v0.1) - the standard variant from the paper."""

    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] in [0,1] -> [B]"""
        return self.net(ref.to(self.device) * 2 - 1, test.to(self.device) * 2 - 1).flatten()

    def np(self, ref: np.ndarray, test: np.ndarray) -> float:
        t = lambda x: torch.from_numpy(x).float().permute(2, 0, 1)[None]
        return float(self(t(ref), t(test))[0])


def all_metrics(ref: np.ndarray, test: np.ndarray, lp: LPIPS) -> dict[str, float]:
    return {'FLIP': flip_error(ref, test), 'LPIPS': lp.np(ref, test),
            'SSIM': ssim(ref, test), 'Hausdorff': hausdorff(ref, test)}


def batch_metrics(refs: np.ndarray, tests: np.ndarray, lp: LPIPS) -> dict[str, np.ndarray]:
    """refs/tests: [N,H,W,3] float in [0,1]. Per-image metrics."""
    out = {'FLIP': [], 'SSIM': [], 'Hausdorff': []}
    for r, t in zip(refs, tests):
        out['FLIP'].append(flip_error(r, t))
        out['SSIM'].append(ssim(r, t))
        out['Hausdorff'].append(hausdorff(r, t))
    tr = torch.from_numpy(refs).float().permute(0, 3, 1, 2)
    tt = torch.from_numpy(tests).float().permute(0, 3, 1, 2)
    out['LPIPS'] = torch.cat([lp(tr[i:i + 100], tt[i:i + 100]) for i in range(0, len(tr), 100)]).cpu().numpy()
    return {k: np.asarray(out[k]) for k in ('FLIP', 'LPIPS', 'SSIM', 'Hausdorff')}
