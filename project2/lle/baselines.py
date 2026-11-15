"""Baseline methods (no learning).

The main baseline required by the assignment is histogram equalisation. We
equalise the luminance channel Y (YCrCb) so that the hue is not changed. The
other methods are used for additional comparison in the documentation:

* ``histeq_rgb``   - equalisation of every RGB channel separately (shows colour distortion),
* ``clahe``        - contrast limited adaptive histogram equalisation (L channel in Lab),
* ``exposure``     - "physical oracle": undoing -3 EV in the linear space, no denoising,
* ``exposure_nlm`` - as above + classical Non-Local Means denoising,
* ``isp_nlm``      - physical pipeline without learning: exposure x8 + NLM, then
  the rawpy ISP (auto-bright from the 99th percentile + BT.709 curve) - a "classical"
  version of what the ``isp`` network variant does.

All functions take and return an RGB uint8 HxWx3 image.
"""
from __future__ import annotations

import cv2
import numpy as np

from .raw_pipeline import auto_bright_gain, render_isp


def histeq_y(img: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def histeq_rgb(img: np.ndarray) -> np.ndarray:
    return np.stack([cv2.equalizeHist(img[..., c]) for c in range(3)], axis=-1)


def clahe(img: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def exposure(img: np.ndarray, ev_shift: float = -3.0, gamma: float = 2.2) -> np.ndarray:
    lin = (img.astype(np.float32) / 255.0) ** gamma * 2.0 ** (-ev_shift)
    return np.clip(np.round(np.clip(lin, 0, 1) ** (1 / gamma) * 255), 0, 255).astype(np.uint8)


def exposure_nlm(img: np.ndarray, h: float = 12.0) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(exposure(img), None, h, h, 7, 21)


def isp_nlm(img: np.ndarray, h: float = 12.0) -> np.ndarray:
    den = exposure_nlm(img, h).astype(np.float32) / 255.0
    lin = den ** 2.2
    out = render_isp(lin, auto_bright_gain(lin))
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


BASELINES = {
    "histeq": histeq_y,  # baseline from the assignment
    "histeq_rgb": histeq_rgb,
    "clahe": clahe,
    "exposure": exposure,
    "exposure_nlm": exposure_nlm,
    "isp_nlm": isp_nlm,
}
