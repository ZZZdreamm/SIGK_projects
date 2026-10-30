"""Classical baseline methods required by the project description."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.restoration import denoise_bilateral, richardson_lucy

from .degradations import gaussian_psf, upscale_bicubic
from .utils import to_float, to_uint8


def baseline_bicubic(lr: np.ndarray, size: int) -> np.ndarray:
    """2.1 baseline: cv2.resize with bicubic interpolation."""
    return upscale_bicubic(lr, size)


def baseline_bilateral(noisy: np.ndarray, sigma_color: float = 0.1, sigma_spatial: float = 1.5) -> np.ndarray:
    """2.2 baseline: skimage.restoration.denoise_bilateral."""
    out = denoise_bilateral(
        noisy.astype(np.float64), sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=-1
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def baseline_richardson_lucy(blurred: np.ndarray, ksize: int, num_iter: int = 10, pad: int | None = None) -> np.ndarray:
    """2.3 baseline: skimage.restoration.richardson_lucy with the true Gaussian PSF,
    applied independently to each colour channel.

    skimage convolves with zero padding, which produces strong ringing at the
    image border, so the image is reflect-padded first and cropped afterwards.
    cv2.GaussianBlur itself uses reflect-101 borders, so this matches the
    forward model."""
    psf = gaussian_psf(ksize).astype(np.float64)
    pad = 2 * ksize if pad is None else pad
    src = cv2.copyMakeBorder(blurred.astype(np.float64), pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    out = np.empty_like(src)
    for c in range(src.shape[2]):
        out[..., c] = richardson_lucy(src[..., c], psf, num_iter=int(num_iter), clip=True)
    out = out[pad:-pad, pad:-pad] if pad else out
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def baseline_telea(masked: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """2.4 baseline: cv2.inpaint with cv2.INPAINT_TELEA."""
    u8 = to_uint8(masked)
    m8 = (mask > 0).astype(np.uint8) * 255
    out = cv2.inpaint(u8, m8, radius, cv2.INPAINT_TELEA)
    return to_float(out)
