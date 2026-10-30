"""Image degradation functions used to build the training/test pairs.

All functions take and return float32 RGB images with values in [0, 1].
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.util import random_noise

# ----------------------------------------------------------------------------
# 2.1 Super-resolution: downscaling
# ----------------------------------------------------------------------------
# Interpolation used to produce the low-resolution input. INTER_AREA is the
# recommended OpenCV method for shrinking images (it averages pixel areas and
# therefore does not alias like INTER_NEAREST / INTER_LINEAR do when the
# scale factor is large).
DOWNSCALE_INTERPOLATION = cv2.INTER_AREA
DOWNSCALE_INTERPOLATION_NAME = "cv2.INTER_AREA"


def downscale(img: np.ndarray, size: int, interpolation: int = DOWNSCALE_INTERPOLATION) -> np.ndarray:
    """Resize an HxWx3 image to size x size (e.g. 256 -> 32 or 64)."""
    return cv2.resize(img, (size, size), interpolation=interpolation)


def upscale_bicubic(img: np.ndarray, size: int) -> np.ndarray:
    """Bicubic upscaling back to size x size (the SR baseline)."""
    return np.clip(cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


# ----------------------------------------------------------------------------
# 2.2 Denoising: additive Gaussian noise
# ----------------------------------------------------------------------------
def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation ``sigma``
    (``skimage.util.random_noise`` takes the *variance*, hence sigma**2)."""
    noisy = random_noise(img, mode="gaussian", mean=0.0, var=float(sigma) ** 2, rng=rng, clip=True)
    return noisy.astype(np.float32)


# ----------------------------------------------------------------------------
# 2.3 Deblurring: Gaussian blur
# ----------------------------------------------------------------------------
def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """cv2.GaussianBlur with a ksize x ksize kernel. sigma=0 lets OpenCV derive
    sigma from the kernel size: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def gaussian_psf(ksize: int) -> np.ndarray:
    """The exact 2-D kernel used by ``gaussian_blur`` (needed by Richardson-Lucy)."""
    k = cv2.getGaussianKernel(ksize, 0)
    return (k @ k.T).astype(np.float32)


def gaussian_sigma(ksize: int) -> float:
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


# ----------------------------------------------------------------------------
# 2.4 Inpainting: random square cut-outs
# ----------------------------------------------------------------------------
def random_cutout(
    img: np.ndarray, hole: int, n_holes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``n_holes`` random ``hole x hole`` squares out of the image.

    Returns (masked_image, mask) where mask is HxW float32 with 1 = missing pixel.
    Missing pixels are set to 0 in the returned image.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    ys = rng.integers(0, h - hole + 1, size=n_holes)
    xs = rng.integers(0, w - hole + 1, size=n_holes)
    for y, x in zip(ys, xs):
        mask[y : y + hole, x : x + hole] = 1.0
    out = img.copy()
    out[mask > 0] = 0.0
    return out, mask
