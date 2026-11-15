"""Generation of (dark + noisy input, ground truth) pairs from RAW files.

The code is based directly on the script from the assignment - the three
functions marked as TODO (``reduce_exposure``, ``apply_gamma_correction``,
``convert_to_8bit``) have been implemented. Additionally:

* ``process_single_file`` also stores the linear image (16-bit PNG). This makes
  it possible to generate *fresh* noise in every training iteration
  (``degrade_torch`` - the same physics implemented in PyTorch on the GPU)
  instead of learning the same noise realisation over and over again.
* Noise is drawn from a generator with a fixed seed per file, so the dataset
  (in particular the test set) is fully reproducible.

Order of operations (physics of image formation):
  RAW -> demosaicing + white balance (rawpy) -> LINEAR image (proportional to the
  number of photons) -> -3 EV (x1/8 light) -> Poisson photon noise (variance
  proportional to the signal) + Gaussian read noise (constant variance) -> clip
  to [0, 1] (pixel saturation) -> gamma curve -> 8-bit quantisation.
Noise has to be added in the linear space - only there the number of collected
photons follows a Poisson distribution. Gamma is non-linear, so applying noise
after it would make it brightness-dependent in a way unlike a real sensor.
"""
from __future__ import annotations

import os

import imageio
import numpy as np
import rawpy

# --------------------------------------------------------------------------- #
#  Code from the assignment (with the TODOs implemented)                        #
# --------------------------------------------------------------------------- #

_rng = np.random.default_rng()


def get_ground_truth(raw):
    """
    Generates the properly exposed, full-color sRGB image (Ground Truth).
    half_size=True is used to speed up processing during training/experiments.
    """
    return raw.postprocess(use_camera_wb=True, half_size=True)


def add_physical_noise(linear_img, alpha=0.001, sigma=0.005, rng=None):
    """
    Simulates physical camera sensor noise.
    alpha: controls Poisson (photon) noise
    sigma: controls Gaussian (electronic read) noise
    """
    rng = rng if rng is not None else _rng
    noisy_img = rng.poisson(linear_img / alpha) * alpha
    noisy_img = noisy_img + rng.normal(0, sigma, linear_img.shape)
    return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)


def get_linear_image(raw):
    """
    Extracts the linear image from RAW data.
    Disables auto-brightness and gamma correction.
    Returns a float32 image in the range [0.0, 1.0].
    """
    linear_16bit = raw.postprocess(
        use_camera_wb=True,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        half_size=True,
    )
    return linear_16bit.astype(np.float32) / 65535.0


def reduce_exposure(linear_img, ev_shift):
    """
    Simulate exposure reduction.
    A change of -1 EV halves the amount of light, so the scale factor is 2^ev_shift
    (-3 EV -> 1/8). This is only valid in the LINEAR light space.
    """
    scale = 2.0 ** ev_shift
    return linear_img * np.float32(scale)


def apply_gamma_correction(img, gamma=2.2):
    """
    Apply (simplified) sRGB gamma correction: image ^ (1 / gamma).
    Values are clipped to [0, 1] first (negative values would give NaN).
    """
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def convert_to_8bit(img):
    """
    Convert the float image (range 0.0 to 1.0) to an 8-bit integer (range 0 to 255)
    with rounding to the nearest integer.
    """
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def get_low_light(raw, ev_shift=-3.0, rng=None, linear_img=None):
    """
    Generates the degraded, low-light input image.
    (``linear_img`` may be passed when it was already computed, to avoid a second
    demosaicing pass.)
    """
    # Step 1: Get linear image data
    linear_img = get_linear_image(raw) if linear_img is None else linear_img
    # Step 2: Mathematically reduce the exposure
    dark_linear = reduce_exposure(linear_img, ev_shift)
    # Step 3: Add physical sensor noise
    noisy_dark_linear = add_physical_noise(dark_linear, rng=rng)
    # Step 4: Apply gamma curve for correct display
    dark_srgb = apply_gamma_correction(noisy_dark_linear)
    # Step 5: Convert back to standard 8-bit format
    final_8bit = convert_to_8bit(dark_srgb)
    return final_8bit


def process_single_file(file_path, output_gt_dir, output_dark_dir, ev_shift,
                        output_linear_dir=None, seed=None):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    gt_path = os.path.join(output_gt_dir, f"{base_name}_gt.png")
    dark_path = os.path.join(output_dark_dir, f"{base_name}_input.png")
    lin_path = (os.path.join(output_linear_dir, f"{base_name}_linear.png")
                if output_linear_dir else None)

    if os.path.exists(gt_path) and os.path.exists(dark_path) and (lin_path is None or os.path.exists(lin_path)):
        print(f"Skipping {base_name} - already processed.")
        return

    print(f"Processing {base_name}...", flush=True)
    try:
        with rawpy.imread(file_path) as raw:
            img_gt = get_ground_truth(raw)
            linear = get_linear_image(raw)
        rng = np.random.default_rng(seed)
        img_dark = get_low_light(None, ev_shift, rng=rng, linear_img=linear)
        assert img_gt.shape == img_dark.shape, (img_gt.shape, img_dark.shape)

        imageio.imsave(gt_path, img_gt)
        imageio.imsave(dark_path, img_dark)
        if lin_path:
            # 16-bit PNG stores the ``output_bps=16`` result losslessly
            # (Pillow/imageio cannot write 16-bit RGB, hence OpenCV - BGR channel order)
            import cv2

            lin16 = np.clip(np.round(linear * 65535.0), 0, 65535).astype(np.uint16)
            cv2.imwrite(lin_path, np.ascontiguousarray(lin16[..., ::-1]))
    except Exception as e:  # noqa: BLE001
        print(f"Error processing {base_name}: {e}")


# --------------------------------------------------------------------------- #
#  The same degradation in PyTorch (on-the-fly noise generation for training)   #
# --------------------------------------------------------------------------- #

def degrade_torch(linear, ev_shift=-3.0, alpha=0.001, sigma=0.005, gamma=2.2,
                  quantize=True):
    """Equivalent of ``get_low_light`` for a Bx3xHxW tensor (linear, [0, 1]).

    ``ev_shift``, ``alpha`` and ``sigma`` may be scalars or Bx1x1x1 tensors
    (degradation parameters sampled separately for every example).
    """
    import torch

    dark = linear * (2.0 ** torch.as_tensor(ev_shift, device=linear.device, dtype=linear.dtype))
    alpha = torch.as_tensor(alpha, device=linear.device, dtype=linear.dtype)
    sigma = torch.as_tensor(sigma, device=linear.device, dtype=linear.dtype)
    noisy = torch.poisson(dark / alpha) * alpha + torch.randn_like(dark) * sigma
    out = noisy.clamp(0.0, 1.0).pow(1.0 / gamma)
    if quantize:
        out = torch.round(out * 255.0) / 255.0
    return out


# --------------------------------------------------------------------------- #
#  ISP model used by rawpy to create the ground truth                           #
# --------------------------------------------------------------------------- #
#
# ``raw.postprocess(use_camera_wb=True, half_size=True)`` with default
# parameters does two things to the linear image:
#   1. auto-bright (LibRaw): multiplies the image by k = 1 / max_c(percentile_99(lin_c)),
#      i.e. so that about 1% of the pixels (in the brightest channel) are clipped,
#   2. applies the BT.709 tone curve with gamma=(2.222, 4.5).
# Verified empirically: bt709(min(k * lin, 1)) reproduces the GT with PSNR ~ 55.6 dB
# (the remaining difference comes only from 8-bit quantisation).

AUTO_BRIGHT_THR = 0.01


def auto_bright_gain(linear, thr: float = AUTO_BRIGHT_THR):
    """k = 1 / max_c quantile_{1-thr}(linear_c); numpy HxWx3 or tensor Bx3xHxW
    (for a tensor returns Bx1x1x1). Requires the WHOLE image - it is a global operation."""
    if isinstance(linear, np.ndarray):
        q = max(float(np.quantile(linear[..., c], 1.0 - thr)) for c in range(3))
        return 1.0 / max(q, 1e-6)
    import torch

    flat = linear.detach().float().flatten(2)  # B x 3 x N
    n = flat.shape[-1]
    kth = max(1, int(round((1.0 - thr) * n)))
    q = flat.kthvalue(kth, dim=-1).values.amax(dim=1)  # B
    return (1.0 / q.clamp_min(1e-6)).view(-1, 1, 1, 1)


def bt709_curve(x):
    """BT.709 curve (gamma 1/2.222 with a linear segment of slope 4.5 near zero)."""
    if isinstance(x, np.ndarray):
        x = np.clip(x, 0.0, 1.0)
        return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(np.maximum(x, 0.018), 0.45) - 0.099)
    import torch

    x = x.clamp(0.0, 1.0)
    return torch.where(x < 0.018, 4.5 * x, 1.099 * x.clamp_min(0.018).pow(0.45) - 0.099)


def render_isp(linear, gain):
    """Linear image (original exposure) -> sRGB as in the GT."""
    return bt709_curve(linear * gain)
