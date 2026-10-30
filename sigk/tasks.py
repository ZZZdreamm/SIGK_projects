"""Task registry: ties together degradation, baseline and model I/O for each
of the four project tasks.

A task is identified by a string ``<task>:<param>``, e.g.::

    sr:32          super-resolution from 32x32  (x8)
    sr:64          super-resolution from 64x64  (x4)
    denoise:0.01   Gaussian noise, sigma = 0.01
    denoise:0.03   Gaussian noise, sigma = 0.03
    deblur:3       Gaussian blur, 3x3 kernel
    deblur:5       Gaussian blur, 5x5 kernel
    inpaint:3      random 3x3 holes
    inpaint:32     random 32x32 holes
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import baselines as B
from . import degradations as D

IMAGE_SIZE = 256

# Number of holes used for inpainting so that roughly 9-10 % of the pixels are
# removed for both hole sizes (3x3: 700 * 9 px, 32x32: 6 * 1024 px of 65536).
DEFAULT_N_HOLES = {3: 700, 32: 6}

TASK_NAMES = ("sr", "denoise", "deblur", "inpaint")
BASELINE_NAMES = {
    "sr": "bicubic",
    "denoise": "bilateral",
    "deblur": "richardson_lucy",
    "inpaint": "telea",
}


BASELINE_PARAMS_FILE = Path(__file__).resolve().parents[1] / "configs" / "baselines.json"


@lru_cache(maxsize=None)
def _baseline_params_file() -> dict:
    if BASELINE_PARAMS_FILE.exists():
        return json.loads(BASELINE_PARAMS_FILE.read_text())
    return {}


def load_baseline_params(task_name: str) -> dict:
    return dict(_baseline_params_file().get(task_name, {}).get("params", {}))


@dataclass
class Degraded:
    """Output of a degradation: ``image`` is always IMAGE_SIZE x IMAGE_SIZE x 3
    (for SR it is the bicubic-upscaled LR image, ``lr`` holds the raw LR input).
    ``mask`` (HxW, 1 = missing) is only set for inpainting."""

    image: np.ndarray
    mask: np.ndarray | None = None
    lr: np.ndarray | None = None


@dataclass(frozen=True)
class TaskSpec:
    task: str
    param: float
    n_holes: int = 0
    size: int = IMAGE_SIZE

    # ------------------------------------------------------------------ names
    @property
    def name(self) -> str:
        p = int(self.param) if float(self.param).is_integer() else self.param
        if self.task == "sr":
            return f"sr_{p}x{p}"
        if self.task == "denoise":
            return f"denoise_sigma_{self.param:g}"
        if self.task == "deblur":
            return f"deblur_gauss_{p}x{p}"
        if self.task == "inpaint":
            return f"inpaint_{p}x{p}"
        raise ValueError(self.task)

    @property
    def description(self) -> str:
        p = self.param
        if self.task == "sr":
            return f"super-resolution {int(p)}x{int(p)} -> {self.size}x{self.size} (x{self.size // int(p)}), downscale {D.DOWNSCALE_INTERPOLATION_NAME}"
        if self.task == "denoise":
            return f"Gaussian noise sigma={p:g}"
        if self.task == "deblur":
            return f"Gaussian blur {int(p)}x{int(p)} (sigma={D.gaussian_sigma(int(p)):.2f})"
        return f"{self.n_holes} random holes of {int(p)}x{int(p)} px"

    @property
    def baseline_name(self) -> str:
        return BASELINE_NAMES[self.task]

    def method_name(self, method: str) -> str:
        """Unambiguous method name for result tables, e.g. richardson_lucy_deblur_gauss_3x3."""
        return f"{method}_{self.name}"

    @property
    def in_channels(self) -> int:
        # inpainting additionally feeds the hole mask to the network
        return 4 if self.task == "inpaint" else 3

    # ------------------------------------------------------------- degrading
    def degrade(self, img: np.ndarray, rng: np.random.Generator) -> Degraded:
        """Create the degraded input from a clean HxWx3 image in [0, 1]."""
        if self.task == "sr":
            lr = D.downscale(img, int(self.param))
            return Degraded(image=D.upscale_bicubic(lr, self.size), lr=lr)
        if self.task == "denoise":
            return Degraded(image=D.gaussian_noise(img, self.param, rng))
        if self.task == "deblur":
            return Degraded(image=D.gaussian_blur(img, int(self.param)))
        if self.task == "inpaint":
            masked, mask = D.random_cutout(img, int(self.param), self.n_holes, rng)
            return Degraded(image=masked, mask=mask)
        raise ValueError(self.task)

    # -------------------------------------------------------------- baseline
    def baseline(self, d: Degraded, **params) -> np.ndarray:
        """Run the classical baseline. ``params`` override the defaults; if not
        given, the values tuned by ``scripts/tune_baselines.py`` are used
        (``configs/baselines.json``)."""
        params = {**load_baseline_params(self.name), **params}
        if self.task == "sr":
            return B.baseline_bicubic(d.lr, self.size)
        if self.task == "denoise":
            return B.baseline_bilateral(d.image, **params)
        if self.task == "deblur":
            return B.baseline_richardson_lucy(d.image, int(self.param), **params)
        if self.task == "inpaint":
            return B.baseline_telea(d.image, d.mask, **params)
        raise ValueError(self.task)

    # ------------------------------------------------------------- model I/O
    def model_input(self, d: Degraded) -> np.ndarray:
        """HxWxC network input (C = in_channels)."""
        if self.task == "inpaint":
            return np.concatenate([d.image, d.mask[:, :, None]], axis=2)
        return d.image

    def has_identity_row(self) -> bool:
        """Whether reporting metrics of the raw degraded input makes sense
        (for SR the 'input' at 256x256 *is* the bicubic baseline)."""
        return self.task != "sr"


def parse_task(s: str) -> TaskSpec:
    """Parse ``'<task>:<param>'`` (optionally ``'inpaint:32:n_holes'``)."""
    parts = s.split(":")
    if len(parts) < 2 or parts[0] not in TASK_NAMES:
        raise ValueError(f"Bad task spec '{s}'. Expected e.g. sr:32, denoise:0.01, deblur:3, inpaint:32")
    task = parts[0]
    param = float(parts[1])
    n_holes = 0
    if task == "inpaint":
        n_holes = int(parts[2]) if len(parts) > 2 else DEFAULT_N_HOLES.get(int(param), 0)
        if n_holes <= 0:
            raise ValueError("inpaint needs n_holes, e.g. inpaint:16:40")
    return TaskSpec(task=task, param=param, n_holes=n_holes)


ALL_TASKS = ["sr:32", "sr:64", "denoise:0.01", "denoise:0.03", "deblur:3", "deblur:5", "inpaint:3", "inpaint:32"]
