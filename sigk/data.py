"""Datasets of 256x256 DIV2K patches with on-the-fly degradation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tasks import TaskSpec
from .utils import hwc_to_tensor, list_images, read_image


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random horizontal/vertical flips and 90-degree rotations."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
    return np.ascontiguousarray(img)


class PatchDataset(Dataset):
    """Yields ``(x, y, mask, idx)`` where

    * ``x``    - network input, C x H x W (C = spec.in_channels)
    * ``y``    - clean target, 3 x H x W
    * ``mask`` - 1 x H x W hole mask for inpainting (zeros otherwise)
    * ``idx``  - patch index

    With ``deterministic=True`` the random degradation (noise, holes) of patch i
    only depends on ``(seed, i)``, so the baseline and the network are always
    evaluated on *exactly* the same degraded input. For training, call
    ``set_epoch`` every epoch to get fresh degradations and augmentations.

    With ``with_baseline=True`` the baseline output is computed inside the
    dataset (and therefore in parallel across DataLoader workers) and returned
    as a 5th element.
    """

    def __init__(
        self,
        root: str | Path,
        spec: TaskSpec,
        train: bool = False,
        deterministic: bool = True,
        seed: int = 0,
        limit: int | None = None,
        with_baseline: bool = False,
    ):
        self.files = list_images(root)
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")
        self.spec = spec
        self.train = train
        self.deterministic = deterministic
        self.seed = seed
        self.epoch = 0
        self.with_baseline = with_baseline

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng([self.seed, idx])
        return np.random.default_rng([self.seed, self.epoch, idx])

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        img = read_image(self.files[idx])
        if self.train:
            img = augment(img, rng)
        d = self.spec.degrade(img, rng)
        x = hwc_to_tensor(self.spec.model_input(d))
        y = hwc_to_tensor(img)
        mask = hwc_to_tensor(d.mask) if d.mask is not None else torch.zeros(1, *img.shape[:2])
        if self.with_baseline:
            return x, y, mask, idx, hwc_to_tensor(self.spec.baseline(d))
        return x, y, mask, idx
