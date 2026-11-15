"""Datasets.

* ``PatchDataset`` - random crops for training. All images are kept in RAM
  (72 images ~ 3.5 GB). Returns a crop of the linear image, the GT and the fixed
  input, plus the auto-bright gain of the given (whole) image. The ``online`` mode
  (in train.py) generates the degradation on the GPU from the linear image
  (``raw_pipeline.degrade_torch``) - every epoch sees a new noise realisation;
  the ``fixed`` mode uses only the ``*_input.png`` files generated once.
* ``FullImageDataset`` - full images (val/test).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_pipeline import auto_bright_gain
from .utils import load_json


def read_rgb(path, unchanged=False) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_split(root, split: str) -> list[str]:
    return load_json(Path(root) / "split.json")[split]


def paths(root, name):
    root = Path(root)
    return root / "input" / f"{name}_input.png", root / "gt" / f"{name}_gt.png", root / "linear" / f"{name}_linear.png"


def to_tensor(img: np.ndarray) -> torch.Tensor:
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / scale


class PatchDataset(Dataset):
    def __init__(self, root, split="train", patch=256, crops_per_image=32):
        self.names = load_split(root, split)
        self.patch, self.cpi = patch, crops_per_image
        self.inp, self.gt, self.lin, self.gain = [], [], [], []
        for n in self.names:
            pi, pg, pl = paths(root, n)
            self.inp.append(read_rgb(pi))
            self.gt.append(read_rgb(pg))
            self.lin.append(read_rgb(pl, unchanged=True))
            # auto-bright gain computed from the WHOLE clean image (a crop does not know it)
            self.gain.append(auto_bright_gain(self.lin[-1].astype(np.float32) / 65535.0))

    def __len__(self):
        return len(self.names) * self.cpi

    def __getitem__(self, idx):
        i = idx % len(self.names)
        rng = np.random.default_rng()
        h, w = self.gt[i].shape[:2]
        p = self.patch
        y, x = rng.integers(0, h - p + 1), rng.integers(0, w - p + 1)
        crops = [a[y:y + p, x:x + p] for a in (self.inp[i], self.gt[i], self.lin[i])]
        # augmentation: flips and 90-degree rotations (they change neither noise statistics nor colours)
        k = int(rng.integers(4))
        flip = bool(rng.integers(2))
        out = []
        for c in crops:
            c = np.rot90(c, k)
            if flip:
                c = c[:, ::-1]
            out.append(to_tensor(c))
        return {"input": out[0], "gt": out[1], "linear": out[2],
                "gain": torch.tensor(self.gain[i], dtype=torch.float32).view(1, 1, 1)}


class FullImageDataset(Dataset):
    def __init__(self, root, split="test", crop: int = 0):
        self.root, self.names, self.crop = root, load_split(root, split), crop

    def __len__(self):
        return len(self.names)

    def _crop(self, a):
        if not self.crop:
            return a
        h, w = a.shape[:2]
        c = self.crop
        y, x = (h - c) // 2, (w - c) // 2
        return a[y:y + c, x:x + c]

    def __getitem__(self, i):
        n = self.names[i]
        pi, pg, _ = paths(self.root, n)
        return {"name": n, "input": to_tensor(self._crop(read_rgb(pi))), "gt": to_tensor(self._crop(read_rgb(pg)))}
