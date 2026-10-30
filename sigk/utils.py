from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1], shape HxWx3."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def write_image(path: str | Path, img: np.ndarray) -> None:
    """Write a float32 RGB image in [0, 1] as PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    u8 = to_uint8(img)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def hwc_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWxC float numpy -> CxHxW float tensor."""
    if img.ndim == 2:
        img = img[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    """CxHxW tensor -> HxWxC float32 numpy clipped to [0, 1]."""
    return np.clip(t.detach().float().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cpu_count() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
