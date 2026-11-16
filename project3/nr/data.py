"""Dataset: images rendered by the provided OpenGL renderer + scene parameters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nr import scene

N_TEST = 600
N_VAL = 200


def params_to_vector(d: dict) -> np.ndarray:
    """Raw parameter vector [obj xyz, light xyz, diffuse rgb (0..255), shininess]."""
    return np.array([*d['model_translation'], *d['light_position'],
                     *d['material_diffuse'], d['material_shininess']], dtype=np.float32)


def vector_to_params(v) -> scene.SceneParams:
    v = [float(x) for x in v]
    return scene.SceneParams(model_translation=v[0:3], light_position=v[3:6],
                             material_diffuse=[int(round(x)) for x in v[6:9]],
                             material_shininess=v[9])


def load_dataset(root: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (file names, params [N,10], images uint8 [N,H,W,3])."""
    root = Path(root)
    rows = [json.loads(l) for l in open(root / 'params.jsonl')]
    files = [r['file'] for r in rows]
    params = np.stack([params_to_vector(r) for r in rows])
    images = np.stack([np.asarray(Image.open(root / f).convert('RGB')) for f in files])
    return files, params, images


def make_split(n: int, seed: int = 0) -> dict[str, list[int]]:
    perm = np.random.default_rng(seed).permutation(n).tolist()
    return {'test': sorted(perm[:N_TEST]),
            'val': sorted(perm[N_TEST:N_TEST + N_VAL]),
            'train': sorted(perm[N_TEST + N_VAL:])}


def get_split(root: str | Path) -> dict[str, list[int]]:
    """Split is created once and stored next to the data, so all runs share it."""
    path = Path(root) / 'split.json'
    if not path.exists():
        n = sum(1 for _ in open(Path(root) / 'params.jsonl'))
        json.dump(make_split(n), open(path, 'w'))
    return json.load(open(path))


class GPUData:
    """The whole dataset fits on the GPU (3000×128×128×3 B ≈ 150 MB)."""

    def __init__(self, root, device):
        self.files, params, images = load_dataset(root)
        self.split = get_split(root)
        self.params = torch.from_numpy(params).to(device)
        self.images = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0

    def subset(self, name):
        idx = torch.tensor(self.split[name], device=self.params.device)
        return self.params[idx], self.images[idx]
