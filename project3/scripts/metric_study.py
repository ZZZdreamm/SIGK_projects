"""How do FLIP / LPIPS / SSIM / Hausdorff react to controlled distortions?

Distortions of the test references whose perceptual severity is known in
advance (a 1 px shift is almost invisible, a missing highlight or a wrong
colour is obvious). A metric that "understands" quality should rank them the
same way. Output: results/metric_study.csv, results/metric_study.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nr import scene
from nr.data import get_split, load_dataset, vector_to_params
from nr.metrics import LPIPS, batch_metrics
from scripts.evaluate import crop, crop_boxes

N = 150


def q8(x):
    return np.round(np.clip(x, 0, 1) * 255) / 255


def shift(img, dx):
    """Shift right by dx px, the uncovered column is black (no wrap-around)."""
    out = np.zeros_like(img)
    out[:, dx:] = img[:, :-dx]
    return out


DISTORTIONS = {
    'shift_1px': ('przesunięcie o 1 px', lambda r, p: shift(r, 1)),
    'shift_3px': ('przesunięcie o 3 px', lambda r, p: shift(r, 3)),
    'blur': ('rozmycie Gaussa σ=1', lambda r, p: cv2.GaussianBlur(r, (0, 0), 1.0)),
    'noise': ('szum Gaussa σ=0,02 (cały obraz)', lambda r, p: r + np.random.default_rng(0).normal(0, 0.02, r.shape)),
    'darker': ('jasność ×0,8', lambda r, p: r * 0.8),
    'wrong_color': ('zły kolor (zamiana kanałów RGB→BRG)', lambda r, p: r[..., [2, 0, 1]]),
    'no_specular': ('brak odbłysku', None),
    'stray_pixel': ('1 jasny piksel (10, 10)', None),
    'black': ('brak kuli (czarny obraz)', lambda r, p: np.zeros_like(r)),
}


def main():
    files, params, images = load_dataset('data/dataset')
    test = get_split('data/dataset')['test'][:N]
    params, refs = params[test], images[test].astype(np.float32) / 255.0
    lp = LPIPS('cuda')
    boxes = crop_boxes(params)
    rows = []
    for key, (desc, fn) in DISTORTIONS.items():
        if key == 'no_specular':
            ref = np.stack([scene.phong_reference(vector_to_params(p)) for p in params])
            dist = np.stack([scene.phong_reference(vector_to_params(p), specular=False) for p in params])
            ref = q8(ref)
        elif key == 'stray_pixel':
            ref = refs
            dist = refs.copy()
            dist[:, 10, 10] = 1.0
        else:
            ref = refs
            dist = np.stack([fn(r, p) for r, p in zip(refs, params)])
        dist = q8(dist).astype(np.float32)
        m = batch_metrics(ref, dist, lp)
        mc = batch_metrics(crop(ref, boxes), crop(dist, boxes), lp)
        rows.append({'distortion': key, 'opis': desc, **{k: float(v.mean()) for k, v in m.items()},
                     **{f'{k}_crop': float(v.mean()) for k, v in mc.items()}})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = Path('results')
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'metric_study.csv', index=False)
    with open(out / 'metric_study.md', 'w') as f:
        for suffix, title in (('', 'cały obraz 128×128'), ('_crop', 'wycinek 64×64 wokół kuli')):
            f.write(f'\n**{title}**\n\n| Zniekształcenie | FLIP ↓ | LPIPS ↓ | SSIM ↑ | Hausdorff [px] ↓ |\n|---|---|---|---|---|\n')
            for r in rows:
                f.write(f"| {r['opis']} | {r['FLIP' + suffix]:.4f} | {r['LPIPS' + suffix]:.4f} | "
                        f"{r['SSIM' + suffix]:.4f} | {r['Hausdorff' + suffix]:.2f} |\n")
    print(open(out / 'metric_study.md').read())


if __name__ == '__main__':
    main()
