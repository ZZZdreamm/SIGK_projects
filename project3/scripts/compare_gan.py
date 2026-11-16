"""Paired comparison GAN vs. pure L1 regression on the 600 test images.

For each encoding the two generators predict the *same* test images, so the
per-image metric differences are paired -> Wilcoxon signed-rank test.
Additionally: sharpness = var(Laplacian(pred)) / var(Laplacian(ref)) on the
64×64 crop around the sphere (<1 = blurrier than the reference, the classic
failure of L1/L2 regression that an adversarial loss should fix).

Needs results/per_image_<run>.csv and results/preds/<run>.npy from evaluate.py.
Output: results/gan_vs_l1.csv, results/gan_vs_l1.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nr.data import get_split, load_dataset
from scripts.evaluate import crop, crop_boxes

PAIRS = [
    ('geometric_gan', 'geometric_l1', 'geometryczne, GAN v1'),
    ('geometric_gan_v2', 'geometric_l1', 'geometryczne, GAN v2'),
    ('camera_gan', 'camera_l1', 'względne (kamera), GAN v1'),
    ('camera_gan_v2', 'camera_l1', 'względne (kamera), GAN v2'),
    ('raw_gan', 'raw_l1', 'surowe, GAN v1'),
    ('raw_gan_v2', 'raw_l1', 'surowe, GAN v2'),
]
METRICS = {'FLIP_crop': -1, 'LPIPS_crop': -1, 'SSIM_crop': 1, 'Hausdorff': -1, 'sharpness': 0}
R = Path('results')


def sharpness(pred8, ref8, boxes):
    def lap_var(x):
        g = x.astype(np.float32).mean(-1) / 255
        return np.array([cv2.Laplacian(im, cv2.CV_32F).var() for im in g])
    return lap_var(crop(pred8, boxes)) / (lap_var(crop(ref8, boxes)) + 1e-12)


def main():
    _, params, images = load_dataset('data/dataset')
    test = get_split('data/dataset')['test']
    params, images = params[test], images[test]
    boxes = crop_boxes(params)
    cache = {}

    def load(run):
        if run not in cache:
            df = pd.read_csv(R / f'per_image_{run}.csv')
            df['sharpness'] = sharpness(np.load(R / 'preds' / f'{run}.npy'), images, boxes)
            cache[run] = df
        return cache[run]

    rows = []
    for gan, l1, label in PAIRS:
        if not (R / f'per_image_{gan}.csv').exists() or not (R / f'per_image_{l1}.csv').exists():
            continue
        a, b = load(gan), load(l1)
        for m, sign in METRICS.items():
            x, y = a[m].to_numpy(), b[m].to_numpy()
            if sign == 0:
                x, y, sign = np.abs(np.log(x)), np.abs(np.log(y)), -1
            better, worse = (sign * (x - y) > 0).mean(), (sign * (x - y) < 0).mean()
            p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            rows.append({'pair': label, 'gan': gan, 'l1': l1, 'metric': m,
                         'gan_mean': a[m].mean(), 'l1_mean': b[m].mean(),
                         'gan_median': a[m].median(), 'l1_median': b[m].median(),
                         'gan_better_frac': better, 'gan_worse_frac': worse, 'p_value': p})
    df = pd.DataFrame(rows)
    df.to_csv(R / 'gan_vs_l1.csv', index=False)
    with open(R / 'gan_vs_l1.md', 'w') as f:
        f.write('| Kodowanie / wariant | Metryka | GAN | L1 | GAN lepszy / gorszy (% obrazów) | p (Wilcoxon) |\n'
                '|---|---|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r['pair']} | {r['metric']} | {r['gan_mean']:.4f} | {r['l1_mean']:.4f} | "
                    f"{100 * r['gan_better_frac']:.0f} / {100 * r['gan_worse_frac']:.0f} | {r['p_value']:.1e} |\n")
    print(open(R / 'gan_vs_l1.md').read())


if __name__ == '__main__':
    main()
