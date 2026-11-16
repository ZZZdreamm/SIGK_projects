"""Evaluate trained neural renderers (and reference baselines) on the 600 test images.

Metrics on the full 128×128 frame (as required) and additionally on a 64×64
crop centred on the sphere (the sphere covers <1 % of the frame, the full-frame
numbers are dominated by the black background).

Outputs: results/per_image_<method>.csv, results/summary.csv, results/table.md,
         results/preds/<method>.npy (uint8 predictions, for the figures)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nr import scene
from nr.data import load_dataset, get_split, vector_to_params
from nr.encoding import CHANNELS, encode
from nr.metrics import LPIPS, batch_metrics
from nr.models import Generator

CROP = 64
METRICS = ['FLIP', 'LPIPS', 'SSIM', 'Hausdorff']
NAMES = {
    'geometric_gan': 'neural renderer (cGAN, kodowanie geometryczne)',
    'camera_gan': 'cGAN, kodowanie względne (kamera)',
    'raw_gan': 'cGAN, surowe parametry',
    'geometric_l1': 'bez GAN (tylko L1), kodowanie geometryczne',
    'geometric_gan_v2': 'cGAN v2 (maska + FM), kodowanie geometryczne',
    'camera_gan_v2': 'cGAN v2, kodowanie względne (kamera)',
    'camera_l1': 'bez GAN (tylko L1), kodowanie względne (kamera)',
    'raw_gan_v2': 'cGAN v2, surowe parametry',
    'raw_l1': 'bez GAN (tylko L1), surowe parametry',
    'black': 'czarny obraz (baseline)',
    'analytic': 'analityczny ray-tracing idealnej kuli',
}


@torch.no_grad()
def predict(ckpt_path: Path, params: np.ndarray, device='cuda') -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    G = Generator(CHANNELS[ckpt['encoding']], ckpt['base']).to(device).eval()
    G.load_state_dict(ckpt['G'])
    p = torch.from_numpy(params).to(device)
    out = torch.cat([G(encode(p[i:i + 100], ckpt['encoding'])) for i in range(0, len(p), 100)])
    return (out.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


def crop_boxes(params: np.ndarray) -> np.ndarray:
    """Top-left corner of a CROP×CROP window centred on the projected sphere centre."""
    c = scene.to_camera(params[:, 0:3])
    ndc = scene.project_ndc(c)
    col = (ndc[:, 0] + 1) / 2 * scene.IMAGE_SIZE
    row = (1 - ndc[:, 1]) / 2 * scene.IMAGE_SIZE
    tl = np.stack([row, col], 1) - CROP / 2
    return np.clip(np.round(tl), 0, scene.IMAGE_SIZE - CROP).astype(int)


def crop(imgs: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    return np.stack([im[r:r + CROP, c:c + CROP] for im, (r, c) in zip(imgs, boxes)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--data', default='data/dataset')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'preds').mkdir(parents=True, exist_ok=True)

    files, params, images = load_dataset(args.data)
    test = get_split(args.data)['test']
    params, images = params[test], images[test]
    files = [files[i] for i in test]
    refs = images.astype(np.float32) / 255.0
    boxes = crop_boxes(params)
    fg_px = (images.max(-1) > 0).sum((1, 2))
    lp = LPIPS('cuda')

    methods = {}
    for run in args.runs:
        run = Path(run)
        if (run / 'best.pt').exists():
            methods[run.name] = lambda run=run: predict(run / 'best.pt', params)
    methods['black'] = lambda: np.zeros_like(images)
    methods['analytic'] = lambda: np.stack([
        (scene.phong_reference(vector_to_params(p)) * 255).round().astype(np.uint8) for p in params])

    rows = []
    for name, fn in methods.items():
        pred8 = fn()
        np.save(out / 'preds' / f'{name}.npy', pred8)
        pred = pred8.astype(np.float32) / 255.0
        full = batch_metrics(refs, pred, lp)
        cr = batch_metrics(crop(refs, boxes), crop(pred, boxes), lp)
        df = pd.DataFrame({'file': files, 'fg_pixels': fg_px,
                           **{m: full[m] for m in METRICS}, **{f'{m}_crop': cr[m] for m in METRICS}})
        df.to_csv(out / f'per_image_{name}.csv', index=False)
        row = {'method': name, **{m: df[m].mean() for m in METRICS},
               **{f'{m}_crop': df[f'{m}_crop'].mean() for m in METRICS},
               'Hausdorff_median': df['Hausdorff'].median(),
               'Hausdorff_empty_edges': int((df['Hausdorff'] >= np.hypot(128, 128) - 1e-6).sum())}
        rows.append(row)
        print(f"{name:15s} " + ' '.join(f"{m} {row[m]:.4f}" for m in METRICS)
              + ' | crop ' + ' '.join(f"{m} {row[m + '_crop']:.4f}" for m in METRICS), flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary.csv', index=False)
    with open(out / 'table.md', 'w') as f:
        for suffix, title in (('', 'cały obraz 128×128'), ('_crop', f'wycinek {CROP}×{CROP} wokół kuli')):
            f.write(f'\n**Zbiór testowy (600 obrazów), {title}**\n\n')
            f.write('| Metoda | FLIP ↓ | LPIPS ↓ | SSIM ↑ | Hausdorff [px] ↓ |\n|---|---|---|---|---|\n')
            for r in rows:
                f.write(f"| {NAMES.get(r['method'], r['method'])} | {r['FLIP' + suffix]:.4f} | "
                        f"{r['LPIPS' + suffix]:.4f} | {r['SSIM' + suffix]:.4f} | {r['Hausdorff' + suffix]:.2f} |\n")
    print(open(out / 'table.md').read())


if __name__ == '__main__':
    main()
