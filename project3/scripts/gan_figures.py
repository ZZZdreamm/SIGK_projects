"""Figures for the GAN vs. L1 section (docs/figures/gan_vs_l1_*.png).

Needs results/preds/*.npy, results/per_image_*.csv (evaluate.py) and
results/gan_vs_l1.csv (compare_gan.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nr.data import get_split, load_dataset
from scripts.evaluate import crop_boxes
from scripts.make_figures import C, FIG, INK, MUTED, RES, imshow, plt, zoom

COLS = [('camera_l1', 'względne, tylko L1'), ('camera_gan_v2', 'względne, cGAN v2'),
        ('raw_l1', 'surowe, tylko L1'), ('raw_gan_v2', 'surowe, cGAN v2')]


def fig_crops(params, images, idx, title):
    boxes = crop_boxes(params)
    preds = {m: np.load(RES / 'preds' / f'{m}.npy') for m, _ in COLS}
    per = {m: pd.read_csv(RES / f'per_image_{m}.csv') for m, _ in COLS}
    fig, axes = plt.subplots(len(idx), 1 + len(COLS), figsize=(2.0 * (1 + len(COLS)), 2.05 * len(idx)))
    for row, i in enumerate(idx):
        imshow(axes[row, 0], zoom(images[i], boxes[i]), 'referencja' if row == 0 else None)
        for c, (m, label) in enumerate(COLS, 1):
            imshow(axes[row, c], zoom(preds[m][i], boxes[i]), label if row == 0 else None)
            axes[row, c].text(1, 62, f'LPIPS {per[m].LPIPS_crop[i]:.3f}', color='white', fontsize=7)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'gan_vs_l1_crops.png')
    plt.close(fig)


def fig_change():
    df = pd.read_csv(RES / 'gan_vs_l1.csv')
    metrics = [('FLIP_crop', 'FLIP (wycinek)'), ('LPIPS_crop', 'LPIPS (wycinek)'),
               ('SSIM_crop', '1 - SSIM (wycinek)'), ('sharpness', 'brak ostrości |log ostrość|')]
    enc = [('geometryczne', 'geometryczne'), ('względne (kamera)', 'względne'), ('surowe', 'surowe')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 2.9), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        for k, (ver, color, dy) in enumerate((('GAN v1', C[1], 0.12), ('GAN v2', C[0], -0.12))):
            for y, (key, _) in enumerate(enc):
                r = df[(df.pair == f'{key}, {ver}') & (df.metric == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                g, l = r.gan_mean, r.l1_mean
                if m == 'SSIM_crop':
                    g, l = 1 - g, 1 - l
                if m == 'sharpness':
                    g, l = abs(np.log(g)), abs(np.log(l))
                ch = 100 * (g - l) / l
                ax.barh(y + dy, ch, height=0.22, color=color, label=ver if y == 0 else None)
                star = '' if r.p_value >= 0.01 else ' *'
                ax.text(ch + (1.5 if ch >= 0 else -1.5), y + dy, f'{ch:+.0f} %{star}', va='center',
                        ha='left' if ch >= 0 else 'right', fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('zmiana względem L1 [%]  (< 0: GAN lepszy)')
        lim = max(abs(v) for v in ax.get_xlim()) * 1.35
        ax.set_xlim(-lim, lim)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(enc)), [e[1] for e in enc])
    axes[0].invert_yaxis()
    fig.suptitle('cGAN względem regresji L1 na zbiorze testowym (600 obrazów); * = p < 0,01 (Wilcoxon)', y=1.08)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / 'gan_vs_l1_change.png')
    plt.close(fig)


def fig_curves():
    runs = [('camera_l1', 'względne, tylko L1', C[1], '-'), ('camera_gan', 'względne, cGAN v1', C[1], ':'),
            ('camera_gan_v2', 'względne, cGAN v2', C[0], '-'),
            ('geometric_l1', 'geometryczne, tylko L1', C[2], '-'),
            ('geometric_gan_v2', 'geometryczne, cGAN v2', C[3], '-')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for r, label, color, ls in runs:
        df = pd.read_csv(Path('runs') / r / 'log.csv')
        v = df.dropna(subset=['val_lpips'])
        axes[0].plot(v.epoch, v.val_lpips, color=color, linestyle=ls, label=label)
        if df.loss_d.abs().sum() > 0:
            axes[1].plot(df.epoch, df.loss_d, color=color, linestyle=ls, label=label)
    axes[1].axhline(0.25, color=MUTED, linewidth=1, linestyle='--')
    axes[1].text(40, 0.27, 'poziom zgadywania LSGAN (0,25)', fontsize=8, color=MUTED)
    axes[1].set_ylim(0, 0.4)
    axes[0].set_yscale('log'), axes[0].set_title('Walidacja: LPIPS')
    axes[1].set_title('Strata dyskryminatora')
    for ax in axes:
        ax.set_xlabel('epoka')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'gan_vs_l1_training.png')
    plt.close(fig)


def main():
    _, params, images = load_dataset('data/dataset')
    test = get_split('data/dataset')['test']
    params, images = params[test], images[test]
    a = pd.read_csv(RES / 'per_image_camera_gan_v2.csv').LPIPS_crop.values
    b = pd.read_csv(RES / 'per_image_camera_l1.csv').LPIPS_crop.values
    gain = b - a
    order = np.argsort(gain)
    idx = [int(order[int(q * (len(order) - 1))]) for q in (0.5, 0.7, 0.85, 0.95, 1.0)]
    fig_crops(params, images, idx, 'Ten sam generator z dyskryminatorem i bez; wiersze: mediana, 70., 85., 95. '
              'i 100. percentyl zysku LPIPS cGAN v2 nad L1 (kodowanie względne)')
    fig_change()
    fig_curves()
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
