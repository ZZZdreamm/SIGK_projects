"""Figures for docs/DOKUMENTACJA.md (docs/figures/*.png)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nr import scene
from nr.data import get_split, load_dataset, vector_to_params
from nr.encoding import encode
from nr.metrics import canny, flip_error, hausdorff
from scripts.evaluate import CROP, crop_boxes

FIG = Path('docs/figures')
RES = Path('results')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
RUNS = ['geometric_gan', 'camera_gan', 'raw_gan', 'geometric_l1']
LABEL = {'geometric_gan': 'cGAN geometryczne', 'camera_gan': 'cGAN względne',
         'raw_gan': 'cGAN surowe', 'geometric_l1': 'tylko L1 (geometryczne)',
         'black': 'czarny obraz', 'analytic': 'ray-tracing kuli'}
COLOR = {r: C[i] for i, r in enumerate(RUNS)}

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 130, 'savefig.bbox': 'tight', 'font.size': 10,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'lines.linewidth': 2,
    'legend.frameon': False,
})


def imshow(ax, img, title=None, **kw):
    ax.imshow(img, interpolation='nearest', **kw)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight='normal')


def zoom(img, box):
    r, c = box
    return img[r:r + CROP, c:c + CROP]


def fig_dataset(files, params, images):
    fig, axes = plt.subplots(3, 8, figsize=(13, 5.2))
    for ax, i in zip(axes.flat, range(24)):
        p = params[i]
        imshow(ax, images[i], f'kd=({int(p[6])},{int(p[7])},{int(p[8])}) n={p[9]:.0f}')
    fig.suptitle('Przykładowe obrazy zbioru (128×128, renderer OpenGL) - kula w losowym miejscu kadru', y=1.0)
    fig.savefig(FIG / 'dataset_samples.png')
    plt.close(fig)


def fig_sampling(params, images):
    stats = json.load(open('data/dataset/sampling_stats.json'))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    labels = {'ok': 'zaakceptowane', 'out_of_view': 'środek poza kadrem', 'behind_camera': 'za kamerą',
              'camera_inside': 'kamera w/przy kuli', 'light_inside': 'światło w kuli'}
    keys = sorted(stats, key=lambda k: -stats[k])
    ax = axes[0]
    ax.barh([labels[k] for k in keys][::-1], [stats[k] for k in keys][::-1], color=C[0], height=0.6)
    for y, k in enumerate(keys[::-1]):
        ax.text(stats[k], y, f' {stats[k]}', va='center', color=INK, fontsize=9)
    ax.set_title('Losowanie z odrzucaniem (15 103 prób)')
    ax.set_xlabel('liczba wylosowanych scen'), ax.grid(axis='y', visible=False)
    ax.set_xlim(0, max(stats.values()) * 1.2)

    ndc = scene.project_ndc(scene.to_camera(params[:, :3]))
    ax = axes[1]
    ax.scatter(ndc[:, 0], ndc[:, 1], s=6, color=C[0], alpha=0.5, linewidths=0)
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_aspect('equal')
    ax.set_title('Położenie środka kuli na ekranie (NDC)')
    ax.set_xlabel('x'), ax.set_ylabel('y')

    fg = (images.max(-1) > 0).sum((1, 2))
    ax = axes[2]
    ax.hist(fg, bins=np.logspace(1.4, 3.3, 35), color=C[0], edgecolor='white', linewidth=1)
    ax.set_xscale('log')
    ax.axvline(np.median(fg), color=INK, linewidth=1, linestyle='--')
    ax.text(np.median(fg), ax.get_ylim()[1] * 0.92, f' mediana {np.median(fg):.0f} px', color=INK, fontsize=9)
    ax.set_title(f'Pole kuli na obrazie (średnio {fg.mean() / 128 ** 2 * 100:.2f} % kadru)')
    ax.set_xlabel('liczba pikseli kuli'), ax.set_ylabel('liczba obrazów')
    fig.tight_layout()
    fig.savefig(FIG / 'sampling.png')
    plt.close(fig)


def fig_degenerate():
    """Scenes the raw U[-20,20] ranges allow - rendered with the analytic ray tracer."""
    cam = scene.CAMERA_POSITION
    cases = [
        ('poprawna scena', [0, 0, 0], [10, 10, 10]),
        ('światło wewnątrz kuli', [0, 0, 0], [0.2, 0.3, 0.4]),
        ('kamera prawie w kuli', list(cam - [0.7, 0.9, 1.3]), [10, 10, 10]),
        ('kula za kamerą / poza kadrem', [18, -15, 19], [0, 0, 0]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (t, o, l) in zip(axes, cases):
        p = scene.SceneParams(o, [200, 80, 40], 10.0, l)
        imshow(ax, scene.phong_reference(p), t)
    fig.suptitle('Sytuacje dopuszczane przez surowe przedziały <-20, 20>', y=1.03)
    fig.savefig(FIG / 'degenerate.png')
    plt.close(fig)


def fig_encoding(params):
    p = torch.from_numpy(params[3:4])
    e = encode(p, 'geometric')[0].numpy()
    q = np.linalg.norm(e[22:25], axis=0)
    panels = [
        (e[17], 'x piksela (CoordConv)', 'RdBu_r'),
        (e[18], 'y piksela (CoordConv)', 'RdBu_r'),
        (np.clip(e[19:22].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1), 'kierunek promienia d', None),
        (np.clip(e[22:25].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1), 'przesunięcie q (RGB)', None),
        (np.minimum(q, 3), '|q| (|q|<1 ⇔ trafienie)', 'viridis'),
        (q < 1, 'maska |q|<1', 'gray'),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.9))
    for ax, (img, t, cmap) in zip(axes, panels):
        imshow(ax, img, t, cmap=cmap)
    fig.suptitle('Mapy warunkujące kodowania „geometric” dla jednej sceny '
                 '(kanały stałe - parametry względne - pominięte)', y=1.04)
    fig.savefig(FIG / 'encoding.png')
    plt.close(fig)


def fig_training():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for r in RUNS:
        f = Path('runs') / r / 'log.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        v = df.dropna(subset=['val_fg_l1'])
        axes[0].plot(v.epoch, v.val_fg_l1, color=COLOR[r], label=LABEL[r])
        axes[1].plot(v.epoch, v.val_lpips, color=COLOR[r], label=LABEL[r])
        if df.loss_d.abs().sum() > 0:
            axes[2].plot(df.epoch, df.adv, color=COLOR[r], label=f'{LABEL[r]} - G adv')
            axes[2].plot(df.epoch, df.loss_d, color=COLOR[r], linestyle=':', label=f'{LABEL[r]} - D')
    axes[0].set_yscale('log'), axes[0].set_title('Walidacja: L1 na obszarze kuli')
    axes[1].set_yscale('log'), axes[1].set_title('Walidacja: LPIPS (cały obraz)')
    axes[2].set_title('Straty adwersarialne (LSGAN)')
    for ax in axes:
        ax.set_xlabel('epoka')
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / 'training_curves.png')
    plt.close(fig)


def load_preds():
    return {m: np.load(RES / 'preds' / f'{m}.npy') for m in RUNS if (RES / 'preds' / f'{m}.npy').exists()}


def fig_comparison(params, images, preds, main, idx, name, title):
    boxes = crop_boxes(params)
    cols = ['referencja'] + [LABEL[m] for m in preds] + [f'FLIP: {LABEL[main]}']
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.0 * len(cols), 2.05 * len(idx)))
    for row, i in enumerate(idx):
        ref = images[i]
        imshow(axes[row, 0], zoom(ref, boxes[i]), cols[0] if row == 0 else None)
        for c, (m, pr) in enumerate(preds.items(), 1):
            imshow(axes[row, c], zoom(pr[i], boxes[i]), cols[c] if row == 0 else None)
            axes[row, c].text(1, 62, f'LPIPS {pd_metrics[m].LPIPS_crop[i]:.3f}', color='white', fontsize=7)
        fv, fmap = flip_error(ref / 255.0, preds[main][i] / 255.0, return_map=True)
        imshow(axes[row, -1], zoom(fmap, boxes[i]), cols[-1] if row == 0 else None)
        axes[row, -1].text(1, 62, f'FLIP {fv:.4f}', color='white', fontsize=7)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_full_frame(images, preds, main, idx):
    fig, axes = plt.subplots(2, len(idx), figsize=(2.1 * len(idx), 4.4))
    for c, i in enumerate(idx):
        imshow(axes[0, c], images[i], 'referencja' if c == 0 else None)
        imshow(axes[1, c], preds[main][i], LABEL[main] if c == 0 else None)
    fig.suptitle('Pełne klatki 128×128 ze zbioru testowego', y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'full_frames.png')
    plt.close(fig)


def fig_boxplots(methods):
    metrics = [('FLIP', 'FLIP ↓'), ('LPIPS', 'LPIPS ↓'), ('SSIM', 'SSIM ↑'), ('Hausdorff', 'Hausdorff [px] ↓')]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, suffix in enumerate(('', '_crop')):
        for ax, (m, t) in zip(axes[row], metrics):
            data = [pd_metrics[k][m + suffix].values for k in methods]
            bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=True, widths=0.6,
                            flierprops={'markersize': 2, 'markerfacecolor': MUTED, 'markeredgecolor': MUTED})
            for patch, k in zip(bp['boxes'], methods):
                patch.set_facecolor(COLOR.get(k, '#bdbcb7'))
                patch.set_edgecolor(INK)
            for med in bp['medians']:
                med.set_color(INK)
            ax.set_xticks(range(1, len(methods) + 1), [LABEL[k] for k in methods], rotation=35, ha='right', fontsize=8)
            ax.set_title(t + (' - wycinek 64×64' if suffix else ' - cały obraz'))
            if m == 'Hausdorff':
                ax.set_yscale('symlog', linthresh=1)
                ax.set_ylim(bottom=0)
            if m in ('FLIP', 'LPIPS'):
                ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(FIG / 'metrics_boxplot.png')
    plt.close(fig)


def fig_edges(images, preds, main, idx):
    fig, axes = plt.subplots(len(idx), 4, figsize=(9, 2.3 * len(idx)))
    boxes = crop_boxes(params_test)
    for row, i in enumerate(idx):
        ref, pr = images[i] / 255.0, preds[main][i] / 255.0
        er, ep = canny(ref), canny(pr)
        over = np.zeros(ref.shape)
        over[er] = [0.16, 0.47, 0.84]
        over[ep] = [0.92, 0.41, 0.20]
        over[er & ep] = [1, 1, 1]
        h = hausdorff(ref, pr)
        for c, (img, t) in enumerate([(ref, 'referencja'), (pr, LABEL[main]),
                                      (er, 'Canny: referencja'), (over, f'krawędzie (H = {h:.2f} px)')]):
            imshow(axes[row, c], zoom(img, boxes[i]), t, cmap='gray' if img.ndim == 2 else None)
    fig.suptitle('Odległość Hausdorffa na krawędziach Canny (niebieski: ref., pomarańczowy: sieć, biały: wspólne)',
                 y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'edges_hausdorff.png')
    plt.close(fig)


def fig_metric_study():
    df = pd.read_csv(RES / 'metric_study.csv')
    metrics = [('FLIP', 'FLIP ↓'), ('LPIPS', 'LPIPS ↓'), ('SSIM', '1 − SSIM ↓'), ('Hausdorff', 'Hausdorff [px] ↓')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    y = np.arange(len(df))[::-1]
    for row, suffix in enumerate(('', '_crop')):
        for ax, (m, t) in zip(axes[row], metrics):
            v = df[m + suffix].values
            v = 1 - v if m == 'SSIM' else v
            ax.barh(y, v, color=C[0], height=0.6)
            for yy, vv in zip(y, v):
                ax.text(vv, yy, f' {vv:.3g}', va='center', fontsize=8, color=INK)
            ax.set_xlim(0, v.max() * 1.3)
            ax.set_title(t + (' - wycinek' if suffix else ' - cały obraz'), fontsize=10)
            ax.grid(axis='y', visible=False)
    axes[0, 0].set_yticks(y, df.opis)
    axes[1, 0].set_yticks(y, df.opis)
    fig.suptitle('Reakcja metryk na kontrolowane zniekształcenia referencji (150 obrazów testowych)', y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'metric_study.png')
    plt.close(fig)


def fig_error_vs_size(main):
    df = pd_metrics[main]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, m in zip(axes, ['LPIPS_crop', 'FLIP_crop']):
        ax.scatter(df.fg_pixels, df[m], s=8, color=C[0], alpha=0.6, linewidths=0)
        ax.set_xscale('log')
        ax.set_xlabel('pole kuli [px]'), ax.set_ylabel(m.replace('_crop', ' (wycinek)'))
        r = np.corrcoef(np.log(df.fg_pixels), df[m])[0, 1]
        ax.set_title(f'{m.replace("_crop", "")} a wielkość kuli (r = {r:.2f}, oś log)')
    fig.tight_layout()
    fig.savefig(FIG / 'error_vs_size.png')
    plt.close(fig)


def fig_demo():
    """Panels saved by the NeuralPhongWindow (renderer + network): ref | net | FLIP."""
    files = sorted(Path('results/demo').glob('neural_*.png'))[:4]
    if not files:
        return
    from PIL import Image
    met = pd.read_csv('results/demo/metrics.csv')
    fig, axes = plt.subplots(len(files), 1, figsize=(7.5, 2.6 * len(files)))
    for ax, f, (_, r) in zip(np.atleast_1d(axes), files, met.iterrows()):
        imshow(ax, np.asarray(Image.open(f)))
        ax.set_title(f'renderer OpenGL | neural renderer | mapa FLIP  -  FLIP {r.FLIP:.4f}, LPIPS {r.LPIPS:.4f}, '
                     f'SSIM {r.SSIM:.4f}, H {r.Hausdorff:.1f} px', fontsize=8, fontweight='normal')
    fig.tight_layout()
    fig.savefig(FIG / 'demo_renderer.png')
    plt.close(fig)


def main():
    global pd_metrics, params_test
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default='geometric_gan')
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    files, params, images = load_dataset('data/dataset')
    split = get_split('data/dataset')
    tr = split['train']
    fig_dataset(files, params[tr], images[tr])
    fig_sampling(params, images)
    fig_degenerate()
    fig_encoding(params[tr])
    fig_training()

    test = split['test']
    params_test, images_test = params[test], images[test]
    preds = load_preds()
    methods = [m for m in RUNS if m in preds] + ['black']
    pd_metrics = {m: pd.read_csv(RES / f'per_image_{m}.csv') for m in methods + ['analytic']
                  if (RES / f'per_image_{m}.csv').exists()}
    if preds:
        fg = pd_metrics[args.main].fg_pixels.values
        order = np.argsort(fg)
        idx = [int(order[int(q * (len(order) - 1))]) for q in (0.1, 0.35, 0.6, 0.8, 0.93, 1.0)]
        fig_comparison(params_test, images_test, preds, args.main, idx, 'comparison.png',
                       'Zbiór testowy: wycinek 64×64 wokół kuli (od najmniejszej do największej)')
        worst = pd_metrics[args.main].LPIPS_crop.values.argsort()[::-1][:5].tolist()
        fig_comparison(params_test, images_test, preds, args.main, worst, 'worst_cases.png',
                       f'Najgorsze przypadki ({LABEL[args.main]}, wg LPIPS na wycinku)')
        fig_full_frame(images_test, preds, args.main, idx)
        fig_boxplots(methods)
        fig_edges(images_test, preds, args.main, idx[2:5])
        fig_error_vs_size(args.main)
    fig_demo()
    if (RES / 'metric_study.csv').exists():
        fig_metric_study()
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
