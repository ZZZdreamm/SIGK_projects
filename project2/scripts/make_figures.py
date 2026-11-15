"""Figures for the documentation (docs/figures).

Requires ``evaluate.py`` to be run first (predictions in results/preds)
and optionally ``robustness.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.data import paths, read_rgb  # noqa: E402
from lle.raw_pipeline import add_physical_noise, apply_gamma_correction, convert_to_8bit, reduce_exposure  # noqa: E402

# palette (fixed order): networks - blue, baselines - grey, input - light grey
NET_C, BASE_C, MUTED = "#2a78d6", "#8a8984", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 2,
})

LABELS = {"input": "input", "histeq": "hist. equalisation (Y)", "histeq_rgb": "hist. equalisation (RGB)",
          "clahe": "CLAHE", "exposure": "exposure ×8", "exposure_nlm": "exposure ×8 + NLM",
          "isp_nlm": "exposure ×8 + NLM + ISP"}


def lab(m):
    return LABELS.get(m, m)


def is_net(m):
    return m not in LABELS


def load(results, method, data, name):
    if method == "input":
        return read_rgb(paths(data, name)[0])
    if method == "gt":
        return read_rgb(paths(data, name)[1])
    return read_rgb(Path(results) / "preds" / method / f"{name}.png")


def fig_pipeline(data, name, out):
    """Stages of input generation (physical degradation model)."""
    _, pg, pl = paths(data, name)
    lin = read_rgb(pl, unchanged=True).astype(np.float32) / 65535
    s = (slice(None, None, 3), slice(None, None, 3))
    lin = lin[s]
    dark = reduce_exposure(lin, -3)
    noisy = add_physical_noise(dark, rng=np.random.default_rng(0))
    inp = convert_to_8bit(apply_gamma_correction(noisy))
    gt = read_rgb(pg)[s]
    panels = [
        ("1. linear image (no gamma)", convert_to_8bit(lin)),
        ("2. −3 EV (×1/8), linear", convert_to_8bit(dark)),
        ("3. + Poisson and Gaussian noise", convert_to_8bit(np.clip(noisy * 8, 0, 1))),
        ("4. gamma 2.2 → network input", inp),
        ("ground truth (rawpy sRGB)", gt),
    ]
    fig, ax = plt.subplots(1, 5, figsize=(16, 2.6))
    for a, (t, im) in zip(ax, panels):
        a.imshow(im)
        a.set_title(t, fontsize=9)
        a.axis("off")
    ax[2].text(0.02, 0.02, "preview ×8", transform=ax[2].transAxes, color="w", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pipeline.png", bbox_inches="tight")
    plt.close(fig)

    # luminance histograms of the input and the GT
    fig, a = plt.subplots(figsize=(6, 2.6))
    for im, c, t in ((inp, BASE_C, "input (−3 EV + noise)"), (gt, NET_C, "ground truth")):
        y = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).ravel()
        a.hist(y, bins=128, range=(0, 255), histtype="step", color=c, linewidth=2, density=True, label=t)
    a.set_xlabel("luminance (8-bit)")
    a.set_ylabel("density")
    a.legend(frameon=False)
    a.set_title("Brightness histogram")
    fig.tight_layout()
    fig.savefig(out / "histogram.png", bbox_inches="tight")
    plt.close(fig)


def fig_grid(results, data, names, methods, out, fname="comparison.png", step=4):
    cols = ["input"] + methods + ["gt"]
    fig, ax = plt.subplots(len(names), len(cols), figsize=(2.6 * len(cols), 1.8 * len(names)))
    ax = np.atleast_2d(ax)
    for r, n in enumerate(names):
        for c, m in enumerate(cols):
            im = load(results, m, data, n)[::step, ::step]
            ax[r, c].imshow(im)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, c].grid(False)
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
            if r == 0:
                ax[r, c].set_title("ground truth" if m == "gt" else lab(m), fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)


def pick_crop(gt, size, mode="detail"):
    """Picks a crop: 'detail' - most structure, 'clipped' - most clipped pixels."""
    g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if mode == "detail":
        # gradient magnitude of a blurred image: real edges/text, not fine grain;
        # clipped pixels are excluded (they are analysed in a separate figure)
        gb = cv2.GaussianBlur(g, (0, 0), 2.0)
        score = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
        score[gt.max(-1) >= 250] = 0
    else:
        score = (gt.max(-1) >= 250).astype(np.float32)
    score = cv2.boxFilter(score, -1, (size, size))
    h, w = g.shape
    score[: size // 2], score[h - size // 2:], score[:, : size // 2], score[:, w - size // 2:] = 0, 0, 0, 0
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return y - size // 2, x - size // 2


def fig_zoom(results, data, name, methods, out, fname, mode="detail", size=256):
    gt = load(results, "gt", data, name)
    y, x = pick_crop(gt, size, mode)
    cols = ["input"] + methods + ["gt"]
    fig, ax = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.3))
    g = gt.astype(np.float32) / 255
    for c, m in enumerate(cols):
        im = load(results, m, data, name)
        crop = im[y:y + size, x:x + size]
        ax[0, c].imshow(crop)
        ax[0, c].set_title("ground truth" if m == "gt" else lab(m), fontsize=9)
        err = np.abs(im.astype(np.float32) / 255 - g).mean(-1)[y:y + size, x:x + size]
        if m == "gt":
            ax[1, c].imshow(gt.max(-1)[y:y + size, x:x + size] >= 250, cmap="gray")
            ax[1, c].set_title("GT clipping mask", fontsize=8)
        else:
            ax[1, c].imshow(err, cmap="magma", vmin=0, vmax=0.3)
            ax[1, c].set_title(f"|error| mean {err.mean():.3f}", fontsize=8)
        for a in ax[:, c]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout(pad=0.3, h_pad=1.6)
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)


def fig_curves(runs, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for i, r in enumerate(runs):
        f = Path(r) / "log.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        c = SERIES[i % len(SERIES)]
        name = Path(r).name
        ax[0].plot(d.epoch, d.loss, color=c, label=name)
        # best value in the legend instead of end-of-line labels (those overlap for close curves)
        ax[1].plot(d.epoch, d.val_psnr, color=c, label=f"{name} (best {d.val_psnr.max():.2f})")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].set_yscale("log")
    ax[1].set_title("Validation PSNR [dB]"); ax[1].set_xlabel("epoch")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_metrics(results, out):
    df = pd.read_csv(Path(results) / "per_image_test.csv")
    order = list(df.groupby("method", sort=False).psnr.mean().sort_values().index)
    specs = [("psnr", "PSNR [dB] ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"), ("deltaE", "ΔE (CIE76) ↓")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 0.42 * len(order) + 1.2), sharey=True)
    for a, (k, t) in zip(ax, specs):
        data = [df[df.method == m][k].values for m in order]
        bp = a.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                       medianprops={"color": INK, "linewidth": 1.5}, flierprops={"markersize": 3})
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(NET_C if is_net(m) else (MUTED if m == "input" else BASE_C))
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        a.set_title(t)
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(range(1, len(order) + 1), [lab(m) for m in order])
    fig.tight_layout()
    fig.savefig(out / "metrics_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def fig_robustness(results, out):
    f = Path(results) / "robustness.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    methods = list(dict.fromkeys(df.method))
    for i, m in enumerate(methods):
        c = SERIES[i % len(SERIES)]
        d = df[(df.kind == "ev") & (df.method == m)].groupby("ev").psnr.mean()
        ax[0].plot(d.index, d.values, "o-", color=c, label=lab(m), markersize=5)
        dn = df[((df.kind == "noise") | (df.ev == -3.0) & (df.kind == "ev")) & (df.method == m)]
        dn = dn.groupby("noise_scale").psnr.mean()
        ax[1].plot(dn.index, dn.values, "o-", color=c, label=lab(m), markersize=5)
    ax[0].axvline(-3, color=MUTED, linestyle="--", linewidth=1)
    ax[0].set_xlabel("exposure reduction [EV] (training: −3)"); ax[0].set_ylabel("PSNR [dB]")
    ax[0].set_title("Different exposure (nominal noise)")
    ax[1].axvline(1, color=MUTED, linestyle="--", linewidth=1)
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("noise alpha and sigma multiplier (training: 1)")
    ax[1].set_title("Different noise level (−3 EV)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", bbox_inches="tight")
    plt.close(fig)


def fig_gain(results, main_net, out):
    """Auto-exposure (k) estimated from the network output vs k from the clean image."""
    df = pd.read_csv(Path(results) / "per_image_test.csv")
    d = df[df.method == main_net]
    if "gain_est" not in d or d.gain_est.isna().all():
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    lo, hi = 0.8, max(d.gain_true.max(), d.gain_est.max()) * 1.1
    ax[0].plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    ax[0].scatter(d.gain_true, d.gain_est, s=36, color=NET_C, edgecolor="white", linewidth=1.5, zorder=3)
    ax[0].set_xlabel("k from the clean linear image"); ax[0].set_ylabel("k from the network output")
    ax[0].set_title("Auto-exposure (99th percentile)")
    ax[1].scatter(d.psnr, d.psnr_oracle_gain, s=36, color=NET_C, edgecolor="white", linewidth=1.5, zorder=3)
    l2, h2 = d.psnr.min() - 1, d.psnr_oracle_gain.max() + 1
    ax[1].plot([l2, h2], [l2, h2], color=MUTED, linewidth=1, linestyle="--")
    ax[1].set_xlabel("PSNR (estimated k) [dB]"); ax[1].set_ylabel("PSNR (true k) [dB]")
    ax[1].set_title("Effect of the k error on PSNR")
    fig.tight_layout()
    fig.savefig(out / "gain_estimation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--main", required=True, help="main network")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(a.results) / "per_image_test.csv")
    names = list(dict.fromkeys(df.image))
    net = df[df.method == a.main].set_index("image")
    # examples: best, median and worst network result + the image with the most clipping
    by = net.psnr.sort_values()
    picks = list(dict.fromkeys([by.index[-1], by.index[len(by) // 2], by.index[0]]))
    clip_img = net.clipped_frac.idxmax()

    fig_pipeline(a.data, names[0], out)
    fig_grid(a.results, a.data, picks, ["histeq", "isp_nlm", a.main], out)
    fig_grid(a.results, a.data, picks[:2], ["histeq", "histeq_rgb", "clahe", "exposure"], out,
             fname="baselines.png")
    fig_zoom(a.results, a.data, picks[min(1, len(picks) - 1)], ["histeq", "isp_nlm", a.main], out, "zoom_detail.png")
    fig_zoom(a.results, a.data, clip_img, ["histeq", "isp_nlm", a.main], out, "zoom_clipped.png", mode="clipped")
    nets = [m for m in dict.fromkeys(df.method) if is_net(m) and m != a.main]
    if nets:
        fig_zoom(a.results, a.data, picks[min(1, len(picks) - 1)], [a.main] + nets, out, "zoom_networks.png")
    fig_curves(a.runs, out)
    fig_metrics(a.results, out)
    fig_robustness(a.results, out)
    fig_gain(a.results, a.main, out)
    print("figures ->", out, "| picks:", picks, "clipped:", clip_img)


if __name__ == "__main__":
    main()
