"""Figures: side-by-side comparisons, training curves, metric bar charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import psnr, ssim


def comparison_grid(
    rows: list[dict[str, np.ndarray]],
    columns: list[str],
    out_path: str | Path,
    target_key: str = "ground_truth",
    title: str | None = None,
) -> None:
    """``rows``: list of dicts name -> HxWx3 image. One figure row per element,
    one column per entry of ``columns``; PSNR/SSIM vs. ``target_key`` are
    printed under every column except the ground truth."""
    n_r, n_c = len(rows), len(columns)
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.0 * n_c, 3.2 * n_r), squeeze=False)
    for r, row in enumerate(rows):
        for c, key in enumerate(columns):
            ax = axes[r, c]
            ax.axis("off")
            img = row.get(key)
            if img is None:
                continue
            ax.imshow(np.clip(img, 0, 1))
            label = key
            if key != target_key and target_key in row and img.shape == row[target_key].shape:
                label += f"\nPSNR {psnr(img, row[target_key]):.2f}  SSIM {ssim(img, row[target_key]):.3f}"
            ax.set_title(label, fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def training_curves(history_csv: str | Path, out_path: str | Path, title: str | None = None) -> None:
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(df["epoch"], df["train_loss"], label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["val_psnr"], color="C1"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val PSNR [dB]")
    axes[2].plot(df["epoch"], df["val_ssim"], color="C2", label="SSIM")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val SSIM")
    if "val_lpips" in df and df["val_lpips"].notna().any():
        ax2 = axes[2].twinx(); ax2.plot(df["epoch"], df["val_lpips"], color="C3", label="LPIPS"); ax2.set_ylabel("val LPIPS")
    for ax in axes:
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def metric_bars(table: pd.DataFrame, out_path: str | Path, title: str | None = None) -> None:
    """``table`` columns: method, psnr, ssim, lpips (one row per method).
    Baselines / degraded input are gray, trained models blue."""
    metrics = [("psnr", "PSNR [dB] (więcej = lepiej)", 2), ("ssim", "SSIM (więcej = lepiej)", 3), ("lpips", "LPIPS (mniej = lepiej)", 3)]
    gray, blue, text, grid, surface = "#8a8985", "#2a78d6", "#0b0b0b", "#e4e3df", "#fcfcfb"
    baseline_prefixes = ("degraded_input", "bicubic", "bilateral", "richardson_lucy", "telea")
    colors = [gray if n.startswith(baseline_prefixes) else blue for n in table["method"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(table) + 1.4), facecolor=surface)
    for ax, (m, lab, d) in zip(axes, metrics):
        ax.set_facecolor(surface)
        ax.set_axisbelow(True)
        bars = ax.barh(table["method"], table[m], color=colors, height=0.62)
        ax.bar_label(bars, labels=[f"{v:.{d}f}" for v in table[m]], padding=3, fontsize=8, color=text)
        ax.set_xlim(0, table[m].max() * 1.18)
        ax.set_xlabel(lab, color="#52514e")
        ax.grid(axis="x", color=grid)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors="#52514e")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
