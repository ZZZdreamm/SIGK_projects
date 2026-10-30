#!/usr/bin/env python
"""Analysis figures for the documentation (written to results/figures/):

* perception_distortion.png - PSNR vs LPIPS of every method (SR)
* val_curves.png            - validation PSNR per epoch for every run
* gain_hist.png             - distribution of the per-image PSNR gain of ResUNet over the baseline
* gain_by_difficulty.png    - mean gain per quartile of baseline PSNR (difficulty)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sigk.utils import load_json  # noqa: E402

TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY = "#8a8985"
RUN_COLORS = {"resunet": "#2a78d6", "resunet-lpips": "#eb6834", "plain": "#1baf7a", "resunet-noca": "#eda100"}
RUN_LABELS = {"resunet": "ResUNet (L1)", "resunet-lpips": "ResUNet (L1 + LPIPS)", "plain": "Plain ResNet",
              "resunet-noca": "ResUNet bez CA"}
LABEL_OFFSETS = {"resunet": (8, -9, "left"), "resunet-noca": (8, 7, "left"), "plain": (-8, 6, "right")}
BASELINES = {"sr": "bicubic", "denoise": "bilateral", "deblur": "richardson_lucy", "inpaint": "telea"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
})


def run_of(method: str, task_name: str) -> str:
    return method[: -len(task_name) - 1]


def perception_distortion(results, out):
    sr = [r for r in results if r["task"] == "sr"]
    if not sr:
        return
    fig, axes = plt.subplots(1, len(sr), figsize=(5.2 * len(sr), 3.9), squeeze=False)
    for ax, r in zip(axes[0], sr):
        for method, m in r["methods"].items():
            run = run_of(method, r["task_name"])
            is_base = run == BASELINES["sr"]
            color = GRAY if is_base else RUN_COLORS.get(run, GRAY)
            ax.scatter(m["psnr"], m["lpips"], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            dx, dy, ha = LABEL_OFFSETS.get(run, (7, 4, "left"))
            ax.annotate("bicubic (baseline)" if is_base else RUN_LABELS.get(run, run), (m["psnr"], m["lpips"]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=TEXT, ha=ha, va="center")
        ax.set_title(f"{r['task_name']}  ({r['description'].split(',')[0]})", loc="left")
        ax.set_xlabel("PSNR [dB]  (więcej = lepiej) →")
        ax.set_ylabel("LPIPS  (mniej = lepiej)")
        ax.margins(x=0.35, y=0.15)
    fig.suptitle("Kompromis zniekształcenie–percepcja: lepsze metody leżą w prawym dolnym rogu", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "perception_distortion.png", dpi=140)
    plt.close(fig)


def val_curves(runs_dir: Path, out):
    groups: dict[str, list] = {}
    for h in sorted(runs_dir.glob("*/history.csv")):
        name = h.parent.name
        run, task_name = name.split("_", 1)
        groups.setdefault(task_name, []).append((run, pd.read_csv(h)))
    if not groups:
        return
    order = sorted(groups, key=lambda t: (not t.startswith("sr"), t))
    fig, axes = plt.subplots(1, len(order), figsize=(4.0 * len(order), 3.4), squeeze=False)
    for ax, task_name in zip(axes[0], order):
        for run, df in sorted(groups[task_name], key=lambda x: list(RUN_COLORS).index(x[0]) if x[0] in RUN_COLORS else 99):
            ax.plot(df.epoch, df.val_psnr, color=RUN_COLORS.get(run, GRAY), label=RUN_LABELS.get(run, run))
        ax.set_title(task_name, loc="left")
        ax.set_xlabel("epoka")
        ax.set_ylabel("PSNR walidacji [dB]")
    handles, labels = {}, {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles[l] = h
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "val_curves.png", dpi=140)
    plt.close(fig)


def gains(results, res: Path):
    data = []
    for r in results:
        csv = res / "per_image" / f"{r['task_name']}.csv"
        model = f"resunet_{r['task_name']}"
        base = f"{BASELINES[r['task']]}_{r['task_name']}"
        if not csv.exists():
            continue
        d = pd.read_csv(csv).pivot(index="idx", columns="method", values="psnr")
        if model in d and base in d:
            data.append((r["task_name"], BASELINES[r["task"]], d[base], d[model] - d[base]))
    return sorted(data, key=lambda x: (not x[0].startswith("sr"), x[0]))


def gain_hist(data, out):
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(4.0 * len(data), 3.2), squeeze=False)
    for ax, (task_name, base, _, g) in zip(axes[0], data):
        ax.hist(g.clip(upper=g.quantile(0.995)), bins=60, color=RUN_COLORS["resunet"], edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(0, color=TEXT, lw=1)
        ax.set_title(task_name, loc="left")
        ax.set_xlabel(f"zysk PSNR względem {base} [dB]")
        ax.set_ylabel("liczba obrazów")
        ax.text(0.97, 0.95, f"lepiej na {100 * (g > 0).mean():.1f}% obrazów\nmediana +{g.median():.2f} dB",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=TEXT)
    fig.suptitle("Rozkład zysku ResUNet nad metodą bazową (każdy obraz testowy osobno)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "gain_hist.png", dpi=140)
    plt.close(fig)


def gain_by_difficulty(data, out):
    if not data:
        return
    labels = ["Q1\n(najtrudniejsze)", "Q2", "Q3", "Q4\n(najłatwiejsze)"]
    fig, axes = plt.subplots(1, len(data), figsize=(4.0 * len(data), 3.2), squeeze=False)
    for ax, (task_name, base, b, g) in zip(axes[0], data):
        q = pd.qcut(b, 4, labels=labels)
        m = g.groupby(q, observed=True).mean()
        bars = ax.bar(range(4), m.values, width=0.62, color=RUN_COLORS["resunet"])
        ax.bar_label(bars, labels=[f"+{v:.2f}" for v in m.values], fontsize=8.5, padding=2, color=TEXT)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(task_name, loc="left")
        ax.set_ylabel("średni zysk PSNR [dB]")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    fig.suptitle("Zysk w zależności od trudności obrazu (kwartyle PSNR metody bazowej)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "gain_by_difficulty.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    results = [load_json(p) for p in sorted(res.glob("*.json"))]
    perception_distortion(results, out)
    val_curves(Path(args.runs), out)
    data = gains(results, res)
    gain_hist(data, out)
    gain_by_difficulty(data, out)
    print("written:", *[p.name for p in sorted(out.glob("*.png")) if not p.name.startswith(("curves_", "sr_", "denoise_", "deblur_", "inpaint_"))])


if __name__ == "__main__":
    main()
