#!/usr/bin/env python
"""Aggregate results/*.json into per-task tables (Markdown, LaTeX, CSV) and
bar charts; collect training curves from runs/.

    python scripts/report.py            # writes results/report.md, results/tables.tex, results/summary.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sigk.tasks import TASK_NAMES  # noqa: E402
from sigk.utils import load_json  # noqa: E402
from sigk.visualize import metric_bars, training_curves  # noqa: E402

TASK_TITLES = {"sr": "Zwiększanie rozdzielczości", "denoise": "Odszumianie", "deblur": "Deblurowanie", "inpaint": "Inpainting"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    res = Path(args.results)

    rows = []
    for p in sorted(res.glob("*.json")):
        r = load_json(p)
        for method, m in r["methods"].items():
            rows.append({"task": r["task"], "task_name": r["task_name"], "description": r["description"],
                         "method": method, "n": m["n"], "mse": m["mse"], "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"],
                         "psnr_std": m.get("psnr_std"), "ssim_std": m.get("ssim_std"), "lpips_std": m.get("lpips_std")})
    if not rows:
        sys.exit(f"no results found in {res}")
    df = pd.DataFrame(rows)
    df.to_csv(res / "summary.csv", index=False)

    md, tex = ["# Wyniki\n"], []
    for task in TASK_NAMES:
        sub = df[df.task == task]
        if sub.empty:
            continue
        title = TASK_TITLES[task]
        md.append(f"\n## {title}\n")
        for task_name, g in sub.groupby("task_name", sort=False):
            n, desc = int(g["n"].iloc[0]), g["description"].iloc[0]
            caption = f"{title}: {desc}. Średnie na zbiorze testowym DIV2K (n = {n} fragmentów 256x256)."
            md.append(f"| Metoda | MSE | PSNR [dB] | SSIM | LPIPS |\n|---|---:|---:|---:|---:|")
            tex.append("\\begin{table}[h]\n\\centering\n\\begin{tabular}{lrrrr}\n\\hline\n"
                       "Metoda & MSE & PSNR [dB] $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\\n\\hline")
            best = {"psnr": g.psnr.max(), "ssim": g.ssim.max(), "lpips": g.lpips.min()}
            for _, r in g.iterrows():
                cell = {k: (f"**{r[k]:.{d}f}**" if r[k] == best[k] else f"{r[k]:.{d}f}") for k, d in (("psnr", 2), ("ssim", 4), ("lpips", 4))}
                md.append(f"| {r.method} | {r.mse:.5f} | {cell['psnr']} | {cell['ssim']} | {cell['lpips']} |")
                tcell = {k: (f"\\textbf{{{r[k]:.{d}f}}}" if r[k] == best[k] else f"{r[k]:.{d}f}") for k, d in (("psnr", 2), ("ssim", 4), ("lpips", 4))}
                name = r.method.replace("_", "\\_")
                tex.append(f"{name} & {r.mse:.5f} & {tcell['psnr']} & {tcell['ssim']} & {tcell['lpips']} \\\\")
            md.append(f"\n*Tabela: {caption}*\n")
            tex.append("\\hline\n\\end{tabular}\n\\caption{" + caption.replace("_", "\\_").replace("->", "$\\rightarrow$") + "}\n\\end{table}\n")
            metric_bars(g[["method", "psnr", "ssim", "lpips"]].reset_index(drop=True), res / "figures" / f"{task_name}_bars.png",
                        title=f"{task_name} ({n} test patches)")
            md.append(f"![](figures/{task_name}_bars.png)\n")
            md.append(f"![](figures/{task_name}_examples.png)\n")

    # training curves
    runs = Path(args.runs)
    curves = []
    if runs.is_dir():
        for run in sorted(runs.iterdir()):
            if (run / "history.csv").exists():
                dst = res / "figures" / f"curves_{run.name}.png"
                training_curves(run / "history.csv", dst, title=run.name)
                curves.append(dst)
    if curves:
        md.append("\n## Krzywe uczenia\n")
        md.extend(f"![](figures/{c.name})\n" for c in curves)

    (res / "report.md").write_text("\n".join(md))
    (res / "tables.tex").write_text("\n".join(tex))
    print("\n".join(md))
    print(f"\nwritten: {res/'report.md'}, {res/'tables.tex'}, {res/'summary.csv'}")


if __name__ == "__main__":
    main()
