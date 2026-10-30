#!/usr/bin/env python
"""Visual comparison figures with *all* methods of a task side by side
(degraded input, baseline, every trained model in runs/*_<task_name>/, ground truth).

The examples are chosen automatically: detailed (high-gradient) test patches,
each from a different DIV2K image. A zoomed crop is shown under every row.

    python scripts/make_figures.py                     # all tasks found in results/
    python scripts/make_figures.py --tasks sr:32 --n 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import load_model  # noqa: E402
from sigk.data import PatchDataset  # noqa: E402
from sigk.metrics import psnr, ssim  # noqa: E402
from sigk.models import compose_output  # noqa: E402
from sigk.tasks import parse_task  # noqa: E402
from sigk.utils import get_device, load_json, read_image, tensor_to_hwc, write_image  # noqa: E402

TASK_FROM_RESULT = {"sr": lambda n: f"sr:{n.split('_')[1].split('x')[0]}",
                    "denoise": lambda n: f"denoise:{n.split('_')[-1]}",
                    "deblur": lambda n: f"deblur:{n.split('_')[-1].split('x')[0]}",
                    "inpaint": lambda n: f"inpaint:{n.split('_')[1].split('x')[0]}"}


def pick_examples(files, n, stride=7):
    """Most detailed patch (Laplacian variance) from n different source images."""
    best = {}
    for i in range(0, len(files), stride):
        img = read_image(files[i])
        score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var()
        src = files[i].stem.split("_")[0]
        if src not in best or score > best[src][0]:
            best[src] = (score, i)
    ranked = sorted(best.values(), reverse=True)
    # skip the very top (often pure noise-like texture) and spread over the ranking
    pool = ranked[: max(n * 4, n)]
    return sorted(i for _, i in pool[:: max(1, len(pool) // n)][:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--data", default="data/patches/test")
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--crop", type=int, default=64, help="size of the zoomed crop")
    ap.add_argument("--seed", type=int, default=2024, help="must match evaluate.py")
    args = ap.parse_args()
    device = get_device()
    res = Path(args.results)

    tasks = args.tasks
    if tasks is None:
        tasks = []
        for p in sorted(res.glob("*.json")):
            r = load_json(p)
            tasks.append(TASK_FROM_RESULT[r["task"]](r["task_name"]))

    for t in tasks:
        spec = parse_task(t)
        ds = PatchDataset(args.data, spec, train=False, deterministic=True, seed=args.seed, with_baseline=True)
        models = [load_model(str(c), device) for c in sorted(Path(args.runs).glob(f"*_{spec.name}/best.pt"))]
        models.sort(key=lambda m: (m[1] != "resunet", m[1]))  # main model first
        idxs = pick_examples(ds.files, args.n)

        rows = []
        for i in idxs:
            x, y, mask, _, base = ds[i]
            row = {}
            row["input (bicubic)" if spec.task == "sr" else "input"] = tensor_to_hwc(x[:3])
            if spec.task != "sr":
                row[spec.baseline_name] = tensor_to_hwc(base)
            xb, mb = x[None].to(device), mask[None].to(device)
            for model, label, _ in models:
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    p = model(xb.contiguous(memory_format=torch.channels_last))
                row[label] = tensor_to_hwc(compose_output(p.float(), xb, mb, spec.task).clamp(0, 1)[0])
            row["ground truth"] = tensor_to_hwc(y)
            rows.append((i, row))

        cols = list(rows[0][1].keys())
        c = args.crop
        fig, axes = plt.subplots(2 * len(rows), len(cols), figsize=(2.6 * len(cols), 2 * 2.75 * len(rows)),
                                 squeeze=False, gridspec_kw={"hspace": 0.25, "wspace": 0.04})
        for r, (i, row) in enumerate(rows):
            gt = row["ground truth"]
            # zoom on the most detailed c x c window
            g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
            lap = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (c, c))
            h = c // 2
            cy, cx = np.unravel_index(np.argmax(lap[h:-h, h:-h]), lap[h:-h, h:-h].shape)
            y0, x0 = cy, cx
            for k, name in enumerate(cols):
                img = row[name]
                ax, axz = axes[2 * r, k], axes[2 * r + 1, k]
                ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                ax.add_patch(plt.Rectangle((x0, y0), c, c, fill=False, ec="red", lw=1))
                axz.imshow(img[y0:y0 + c, x0:x0 + c], interpolation="nearest"); axz.axis("off")
                label = name if r == 0 else ""
                if name != "ground truth":
                    label += ("\n" if label else "") + f"{psnr(img, gt):.2f} dB / {ssim(img, gt):.3f}"
                ax.set_title(label, fontsize=9)
                write_image(res / "images" / spec.name / f"{i:05d}_{name.replace(' ', '_')}.png", img)
            axes[2 * r, 0].set_ylabel(ds.files[i].stem, fontsize=8)
        fig.suptitle(f"{spec.name}: {spec.description}\n(PSNR / SSIM względem oryginału)", fontsize=11,
                     y=axes[0, 0].get_position().y1 + 0.035, va="bottom")
        out = res / "figures" / f"{spec.name}_examples.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"{spec.name}: {len(models)} model(s), examples {idxs} -> {out}")


if __name__ == "__main__":
    main()
