#!/usr/bin/env python
"""Grid-search the parameters of the classical baselines on the *validation*
split (never on the test set) and store the best ones (by PSNR) in
configs/baselines.json. evaluate.py then uses them automatically, so the
network is compared against a reasonably tuned baseline.

    python scripts/tune_baselines.py                       # all tasks
    python scripts/tune_baselines.py --tasks deblur:3 deblur:5 --limit 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sigk.metrics import psnr, ssim  # noqa: E402
from sigk.tasks import ALL_TASKS, BASELINE_PARAMS_FILE, parse_task  # noqa: E402
from sigk.utils import cpu_count, list_images, read_image  # noqa: E402

GRIDS = {
    "denoise": {"sigma_color": [0.005, 0.01, 0.02, 0.05, 0.1], "sigma_spatial": [1.0, 3.0, 5.0, 8.0]},
    "deblur": {"num_iter": [5, 10, 20, 30, 50, 100, 200]},
    "inpaint": {"radius": [1, 3, 5, 9, 15, 25]},
}


def _score(args):
    task, files, params, seed = args
    spec = parse_task(task)
    ps, ss = [], []
    for i, f in enumerate(files):
        img = read_image(f)
        d = spec.degrade(img, np.random.default_rng([seed, i]))
        out = spec.baseline(d, **params)
        ps.append(psnr(out, img)); ss.append(ssim(out, img))
    return params, float(np.mean(ps)), float(np.mean(ss))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=ALL_TASKS)
    ap.add_argument("--data", default="data/patches/val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=cpu_count())
    args = ap.parse_args()

    files = list_images(args.data)[: args.limit]
    cfg = json.loads(BASELINE_PARAMS_FILE.read_text()) if BASELINE_PARAMS_FILE.exists() else {}
    for t in args.tasks:
        spec = parse_task(t)
        grid = GRIDS.get(spec.task)
        if not grid:
            print(f"{spec.name}: nothing to tune ({spec.baseline_name})")
            continue
        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(_score, [(t, files, c, 777) for c in combos]))
        results.sort(key=lambda r: -r[1])
        print(f"\n{spec.name} ({spec.baseline_name}) on {len(files)} val patches")
        for params, p, s in results:
            print(f"  {json.dumps(params):50s} PSNR {p:7.3f}  SSIM {s:.4f}")
        best = results[0]
        cfg[spec.name] = {"method": spec.baseline_name, "params": best[0], "val_psnr": best[1], "val_ssim": best[2],
                          "n_val": len(files), "grid": grid}
    BASELINE_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PARAMS_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"\nsaved -> {BASELINE_PARAMS_FILE}")


if __name__ == "__main__":
    main()
