"""Generates the dataset from NEF files: GT (sRGB), low-light input, linear image.

Split (CSV order, first 100 photos Nikon D7000 / Indoor):
  * 1–72   → train
  * 73–80  → val   (selection of the best checkpoint)
  * 81–100 → test  (the last 20 - required by the assignment)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.download import nef_filename, read_raise_rows  # noqa: E402
from lle.raw_pipeline import process_single_file  # noqa: E402
from lle.utils import save_json  # noqa: E402


def split_of(i: int, n_train: int, n_val: int) -> str:
    if i < n_train:
        return "train"
    if i < n_train + n_val:
        return "val"
    return "test"


def _job(args):
    nef, out, ev, seed = args
    process_single_file(nef, os.path.join(out, "gt"), os.path.join(out, "input"), ev,
                        output_linear_dir=os.path.join(out, "linear"), seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="RAISE_258.csv")
    ap.add_argument("--nef-dir", default="data/nef")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=72)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--ev", type=float, default=-3.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    for d in ("gt", "input", "linear"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    rows = read_raise_rows(a.csv, limit=a.limit)
    split = {"train": [], "val": [], "test": []}
    jobs = []
    for i, row in enumerate(rows):
        name = os.path.splitext(nef_filename(row))[0]
        split[split_of(i, a.n_train, a.n_val)].append(name)
        nef = os.path.join(a.nef_dir, nef_filename(row))
        if not os.path.exists(nef):
            print(f"Missing {nef} - run scripts/download.py first")
            continue
        jobs.append((nef, a.out, a.ev, 1000 + i))  # fixed noise seed for every image

    # rawpy (OpenMP) + fork may deadlock -> spawn
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_job, jobs))

    save_json({"ev_shift": a.ev, "alpha": 0.001, "sigma": 0.005, "gamma": 2.2, **split},
              os.path.join(a.out, "split.json"))
    print({k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
