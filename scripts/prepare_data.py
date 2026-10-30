#!/usr/bin/env python
"""Cut DIV2K HR images into 256x256 patches.

Expected input layout (as downloaded from https://data.vision.ee.ethz.ch/cvl/DIV2K/):

    <div2k_dir>/DIV2K_train_HR/0001.png ... 0800.png
    <div2k_dir>/DIV2K_valid_HR/0801.png ... 0900.png

Output:

    <out>/train/  random crops from DIV2K_train_HR images 1..(800-val_images)
    <out>/val/    deterministic tiles from the last ``val_images`` training images
    <out>/test/   non-overlapping grid of tiles from *all* DIV2K_valid_HR images
                  (this is the test set on which the final metrics are reported)

With ``--mode resize`` every image is instead resized as a whole to 256x256
(fewer, but complete, images; aspect ratio is not preserved).

Training degradations (noise, blur, holes, downscaling) are applied on the fly
by ``sigk.data.PatchDataset``; only the clean patches are stored here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sigk.utils import list_images, read_image, write_image  # noqa: E402


def grid_tiles(img: np.ndarray, patch: int, max_tiles: int | None) -> list[np.ndarray]:
    h, w = img.shape[:2]
    ny, nx = h // patch, w // patch
    # center the grid on the image
    oy, ox = (h - ny * patch) // 2, (w - nx * patch) // 2
    coords = [(oy + i * patch, ox + j * patch) for i in range(ny) for j in range(nx)]
    if max_tiles and len(coords) > max_tiles:
        pick = np.linspace(0, len(coords) - 1, max_tiles).round().astype(int)
        coords = [coords[i] for i in pick]
    return [img[y : y + patch, x : x + patch] for y, x in coords]


def random_crops(img: np.ndarray, patch: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        y = int(rng.integers(0, h - patch + 1))
        x = int(rng.integers(0, w - patch + 1))
        out.append(img[y : y + patch, x : x + patch])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--div2k-dir", required=True, help="folder containing DIV2K_train_HR and DIV2K_valid_HR")
    ap.add_argument("--out", default="data/patches")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--train-crops-per-image", type=int, default=16, help="random crops per training image")
    ap.add_argument("--val-images", type=int, default=40, help="last N training images are used for validation")
    ap.add_argument("--val-tiles-per-image", type=int, default=4)
    ap.add_argument("--test-tiles-per-image", type=int, default=0, help="0 = full non-overlapping grid")
    ap.add_argument("--mode", choices=["crop", "resize"], default="crop",
                    help="crop: 256x256 patches at native resolution (default); "
                         "resize: whole image resized to 256x256 with cv2.INTER_AREA (one image per file)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    div2k = Path(args.div2k_dir)
    train_dir = div2k / "DIV2K_train_HR"
    test_dir = div2k / "DIV2K_valid_HR"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            sys.exit(f"Missing folder: {d}")
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    train_files = list_images(train_dir)
    val_files = train_files[len(train_files) - args.val_images :] if args.val_images else []
    train_files = train_files[: len(train_files) - args.val_images]
    test_files = list_images(test_dir)

    counts = {}
    for split, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        n = 0
        for f in tqdm(files, desc=f"{split:5s}"):
            img = read_image(f)
            if img.shape[0] < args.patch or img.shape[1] < args.patch:
                continue
            if args.mode == "resize":
                tiles = [cv2.resize(img, (args.patch, args.patch), interpolation=cv2.INTER_AREA)]
            elif split == "train":
                tiles = random_crops(img, args.patch, args.train_crops_per_image, rng)
            elif split == "val":
                tiles = grid_tiles(img, args.patch, args.val_tiles_per_image)
            else:
                tiles = grid_tiles(img, args.patch, args.test_tiles_per_image or None)
            for k, t in enumerate(tiles):
                write_image(out / split / f"{f.stem}_{k:03d}.png", t)
                n += 1
        counts[split] = n
    print("patches written:", counts, "->", out)


if __name__ == "__main__":
    main()
