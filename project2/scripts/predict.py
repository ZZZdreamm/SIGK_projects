"""Demo: enhance a single image with a trained network.

Accepts either
  * a RAW file (.NEF etc.) - the low-light input and the ground truth are generated
    exactly as in the dataset (-3 EV + Poisson/Gaussian noise), so metrics can be shown, or
  * an already dark 8-bit image (.png/.jpg) - optionally with --gt for metrics.

Writes <stem>_input.png, <stem>_histeq.png, <stem>_network.png and a side-by-side
<stem>_comparison.png into --out.

  uv run python scripts/predict.py data/nef/redaf7d93t.NEF
  uv run python scripts/predict.py data/dataset/input/rc1d2751dt_input.png \
      --gt data/dataset/gt/rc1d2751dt_gt.png --checkpoint runs/isp_jitter/best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.baselines import histeq_y  # noqa: E402
from lle.data import read_rgb  # noqa: E402
from lle.metrics import LPIPS, psnr, ssim  # noqa: E402
from lle.models import build_model, infer_full  # noqa: E402
from lle.raw_pipeline import get_ground_truth, get_linear_image, get_low_light  # noqa: E402
from lle.utils import get_device  # noqa: E402

RAW_EXTS = {".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2"}


def load_inputs(path: Path, gt_path: str | None, ev: float, seed: int):
    if path.suffix.lower() in RAW_EXTS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            gt = get_ground_truth(raw)
            lin = get_linear_image(raw)
        dark = get_low_light(None, ev, rng=np.random.default_rng(seed), linear_img=lin)
        return dark, gt
    dark = read_rgb(path)
    return dark, (read_rgb(gt_path) if gt_path else None)


def save(path: Path, img: np.ndarray):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="RAW file (.NEF) or a dark 8-bit image")
    ap.add_argument("--gt", default=None, help="ground truth for an 8-bit input (optional)")
    ap.add_argument("--checkpoint", default="runs/isp/best.pt")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--ev", type=float, default=-3.0, help="exposure reduction for RAW input")
    ap.add_argument("--seed", type=int, default=0, help="noise seed for RAW input")
    a = ap.parse_args()

    dev = get_device()
    path = Path(a.image)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dark, gt = load_inputs(path, a.gt, a.ev, a.seed)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    x = torch.from_numpy(dark.transpose(2, 0, 1)).float().div(255)[None].to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        pred = infer_full(model, x).float()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t
    net = np.clip(np.round(pred[0].cpu().numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    heq = histeq_y(dark)

    stem = path.stem.replace("_input", "")
    for suffix, img in (("input", dark), ("histeq", heq), ("network", net)):
        save(out / f"{stem}_{suffix}.png", img)

    panels = [("input (−3 EV + noise)", dark), ("baseline: hist. equalisation", heq),
              (f"network ({Path(a.checkpoint).parent.name})", net)]
    if gt is not None:
        lp = LPIPS(dev)
        g = gt.astype(np.float32) / 255
        gt_t = torch.from_numpy(g.transpose(2, 0, 1))[None]
        print(f"{'method':<10} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        titled = []
        for (title, img), key in zip(panels, ("input", "histeq", "network")):
            p = img.astype(np.float32) / 255
            m = (psnr(p, g), ssim(p, g), float(lp(torch.from_numpy(p.transpose(2, 0, 1))[None], gt_t)[0]))
            print(f"{key:<10} {m[0]:7.2f} {m[1]:7.4f} {m[2]:7.4f}")
            titled.append((f"{title}\nPSNR {m[0]:.2f} dB · SSIM {m[1]:.3f} · LPIPS {m[2]:.3f}", img))
        panels = titled + [("ground truth", gt)]

    fig, ax = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6))
    for axis, (title, img) in zip(ax, panels):
        axis.imshow(img)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"network time: {elapsed:.2f} s ({dark.shape[1]}x{dark.shape[0]}) -> {out}/{stem}_*.png")


if __name__ == "__main__":
    main()
