"""Evaluation on the 20 test images (full half-size resolution).

For every method (unchanged input, baselines, trained networks) it computes:
  * PSNR, SSIM, LPIPS (AlexNet) w.r.t. the GT - the metrics required by the assignment,
  * Delta E (CIE76, in Lab space) - colour error,
  * PSNR in clipped regions (GT >= 250 in any channel) and in dark regions
    (GT luminance < 0.1) - analysis of special cases.

  uv run python scripts/evaluate.py --runs runs/unet_prior_l1 runs/unet_srgb_l1 --main unet_prior_l1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.color import deltaE_cie76, rgb2lab
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.baselines import BASELINES  # noqa: E402
from lle.data import load_split, paths, read_rgb  # noqa: E402
from lle.metrics import LPIPS, mse, psnr_from_mse, ssim  # noqa: E402
from lle.models import build_model, infer_full  # noqa: E402
from lle.raw_pipeline import auto_bright_gain, render_isp  # noqa: E402
from lle.utils import get_device  # noqa: E402


def load_net(run: Path, dev):
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["config"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m


def masked_psnr(p, g, mask):
    if mask.sum() < 100:
        return np.nan
    return psnr_from_mse(float(np.mean((p[mask] - g[mask]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--main", default=None, help="network for the main table (directory name in runs/)")
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES))
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    dev = get_device()
    out = Path(a.out)
    lp = LPIPS(dev)
    nets = {Path(r).name: load_net(Path(r), dev) for r in a.runs}
    rows = []
    for name in tqdm(load_split(a.data, a.split)):
        pi, pg, pl = paths(a.data, name)
        inp, gt = read_rgb(pi), read_rgb(pg)
        gain_true = auto_bright_gain(read_rgb(pl, unchanged=True).astype(np.float32) / 65535)
        preds, times, extra = {"input": inp}, {}, {}
        for b in a.baselines:
            t = time.time()
            preds[b] = BASELINES[b](inp)
            times[b] = time.time() - t
        x = torch.from_numpy(inp.transpose(2, 0, 1)).float().div(255)[None].to(dev)
        for n, m in nets.items():
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = infer_full(m, x).float()
            torch.cuda.synchronize()
            times[n] = time.time() - t
            preds[n] = np.clip(np.round(p[0].cpu().numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
            if m.input_mode == "isp":
                # auto-exposure estimation accuracy: k from the network output vs k from the clean image
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lin = m.restore(x).float().clamp(0, 1).pow(2.2)
                po = render_isp(lin, gain_true)[0].cpu().numpy().transpose(1, 2, 0)
                extra[n] = {"gain_true": gain_true, "gain_est": float(auto_bright_gain(lin)),
                            "psnr_oracle_gain": psnr_from_mse(mse(np.round(po * 255) / 255, gt / 255.0))}

        g = gt.astype(np.float32) / 255
        g_lab = rgb2lab(g)
        clipped = gt.max(-1) >= 250
        dark = (0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]) < 0.1
        gt_t = torch.from_numpy(g.transpose(2, 0, 1))[None]
        for meth, pr in preds.items():
            p = pr.astype(np.float32) / 255
            if not a.no_save and meth != "input":
                d = out / "preds" / meth
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(pr, cv2.COLOR_RGB2BGR))
            rows.append({
                "image": name, "method": meth,
                "psnr": psnr_from_mse(mse(p, g)),
                "ssim": ssim(p, g),
                "lpips": float(lp(torch.from_numpy(p.transpose(2, 0, 1))[None], gt_t)[0]),
                "deltaE": float(deltaE_cie76(rgb2lab(p), g_lab).mean()),
                "psnr_clipped": masked_psnr(p, g, clipped),
                "psnr_dark": masked_psnr(p, g, dark),
                "clipped_frac": float(clipped.mean()),
                "time_s": times.get(meth, 0.0),
                **extra.get(meth, {}),
            })
        pd.DataFrame(rows).to_csv(out / f"per_image_{a.split}.csv", index=False)

    df = pd.DataFrame(rows)
    summ = df.groupby("method", sort=False).agg(
        psnr=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim=("ssim", "mean"), ssim_std=("ssim", "std"),
        lpips=("lpips", "mean"), lpips_std=("lpips", "std"),
        deltaE=("deltaE", "mean"), psnr_clipped=("psnr_clipped", "mean"),
        psnr_dark=("psnr_dark", "mean"), time_s=("time_s", "mean"),
    )
    summ.to_csv(out / f"summary_{a.split}.csv")
    print(summ.round(4).to_string())

    main_net = a.main or (next(iter(nets)) if nets else None)
    lines = ["| Method | PSNR [dB] ↑ | SSIM ↑ | LPIPS ↓ |", "|---|---|---|---|"]
    for label, meth in (("baseline (histogram equalisation)", "histeq"), (f"network ({main_net})", main_net)):
        if meth in summ.index:
            r = summ.loc[meth]
            lines.append(f"| {label} | {r.psnr:.2f} ± {r.psnr_std:.2f} | {r.ssim:.4f} ± {r.ssim_std:.4f} | {r.lpips:.4f} ± {r.lpips_std:.4f} |")
    (out / f"table_{a.split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
