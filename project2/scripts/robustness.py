"""Robustness to conditions different from training.

Test images are degraded again (from the stored linear image) for different
exposure reductions (EV) and noise levels. PSNR / SSIM of the networks and the
baselines are reported. The networks are trained only for -3 EV, so the test
shows how much the model has "memorised" the specific degradation.

  uv run python scripts/robustness.py --runs runs/unet_prior runs/unet_prior_jitter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.baselines import BASELINES  # noqa: E402
from lle.data import load_split, paths, read_rgb  # noqa: E402
from lle.metrics import mse, psnr_from_mse, ssim  # noqa: E402
from lle.models import build_model, infer_full  # noqa: E402
from lle.raw_pipeline import degrade_torch  # noqa: E402
from lle.utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--evs", nargs="+", type=float, default=[-2.0, -2.5, -3.0, -3.5, -4.0])
    ap.add_argument("--noise-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", default="results/robustness.csv")
    a = ap.parse_args()

    dev = get_device()
    nets = {}
    for r in a.runs:
        ck = torch.load(Path(r) / "best.pt", map_location="cpu", weights_only=False)
        m = build_model(ck["config"]).to(dev).eval()
        m.load_state_dict(ck["model"])
        nets[Path(r).name] = m

    # conditions: EV change at nominal noise and noise change at -3 EV
    conds = [("ev", ev, 1.0) for ev in a.evs] + [("noise", -3.0, s) for s in a.noise_scales if s != 1.0]
    rows = []
    torch.manual_seed(0)
    for name in tqdm(load_split(a.data, "test")):
        _, pg, pl = paths(a.data, name)
        g = read_rgb(pg).astype(np.float32) / 255
        lin = torch.from_numpy(read_rgb(pl, unchanged=True).astype(np.float32) / 65535).permute(2, 0, 1)[None].to(dev)
        for kind, ev, ns in conds:
            x = degrade_torch(lin, ev, 0.001 * ns, 0.005 * ns)
            xin = np.clip(np.round(x[0].cpu().numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
            preds = {b: BASELINES[b](xin).astype(np.float32) / 255 for b in ("histeq", "isp_nlm")}
            for n, m in nets.items():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    preds[n] = infer_full(m, x).float()[0].cpu().numpy().transpose(1, 2, 0)
            for meth, p in preds.items():
                rows.append({"image": name, "kind": kind, "ev": ev, "noise_scale": ns, "method": meth,
                             "psnr": psnr_from_mse(mse(p, g)), "ssim": ssim(p, g)})
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["kind", "ev", "noise_scale", "method"])[["psnr", "ssim"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
