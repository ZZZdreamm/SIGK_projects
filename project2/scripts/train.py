"""Training of the LowLightNet network.

Examples:
  uv run python scripts/train.py --name isp
  uv run python scripts/train.py --name e2e_srgb --input-mode srgb
  uv run python scripts/train.py --name isp_fixed --data-mode fixed
  uv run python scripts/train.py --name isp_lpips --lpips-weight 0.1
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.data import FullImageDataset, PatchDataset  # noqa: E402
from lle.losses import RestorationLoss  # noqa: E402
from lle.metrics import psnr  # noqa: E402
from lle.models import build_model, infer_full  # noqa: E402
from lle.raw_pipeline import degrade_torch  # noqa: E402
from lle.utils import count_parameters, get_device, save_json, seed_everything  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data-mode", choices=["online", "fixed"], default="online",
                    help="online: fresh noise in every iteration; fixed: only the stored PNG pairs")
    ap.add_argument("--noise-jitter", action="store_true",
                    help="random noise level (alpha, sigma x[0.5, 2]) - better robustness")
    ap.add_argument("--input-mode", choices=["srgb", "prior", "isp"], default="isp")
    ap.add_argument("--lin-weight", type=float, default=1.0,
                    help="isp: weight of the L1 loss on the recovered clean image (before the ISP)")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--no-ca", action="store_true")
    ap.add_argument("--pixel-loss", default="l1", choices=["l1", "l2", "charbonnier"])
    ap.add_argument("--lpips-weight", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--crops-per-image", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse()
    seed_everything(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = get_device()
    out = Path(a.runs) / a.name
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(a.data)
    cfg = {**vars(a), "ev_shift": meta["ev_shift"], "channel_attention": not a.no_ca}
    save_json(cfg, out / "config.json")

    model = build_model(cfg).to(dev)
    print(f"params: {count_parameters(model) / 1e6:.2f} M")
    train_ds = PatchDataset(a.data, "train", a.patch, a.crops_per_image)
    val_ds = FullImageDataset(a.data, "val")  # full images - auto-exposure is global
    dl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=True, persistent_workers=True)
    crit = RestorationLoss(a.pixel_loss, a.lpips_weight).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))
    tb = SummaryWriter(out / "tb")

    best, log_rows, step = -1.0, [], 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for b in tqdm(dl, desc=f"ep {ep}/{a.epochs}", leave=False, disable=not sys.stderr.isatty()):
            gt = b["gt"].to(dev, non_blocking=True)
            if a.data_mode == "online":
                lin = b["linear"].to(dev, non_blocking=True)
                alpha, sigma = meta["alpha"], meta["sigma"]
                if a.noise_jitter:
                    B = lin.shape[0]
                    alpha = alpha * 2 ** torch.empty(B, 1, 1, 1, device=dev).uniform_(-1, 1)
                    sigma = sigma * 2 ** torch.empty(B, 1, 1, 1, device=dev).uniform_(-1, 1)
                x = degrade_torch(lin, meta["ev_shift"], alpha, sigma, meta["gamma"])
            else:
                x = b["input"].to(dev, non_blocking=True)
            lin_gt = b["linear"].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                y = model.restore(x)
            if a.input_mode == "isp":
                # loss in GT space (after the ISP with the known gain of the image)
                # + loss on the clean image before the ISP (gradient also in clipped areas)
                pred = model.finish(y.float(), b["gain"].to(dev))
                loss, logs = crit(pred, gt)
                l_lin = (y.float() - lin_gt.clamp(0, 1).pow(1 / 2.2)).abs().mean()
                logs["lin"] = l_lin.item()
                loss = loss + a.lin_weight * l_lin
            else:
                loss, logs = crit(y.float(), gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 50 == 0:
                for k, v in logs.items():
                    tb.add_scalar(f"train/{k}", v, step)

        if ep % a.val_every and ep != a.epochs:
            print(f"ep {ep:3d}  loss {np.mean(losses):.4f}  ({time.time() - t0:.0f}s)", flush=True)
            continue
        val_psnr = validate(model, val_ds, dev)
        row = {"epoch": ep, "loss": float(np.mean(losses)), "val_psnr": val_psnr,
               "lr": sched.get_last_lr()[0], "time_s": time.time() - t0}
        log_rows.append(row)
        tb.add_scalar("train/epoch_loss", row["loss"], ep)
        tb.add_scalar("val/psnr", val_psnr, ep)
        print(f"ep {ep:3d}  loss {row['loss']:.4f}  val PSNR {val_psnr:.2f}  ({row['time_s']:.0f}s)", flush=True)
        state = {"model": model.state_dict(), "config": cfg, "epoch": ep, "val_psnr": val_psnr}
        torch.save(state, out / "last.pt")
        if val_psnr > best:
            best = val_psnr
            torch.save(state, out / "best.pt")
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(log_rows)
    print(f"best val PSNR {best:.2f}")


def load_meta(root):
    from lle.utils import load_json

    return load_json(Path(root) / "split.json")


@torch.no_grad()
def validate(model, ds, dev) -> float:
    model.eval()
    vals = []
    for i in range(len(ds)):
        s = ds[i]
        x = s["input"][None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            p = infer_full(model, x).float()
        vals.append(psnr(p[0].cpu().numpy(), s["gt"].numpy()))
    return float(np.mean(vals))


if __name__ == "__main__":
    main()
