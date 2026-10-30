#!/usr/bin/env python
"""Train a restoration network for one task configuration.

Example:
    python scripts/train.py --task sr:32 --epochs 30
    python scripts/train.py --task inpaint:32 --model resunet --loss l1 --lpips-weight 0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sigk.data import PatchDataset  # noqa: E402
from sigk.losses import RestorationLoss  # noqa: E402
from sigk.metrics import LPIPS, MetricAccumulator  # noqa: E402
from sigk.models import build_model, compose_output  # noqa: E402
from sigk.tasks import parse_task  # noqa: E402
from sigk.utils import count_parameters, cpu_count, get_device, seed_everything  # noqa: E402
from sigk.visualize import training_curves  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="e.g. sr:32, denoise:0.01, deblur:3, inpaint:32")
    ap.add_argument("--data", default="data/patches", help="folder with train/ val/ test/ patch dirs")
    ap.add_argument("--out", default=None, help="run dir (default runs/<model>_<task>)")
    ap.add_argument("--model", default="resunet", choices=["resunet", "plain"])
    ap.add_argument("--base", type=int, default=64, help="base channel width")
    ap.add_argument("--depth", type=int, default=3, help="U-Net depth (resunet only)")
    ap.add_argument("--n-blocks", type=int, default=2, help="res blocks per stage (resunet) / total (plain: 16 typical)")
    ap.add_argument("--no-ca", action="store_true", help="disable channel attention")
    ap.add_argument("--loss", default="l1", choices=["l1", "l2", "charbonnier"])
    ap.add_argument("--lpips-weight", type=float, default=0.0, help="add w * LPIPS(vgg) perceptual loss")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=min(8, cpu_count()))
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=None, help="use only the first N training patches")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="resume from <out>/last.pt")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


@torch.no_grad()
def validate(model, loader, spec, device, lpips_fn, amp):
    model.eval()
    acc = MetricAccumulator(lpips_fn)
    for x, y, mask, idx in loader:
        x, y, mask = x.to(device, non_blocking=True), y.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(x.contiguous(memory_format=torch.channels_last))
        pred = compose_output(pred.float(), x, mask, spec.task)
        acc.update("model", pred, y, idx)
    return acc.summary()["model"]


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    spec = parse_task(args.task)
    out = Path(args.out or f"runs/{args.model}_{spec.name}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds = PatchDataset(Path(args.data) / "train", spec, train=True, deterministic=False, seed=args.seed, limit=args.train_limit)
    val_ds = PatchDataset(Path(args.data) / "val", spec, train=False, deterministic=True, seed=12345, limit=args.val_limit)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model_kwargs = dict(base=args.base, channel_attention=not args.no_ca)
    if args.model == "resunet":
        model_kwargs.update(depth=args.depth, n_blocks=args.n_blocks)
    else:
        model_kwargs.update(n_blocks=args.n_blocks if args.n_blocks != 2 else 16)
    model = build_model(args.model, in_channels=spec.in_channels, **model_kwargs).to(device)
    model = model.to(memory_format=torch.channels_last)  # ~30% faster convs with AMP
    criterion = RestorationLoss(args.loss, args.lpips_weight).to(device)
    lpips_fn = LPIPS(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    steps_per_epoch = len(train_dl)
    total_steps = args.epochs * steps_per_epoch
    warmup = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=amp)

    config = {**vars(args), "task_name": spec.name, "task_description": spec.description,
              "in_channels": spec.in_channels, "model_kwargs": model_kwargs,
              "n_parameters": count_parameters(model), "n_train": len(train_ds), "n_val": len(val_ds)}
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"task={spec.name} ({spec.description})  model={args.model} params={config['n_parameters']/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)} device={device} amp={amp}")

    start_epoch, best_psnr, history = 0, -1.0, []
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"]); start_epoch = ck["epoch"] + 1; best_psnr = ck["best_psnr"]; history = ck["history"]
        print(f"resumed from epoch {start_epoch}")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_psnr": best_psnr, "history": history,
                    "config": config}, path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        t0, running, n = time.time(), 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for x, y, mask in (b[:3] for b in pbar):
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred = model(x.contiguous(memory_format=torch.channels_last))
            pred = compose_output(pred.float(), x, mask, spec.task)
            loss, logs = criterion(pred, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item(); n += 1
            pbar.set_postfix(loss=f"{running/n:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        val = validate(model, val_dl, spec, device, lpips_fn, amp)
        row = {"epoch": epoch + 1, "train_loss": running / max(n, 1), "val_psnr": val["psnr"], "val_ssim": val["ssim"],
               "val_lpips": val["lpips"], "val_mse": val["mse"], "lr": sched.get_last_lr()[0], "time_s": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch+1:3d} loss {row['train_loss']:.4f} | val PSNR {val['psnr']:.3f} SSIM {val['ssim']:.4f} "
              f"LPIPS {val['lpips']:.4f} | {row['time_s']:.0f}s")
        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            save(out / "best.pt", epoch)
        save(out / "last.pt", epoch)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

    if history:
        training_curves(out / "history.csv", out / "training_curves.png", title=f"{args.model} / {spec.name}")
    print(f"done. best val PSNR {best_psnr:.3f}. checkpoints in {out}")


if __name__ == "__main__":
    main()
