#!/usr/bin/env python
"""Evaluate the baseline and trained model(s) on the full DIV2K test set
(all 256x256 tiles in data/patches/test) with MSE / PSNR / SSIM / LPIPS.

Examples:
    python scripts/evaluate.py --task sr:32                                   # baseline only
    python scripts/evaluate.py --task sr:32 --checkpoint runs/resunet_sr_32x32/best.pt
    python scripts/evaluate.py --task deblur:3 --checkpoint runs/resunet_deblur_gauss_3x3/best.pt runs/plain_deblur_gauss_3x3/best.pt

Results are merged into results/<task_name>.json (so baselines and models can
be evaluated in separate invocations), per-image metrics go to
results/per_image/<task_name>.csv and example comparisons to results/figures/.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sigk.data import PatchDataset  # noqa: E402
from sigk.metrics import LPIPS, MetricAccumulator  # noqa: E402
from sigk.models import build_model, compose_output  # noqa: E402
from sigk.tasks import parse_task  # noqa: E402
from sigk.utils import cpu_count, get_device, load_json, save_json, tensor_to_hwc, write_image  # noqa: E402
from sigk.visualize import comparison_grid  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", nargs="*", default=[], help="model checkpoints (best.pt) to evaluate")
    ap.add_argument("--data", default="data/patches/test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=min(8, cpu_count()))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test patches")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--n-vis", type=int, default=6, help="number of example patches in the comparison figure")
    ap.add_argument("--seed", type=int, default=2024, help="seed of the deterministic test degradations")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_model(path: str, device: torch.device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"], in_channels=cfg["in_channels"], **cfg["model_kwargs"]).to(device).eval()
    model.load_state_dict(ck["model"])
    model = model.to(memory_format=torch.channels_last)
    run_name = Path(path).resolve().parent.name
    label = run_name.replace(f"_{cfg['task_name']}", "") or cfg["model"]
    return model, label, cfg


def main():
    args = parse_args()
    device = get_device(args.device)
    spec = parse_task(args.task)
    out = Path(args.out)
    with_baseline = not args.skip_baseline

    ds = PatchDataset(args.data, spec, train=False, deterministic=True, seed=args.seed, limit=args.limit, with_baseline=with_baseline)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    models = [load_model(p, device) for p in args.checkpoint]
    for _, label, cfg in models:
        assert cfg["task_name"] == spec.name, f"checkpoint {label} was trained for {cfg['task_name']}, not {spec.name}"

    acc = MetricAccumulator(LPIPS(device))
    name_input = spec.method_name("degraded_input")
    name_base = spec.method_name(spec.baseline_name)
    vis_rows: list[dict] = []

    for batch in tqdm(dl, desc=f"evaluating {spec.name}"):
        x, y, mask, idx = batch[:4]
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        preds = {}
        if spec.has_identity_row():
            preds[name_input] = x[:, :3]
        if with_baseline:
            preds[name_base] = batch[4].to(device)
        for model, label, _ in models:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                p = model(x.contiguous(memory_format=torch.channels_last))
            preds[spec.method_name(label)] = compose_output(p.float(), x, mask, spec.task).clamp(0, 1)
        for name, p in preds.items():
            acc.update(name, p, y, idx)
        for i in range(x.shape[0]):
            if len(vis_rows) >= args.n_vis:
                break
            row = {"ground_truth": tensor_to_hwc(y[i]), "input": tensor_to_hwc(x[i, :3])}
            row.update({n: tensor_to_hwc(p[i]) for n, p in preds.items() if n != name_input})
            vis_rows.append((int(idx[i]), row))

    summary = acc.summary()
    # merge with previous results for this task
    json_path = out / f"{spec.name}.json"
    result = load_json(json_path) if json_path.exists() else {}
    result.update({"task": spec.task, "task_name": spec.name, "description": spec.description, "n_test_images": len(ds)})
    result.setdefault("methods", {}).update(summary)
    save_json(result, json_path)

    out.joinpath("per_image").mkdir(parents=True, exist_ok=True)
    csv_path = out / "per_image" / f"{spec.name}.csv"
    rows = acc.rows
    if csv_path.exists():  # keep rows of methods evaluated in earlier invocations
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["method"] not in summary] + rows
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "idx", "mse", "psnr", "ssim", "lpips"]); w.writeheader(); w.writerows(rows)

    # example figure + raw example images
    columns = ["input"] + [n for n in summary if n != name_input] + ["ground_truth"]
    comparison_grid([r for _, r in vis_rows], columns, out / "figures" / f"{spec.name}_examples.png", title=spec.description)
    for i, row in vis_rows:
        for n, img in row.items():
            write_image(out / "images" / spec.name / f"{i:05d}_{n}.png", img)

    print(f"\n{spec.name}: {spec.description}  (n = {len(ds)} test patches)")
    print(f"{'method':48s} {'MSE':>9s} {'PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s}")
    for name, m in summary.items():
        print(f"{name:48s} {m['mse']:9.5f} {m['psnr']:8.3f} {m['ssim']:7.4f} {m['lpips']:7.4f}")
    print(f"saved -> {json_path}")


if __name__ == "__main__":
    main()
