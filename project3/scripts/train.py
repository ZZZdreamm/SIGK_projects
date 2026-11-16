"""Train the conditional GAN neural renderer.

    uv run python scripts/train.py --name geometric_gan --encoding geometric
    uv run python scripts/train.py --name geometric_l1 --encoding geometric --adv-weight 0
    uv run python scripts/train.py --name geometric_gan_v2 --encoding geometric --adv-mask --fm-weight 10 --select lpips
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nr.data import GPUData
from nr.encoding import CHANNELS, encode
from nr.metrics import LPIPS
from nr.models import Generator, PatchDiscriminator, count_params


def fg_mask(img: torch.Tensor, dilate: int = 2) -> torch.Tensor:
    """Sphere pixels (background is exactly black) dilated by `dilate` px."""
    m = (img.amax(1, keepdim=True) > 0).float()
    return F.max_pool2d(m, 2 * dilate + 1, 1, dilate)


def rec_loss(pred, ref, fg_weight):
    """L1 over the whole image + (L1 + MSE) averaged over the (dilated) sphere.

    The sphere covers <1 % of the image, without the second term the loss is
    dominated by the trivial black background. The MSE part matters: the median
    of the pixels inside the (dilated) mask is close to black (dark side of the
    sphere + the dilation ring), so with L1 alone "everything black" is a strong
    local minimum; MSE is mean-seeking and pulls the network out of it."""
    err = (pred - ref).abs()
    m = fg_mask(ref, dilate=1)
    denom = m.sum() * 3 + 1
    fg = (err * m).sum() / denom
    fg2 = (err ** 2 * m).sum() / denom
    return err.mean() + fg_weight * (fg + 5 * fg2), fg


def adv_mask(ref, fake, shape, dilate=4):
    """D output cells that see the sphere (in the reference or in the prediction).

    Without it the adversarial loss is averaged over a 14×14 map whose cells are
    >95 % pure black background - identical for real and fake - so D settles at
    the LSGAN chance level (0.25) and its gradient on the sphere vanishes."""
    m = torch.maximum(fg_mask(ref, dilate), fg_mask((fake.detach() > 0.02).float(), dilate))
    return F.adaptive_max_pool2d(m, shape)


def masked_mean(x, m):
    return (x * m).sum() / (m.sum() + 1e-6)


@torch.no_grad()
def evaluate(G, params, images, encoding, lp, bs=100):
    l1, fg, lpv = 0.0, 0.0, 0.0
    outs = []
    for i in range(0, len(params), bs):
        pred = G(encode(params[i:i + bs], encoding))
        ref = images[i:i + bs]
        l1 += (pred - ref).abs().mean().item() * len(ref)
        fg += rec_loss(pred, ref, 1.0)[1].item() * len(ref)
        lpv += lp(ref, pred).sum().item()
        outs.append(pred)
    n = len(params)
    return {'val_l1': l1 / n, 'val_fg_l1': fg / n, 'val_lpips': lpv / n}, torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--encoding', default='geometric', choices=list(CHANNELS))
    ap.add_argument('--data', default='data/dataset')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rec-weight', type=float, default=100.0)
    ap.add_argument('--fg-weight', type=float, default=1.0)
    ap.add_argument('--adv-weight', type=float, default=1.0, help='0 = no discriminator (pure L1 regression)')
    ap.add_argument('--adv-mask', action='store_true', help='adversarial loss only on D cells covering the sphere')
    ap.add_argument('--fm-weight', type=float, default=0.0, help='D feature-matching loss (pix2pixHD)')
    ap.add_argument('--select', default='fg_l1', choices=['fg_l1', 'lpips'], help='checkpoint selection metric')
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path('runs') / args.name
    (out / 'samples').mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / 'config.json', 'w'), indent=2)

    data = GPUData(args.data, dev)
    tr_p, tr_i = data.subset('train')
    va_p, va_i = data.subset('val')
    cin = CHANNELS[args.encoding]
    G = Generator(cin, args.base).to(dev)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    use_gan = args.adv_weight > 0
    D = PatchDiscriminator(cin).to(dev) if use_gan else None
    print(f'G: {count_params(G) / 1e6:.2f} M params' + (f', D: {count_params(D) / 1e6:.2f} M' if use_gan else ''))

    opt_g = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999)) if use_gan else None
    steps_per_epoch = len(tr_p) // args.batch
    total = args.epochs * steps_per_epoch
    sched = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, 2 * (1 - s / total)))
             for o in (opt_g, opt_d) if o is not None]
    lp = LPIPS(dev)
    amp = lambda: torch.autocast('cuda', torch.bfloat16)

    log = open(out / 'log.csv', 'w', newline='')
    writer = None
    best = float('inf')
    vis_idx = torch.arange(16, device=dev)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        G.train()
        perm = torch.randperm(len(tr_p), device=dev)
        acc = {'loss_g': 0.0, 'loss_d': 0.0, 'rec': 0.0, 'adv': 0.0}
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            p, ref = tr_p[idx], tr_i[idx]
            cond = encode(p, args.encoding)
            with amp():
                fake = G(cond).float()

            if use_gan:
                opt_d.zero_grad(set_to_none=True)
                with amp():
                    d_real, d_fake = D(cond, ref).float(), D(cond, fake.detach()).float()
                if args.adv_mask:
                    am = adv_mask(ref, fake, d_real.shape[-2:])
                    loss_d = 0.5 * (masked_mean((d_real - 1) ** 2, am) + masked_mean(d_fake ** 2, am))
                else:
                    loss_d = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
                loss_d.backward()
                opt_d.step()
                acc['loss_d'] += loss_d.item()

            opt_g.zero_grad(set_to_none=True)
            rec, _ = rec_loss(fake, ref, args.fg_weight)
            loss_g = args.rec_weight * rec
            if use_gan:
                with amp():
                    d_out, f_fake = D(cond, fake, features=True)
                    if args.fm_weight > 0:
                        with torch.no_grad():
                            _, f_real = D(cond, ref, features=True)
                d_out = d_out.float()
                adv = masked_mean((d_out - 1) ** 2, am) if args.adv_mask else ((d_out - 1) ** 2).mean()
                loss_g = loss_g + args.adv_weight * adv
                if args.fm_weight > 0:
                    fm = sum((a.float() - b.float()).abs().mean() for a, b in zip(f_fake, f_real)) / len(f_fake)
                    loss_g = loss_g + args.fm_weight * fm
                acc['adv'] += adv.item()
            loss_g.backward()
            opt_g.step()
            for sc in sched:
                sc.step()
            with torch.no_grad():
                for pe, pg in zip(G_ema.parameters(), G.parameters()):
                    pe.lerp_(pg, 1 - args.ema)
            acc['loss_g'] += loss_g.item()
            acc['rec'] += rec.item()

        row = {'epoch': epoch, 'time': round(time.time() - t0, 1),
               **{k: v / steps_per_epoch for k, v in acc.items()}}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, preds = evaluate(G_ema, va_p, va_i, args.encoding, lp)
            row.update(val)
            score = val['val_fg_l1'] if args.select == 'fg_l1' else val['val_lpips']
            if score < best:
                best = score
                torch.save({'G': G_ema.state_dict(), 'encoding': args.encoding, 'base': args.base,
                            'epoch': epoch, **val}, out / 'best.pt')
            if epoch % (args.eval_every * 10) == 0 or epoch == args.epochs:
                save_image(torch.cat([va_i[vis_idx], preds[vis_idx]]), out / 'samples' / f'{epoch:04}.png', nrow=16)
            print(f"[{args.name}] ep {epoch:4d} {row['time']:7.0f}s rec {row['rec']:.4f} "
                  f"adv {row['adv']:.3f} D {row['loss_d']:.3f} | val L1 {val['val_l1']:.4f} "
                  f"fgL1 {val['val_fg_l1']:.4f} LPIPS {val['val_lpips']:.4f}", flush=True)
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=['epoch', 'time', 'loss_g', 'loss_d', 'rec', 'adv',
                                                     'val_l1', 'val_fg_l1', 'val_lpips'])
            writer.writeheader()
        writer.writerow(row)
        log.flush()
    torch.save({'G': G_ema.state_dict(), 'encoding': args.encoding, 'base': args.base, 'epoch': args.epochs},
               out / 'last.pt')


if __name__ == '__main__':
    main()
