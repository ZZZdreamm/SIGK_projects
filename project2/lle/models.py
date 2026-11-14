"""Network architecture for low light image enhancement.

``ResUNet`` - a U-Net (as in "Learning to See in the Dark", Chen et al. 2018) whose
encoder/decoder stages are built from residual blocks with channel attention
(as in RCAN). Channel attention uses global average pooling of the features, so
it gives the network information about the global brightness/colour of the scene.

``LowLightNet`` - a wrapper that implements the "physical" part of the task:

* ``input_mode="srgb"``  - the network gets the raw dark sRGB input
  and predicts a correction to it (global residual connection),
* ``input_mode="prior"`` - the input is first linearised (x^2.2),
  multiplied by 2^(-EV) = 8 (undoing the exposure reduction) and gamma-encoded
  again. The network gets both images (6 channels) and the residual connection
  starts from the exposure-compensated image - so the network mainly has to
  remove noise and fix the tone curve/colours instead of "inventing" brightness.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .raw_pipeline import auto_bright_gain, render_isp


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // reduction, 4), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int, ca: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ca = ChannelAttention(ch) if ca else nn.Identity()

    def forward(self, x):
        return x + self.ca(self.conv2(self.act(self.conv1(x))))


def _blocks(ch: int, n: int, ca: bool) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(ch, ca=ca) for _ in range(n)])


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=32, depth=4, n_blocks=1,
                 max_channels=256, channel_attention=True):
        super().__init__()
        chs = [min(base * 2**i, max_channels) for i in range(depth + 1)]
        self.depth = depth
        self.head = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc = nn.ModuleList([_blocks(chs[i], n_blocks, channel_attention) for i in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1) for i in range(depth)])
        self.bottleneck = _blocks(chs[depth], n_blocks, channel_attention)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * chs[i], chs[i], 1) for i in range(depth)])
        self.dec = nn.ModuleList([_blocks(chs[i], n_blocks, channel_attention) for i in range(depth)])
        self.tail = nn.Conv2d(chs[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.tail.weight)  # start: output = the image the correction is added to
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        h = self.head(x)
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for i in reversed(range(self.depth)):
            h = self.up[i](h)
            h = self.fuse[i](torch.cat([h, skips[i]], dim=1))
            h = self.dec[i](h)
        return self.tail(h)


def exposure_prior(x: torch.Tensor, ev_shift: float = -3.0, gamma: float = 2.2) -> torch.Tensor:
    """Undo the exposure reduction in the linear space (no denoising)."""
    return (x.clamp(0, 1).pow(gamma) * 2.0 ** (-ev_shift)).clamp(0, 1).pow(1.0 / gamma)


class LowLightNet(nn.Module):
    """Modes (``input_mode``):

    * ``srgb``  - end-to-end: sRGB input -> GT-like output (residual from the input),
    * ``prior`` - end-to-end with an exposure-compensated input (residual from it),
    * ``isp``   - split that follows the physics of how the GT is produced:
        ``restore``: the network recovers the CLEAN image at the original exposure
        (gamma 2.2 encoded: y ~ lin^(1/2.2)) - denoising + undoing -3 EV,
        ``finish``: the deterministic rawpy ISP - auto-bright from the 99th
        percentile of the WHOLE image + BT.709 curve (``raw_pipeline.render_isp``).
    """

    def __init__(self, input_mode: str = "prior", ev_shift: float = -3.0, **unet_kwargs):
        super().__init__()
        assert input_mode in ("srgb", "prior", "isp")
        self.input_mode = input_mode
        self.ev_shift = ev_shift
        in_ch = 3 if input_mode == "srgb" else 6
        self.unet = ResUNet(in_channels=in_ch, out_channels=3, **unet_kwargs)

    def restore(self, x):
        if self.input_mode == "srgb":
            base, inp = x, x
        else:
            base = exposure_prior(x, self.ev_shift)
            inp = torch.cat([x, base], dim=1)
        # pad to a multiple of 2^depth (full images have arbitrary size)
        m = 2 ** self.unet.depth
        h, w = x.shape[-2:]
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        out = self.unet(inp)[..., :h, :w]
        return base + out

    def finish(self, y, gain=None):
        """y -> output image. For ``isp``, gain=None means the auto-bright gain is
        estimated from the (full!) image y."""
        if self.input_mode != "isp":
            return y
        lin = y.float().clamp(0, 1).pow(2.2)
        if gain is None:
            gain = auto_bright_gain(lin)
        return render_isp(lin, gain)

    def forward(self, x, gain=None):
        return self.finish(self.restore(x), gain)


def build_model(cfg: dict) -> LowLightNet:
    return LowLightNet(
        input_mode=cfg.get("input_mode", "prior"),
        ev_shift=cfg.get("ev_shift", -3.0),
        base=cfg.get("base", 32),
        depth=cfg.get("depth", 4),
        n_blocks=cfg.get("n_blocks", 1),
        channel_attention=cfg.get("channel_attention", True),
    )


@torch.no_grad()
def infer_full(model: nn.Module, x: torch.Tensor, tile: int = 0, overlap: int = 64) -> torch.Tensor:
    """Prediction for a full 1x3xHxW image (the ``finish`` part - including the global
    auto-exposure - is always computed on the whole image). ``tile=0`` - the whole image
    at once (global context, needed e.g. by channel attention); when out of memory it
    falls back to overlapping tiles blended with a linear ramp at the borders."""
    if tile <= 0:
        try:
            return model.finish(model.restore(x)).float().clamp(0, 1)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            tile = 1024
    _, c, h, w = x.shape
    out = torch.zeros(1, 3, h, w, device=x.device)
    wsum = torch.zeros(1, 1, h, w, device=x.device)
    step = tile - overlap
    ramp = torch.ones(tile, device=x.device)
    r = torch.linspace(0.05, 1, overlap, device=x.device)
    ramp[:overlap], ramp[-overlap:] = r, r.flip(0)
    win = (ramp[:, None] * ramp[None, :])[None, None]
    for y in range(0, max(h - overlap, 1), step):
        for xx in range(0, max(w - overlap, 1), step):
            y0, x0 = min(y, max(h - tile, 0)), min(xx, max(w - tile, 0))
            patch = x[..., y0:y0 + tile, x0:x0 + tile]
            pred = model.restore(patch).float().clamp(0, 1)
            ww = win[..., : pred.shape[-2], : pred.shape[-1]]
            out[..., y0:y0 + pred.shape[-2], x0:x0 + pred.shape[-1]] += pred * ww
            wsum[..., y0:y0 + pred.shape[-2], x0:x0 + pred.shape[-1]] += ww
    return model.finish(out / wsum).float().clamp(0, 1)
