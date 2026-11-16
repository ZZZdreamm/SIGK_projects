"""Conditional GAN: U-Net generator (params -> image) and a PatchGAN discriminator."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _gn(c):
    return nn.GroupNorm(8, c)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                  _gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 4, stride=2, padding=1)
        self.res = ResBlock(cout)

    def forward(self, x):
        return self.res(self.conv(x))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin + cskip, cout, 3, padding=1)
        self.res = ResBlock(cout)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.res(self.conv(torch.cat([x, skip], 1)))


class Generator(nn.Module):
    """U-Net on the conditioning maps. The full-resolution path (1×1 + 3×3 convs)
    can evaluate per-pixel shading, the coarse levels give a large receptive
    field (needed when the network must find the sphere itself - `raw`)."""

    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()
        b = base
        self.stem = nn.Sequential(nn.Conv2d(in_ch, b, 1), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1),
                                  ResBlock(b))
        self.d1, self.d2 = Down(b, 2 * b), Down(2 * b, 4 * b)
        self.d3, self.d4 = Down(4 * b, 4 * b), Down(4 * b, 8 * b)
        self.mid = nn.Sequential(ResBlock(8 * b), ResBlock(8 * b))
        self.u4, self.u3 = Up(8 * b, 4 * b, 4 * b), Up(4 * b, 4 * b, 4 * b)
        self.u2, self.u1 = Up(4 * b, 2 * b, 2 * b), Up(2 * b, b, b)
        self.head = nn.Sequential(ResBlock(b), _gn(b), nn.SiLU(), nn.Conv2d(b, 3, 1))

    def forward(self, cond):
        s0 = self.stem(cond)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.mid(self.d4(s3))
        x = self.u4(x, s3)
        x = self.u3(x, s2)
        x = self.u2(x, s1)
        x = self.u1(x, s0)
        return torch.sigmoid(self.head(x))


class PatchDiscriminator(nn.Module):
    """pix2pix-style PatchGAN (receptive field 70×70) with spectral norm,
    conditioned on the same maps as the generator."""

    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()
        b = base

        def block(cin, cout, stride):
            return [spectral_norm(nn.Conv2d(cin, cout, 4, stride, 1)), nn.LeakyReLU(0.2, True)]

        self.net = nn.Sequential(*block(in_ch + 3, b, 2), *block(b, 2 * b, 2), *block(2 * b, 4 * b, 2),
                                 *block(4 * b, 8 * b, 1), spectral_norm(nn.Conv2d(8 * b, 1, 4, 1, 1)))

    def forward(self, cond, img, features: bool = False):
        x = torch.cat([cond, img * 2 - 1], 1)
        if not features:
            return self.net(x)
        feats = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return x, feats


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
