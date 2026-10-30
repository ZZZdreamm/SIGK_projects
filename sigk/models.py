"""Network architectures.

``ResUNet`` (the proposed architecture) is a U-Net whose encoder/decoder stages
are built from residual blocks with channel attention (as in RCAN). A global
residual connection makes the network predict the *correction* to its input,
which suits all four restoration tasks because the degraded input is already
a good first guess of the output.

``PlainResNet`` (an EDSR-like single-scale residual network) is provided as a
second architecture for ablation experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
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
    def __init__(self, ch: int, ca: bool = True, res_scale: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.ca = ChannelAttention(ch) if ca else nn.Identity()
        self.res_scale = res_scale

    def forward(self, x):
        r = self.conv2(F.relu(self.conv1(x), inplace=True))
        return x + self.res_scale * self.ca(r)


def _blocks(ch: int, n: int, ca: bool) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(ch, ca=ca) for _ in range(n)])


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 64,
        depth: int = 3,
        n_blocks: int = 2,
        max_channels: int = 256,
        channel_attention: bool = True,
        global_residual: bool = True,
    ):
        super().__init__()
        chs = [min(base * 2**i, max_channels) for i in range(depth + 1)]
        self.global_residual = global_residual
        self.out_channels = out_channels

        self.head = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc = nn.ModuleList([_blocks(chs[i], n_blocks, channel_attention) for i in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1) for i in range(depth)])
        self.bottleneck = _blocks(chs[depth], n_blocks, channel_attention)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * chs[i], chs[i], 1) for i in range(depth)])
        self.dec = nn.ModuleList([_blocks(chs[i], n_blocks, channel_attention) for i in range(depth)])
        self.tail = nn.Conv2d(chs[0], out_channels, 3, padding=1)
        if global_residual:  # start as identity mapping (= degraded input / bicubic for SR)
            nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        h = self.head(x)
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for i in reversed(range(len(self.dec))):
            h = self.up[i](h)
            h = self.fuse[i](torch.cat([h, skips[i]], dim=1))
            h = self.dec[i](h)
        out = self.tail(h)
        if self.global_residual:
            out = out + x[:, : self.out_channels]
        return out


class PlainResNet(nn.Module):
    """EDSR-style: head conv -> N residual blocks -> conv -> + global residual."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 64,
        n_blocks: int = 16,
        channel_attention: bool = True,
        global_residual: bool = True,
    ):
        super().__init__()
        self.global_residual = global_residual
        self.out_channels = out_channels
        self.head = nn.Conv2d(in_channels, base, 3, padding=1)
        self.body = _blocks(base, n_blocks, channel_attention)
        self.body_conv = nn.Conv2d(base, base, 3, padding=1)
        self.tail = nn.Conv2d(base, out_channels, 3, padding=1)
        if global_residual:
            nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        h = self.head(x)
        h = h + self.body_conv(self.body(h))
        out = self.tail(h)
        if self.global_residual:
            out = out + x[:, : self.out_channels]
        return out


MODELS = {"resunet": ResUNet, "plain": PlainResNet}


def build_model(name: str, in_channels: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', choose from {list(MODELS)}")
    return MODELS[name](in_channels=in_channels, **kwargs)


def compose_output(pred: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, task: str) -> torch.Tensor:
    """Final prediction. For inpainting we keep the known pixels untouched and
    only use the network output inside the holes."""
    if task == "inpaint":
        return mask * pred + (1.0 - mask) * x[:, :3]
    return pred
