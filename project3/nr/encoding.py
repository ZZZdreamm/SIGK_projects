"""Encodings of the scene parameters into per-pixel conditioning maps.

All three encodings produce a tensor [B, C, H, W] that is the input of the
generator (and the condition of the discriminator):

* ``raw``       - the 10 raw parameters (world space) broadcast to every pixel
                  + pixel coordinates (CoordConv). The network has to learn the
                  camera projection by itself.
* ``camera``    - parameters expressed relative to the camera / the object:
                  object centre in camera space, its projection on the screen and
                  apparent (angular) size, direction and distance of the light
                  *relative to the object* (in camera space), colour, shininess
                  + pixel coordinates.
* ``geometric`` - ``camera`` + two per-pixel maps derived analytically from the
                  parameters: camera ray direction and the offset of the ray
                  from the direction to the sphere centre, normalised by the
                  angular radius of the sphere (|q| < 1 <=> the ray hits the
                  sphere). No shading is computed - the network still has to
                  learn normals, N·L, the reflection, the highlight etc.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from nr import scene

ENCODINGS = ('raw', 'camera', 'geometric')
CHANNELS = {'raw': 12, 'camera': 19, 'geometric': 25}


_RAYS = {}


def _rays(size, device):
    key = (size, str(device))
    if key not in _RAYS:
        _RAYS[key] = _const(scene.pixel_rays(size), device).permute(2, 0, 1)[None]
    return _RAYS[key]


def _const(a, device):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def _broadcast(v: torch.Tensor, size: int) -> torch.Tensor:
    return v[:, :, None, None].expand(-1, -1, size, size)


def _coords(b: int, size: int, device) -> torch.Tensor:
    c = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    y, x = torch.meshgrid(-c, c, indexing='ij')
    return torch.stack([x, y])[None].expand(b, -1, -1, -1)


def camera_features(p: torch.Tensor) -> torch.Tensor:
    """[B,10] raw params -> [B,17] relative / camera-space features."""
    dev = p.device
    R, t = _const(scene.VIEW[:3, :3], dev), _const(scene.VIEW[:3, 3], dev)
    obj, light = p[:, 0:3], p[:, 3:6]
    color, shin = p[:, 6:9] / 255.0, p[:, 9:10]
    c = obj @ R.T + t
    lc = light @ R.T + t
    depth = -c[:, 2:3]
    ndc = c[:, :2] / (depth * scene.TAN_HALF_FOV)
    dist = c.norm(dim=1, keepdim=True)
    sin_a = (scene.SPHERE_RADIUS / dist).clamp(max=1.0)
    rel = lc - c
    rel_d = rel.norm(dim=1, keepdim=True)
    feats = [
        c / scene.POS_RANGE,
        c / dist,
        ndc,
        sin_a * 10,
        rel / rel_d,
        torch.log(rel_d) / math.log(40.0),
        color,
        (shin - scene.SHININESS_RANGE[0]) / (scene.SHININESS_RANGE[1] - scene.SHININESS_RANGE[0]),
    ]
    return torch.cat(feats, 1)


def raw_features(p: torch.Tensor) -> torch.Tensor:
    return torch.cat([p[:, 0:6] / scene.POS_RANGE, p[:, 6:9] / 255.0,
                      (p[:, 9:10] - 3.0) / 17.0], 1)


def encode(p: torch.Tensor, kind: str, size: int = scene.IMAGE_SIZE) -> torch.Tensor:
    b, dev = p.shape[0], p.device
    xy = _coords(b, size, dev)
    if kind == 'raw':
        return torch.cat([_broadcast(raw_features(p), size), xy], 1)
    f = camera_features(p)
    out = [_broadcast(f, size), xy]
    if kind == 'geometric':
        rays = _rays(size, dev).expand(b, -1, -1, -1)
        cdir = f[:, 3:6][:, :, None, None]
        sin_a = (f[:, 8:9] / 10)[:, :, None, None]
        cos_t = (rays * cdir).sum(1, keepdim=True)
        q = (rays - cos_t * cdir) / sin_a
        q = q / q.norm(dim=1, keepdim=True).clamp(min=2.0) * 2.0
        out += [rays, q]
    elif kind != 'camera':
        raise ValueError(kind)
    return torch.cat(out, 1)
