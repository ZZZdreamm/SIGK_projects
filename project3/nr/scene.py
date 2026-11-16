"""Scene definition shared by the OpenGL renderer and the neural renderer.

Everything here is plain numpy, so the dataset generator (inside the provided
moderngl renderer) and the PyTorch code use exactly the same camera and the
same parameter sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

IMAGE_SIZE = 128
SPHERE_RADIUS = 1.0

CAMERA_POSITION = np.array([5.0, 5.0, 15.0])
CAMERA_TARGET = np.array([0.0, 0.0, 0.0])
CAMERA_UP = np.array([0.0, 1.0, 0.0])
FOV_Y_DEG = 45.0
NEAR, FAR = 0.1, 1000.0

POS_RANGE = 20.0
SHININESS_RANGE = (3.0, 20.0)
MATERIAL_AMBIENT = np.array([76, 76, 76]) / 255.0
MATERIAL_SPECULAR = np.array([255, 255, 255]) / 255.0
LIGHT_AMBIENT = np.array([25, 25, 25]) / 255.0
LIGHT_DIFFUSE = np.array([255, 255, 255]) / 255.0
LIGHT_SPECULAR = np.array([255, 255, 255]) / 255.0

MIN_CAMERA_DISTANCE = SPHERE_RADIUS + 1.0
MIN_LIGHT_DISTANCE = SPHERE_RADIUS * 1.25


@dataclass
class SceneParams:
    model_translation: list[float]
    material_diffuse: list[int]
    material_shininess: float
    light_position: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def look_at(eye, target, up) -> np.ndarray:
    """World -> camera (OpenGL convention, camera looks along -z)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


VIEW = look_at(CAMERA_POSITION, CAMERA_TARGET, CAMERA_UP)
TAN_HALF_FOV = np.tan(np.deg2rad(FOV_Y_DEG) / 2)


def to_camera(p_world: np.ndarray) -> np.ndarray:
    """Point(s) [..., 3] world -> camera space."""
    return p_world @ VIEW[:3, :3].T + VIEW[:3, 3]


def dir_to_camera(d_world: np.ndarray) -> np.ndarray:
    return d_world @ VIEW[:3, :3].T


def project_ndc(p_cam: np.ndarray) -> np.ndarray:
    """Camera-space point(s) -> NDC xy (aspect 1). y is up."""
    depth = -p_cam[..., 2]
    return p_cam[..., :2] / (depth[..., None] * TAN_HALF_FOV)


def pixel_rays(size: int = IMAGE_SIZE) -> np.ndarray:
    """Unit ray directions (camera space) through pixel centres, [H, W, 3].

    Row 0 is the top of the saved image (the renderer flips the framebuffer).
    """
    c = (np.arange(size) + 0.5) / size * 2 - 1
    x = c[None, :].repeat(size, 0)
    y = -c[:, None].repeat(size, 1)
    d = np.stack([x * TAN_HALF_FOV, y * TAN_HALF_FOV, -np.ones_like(x)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def is_valid(obj: np.ndarray, light: np.ndarray) -> tuple[bool, str]:
    """Reject degenerate scenes allowed by the raw U[-20,20] ranges."""
    if np.linalg.norm(obj - CAMERA_POSITION) < MIN_CAMERA_DISTANCE:
        return False, "camera_inside"
    if np.linalg.norm(light - obj) < MIN_LIGHT_DISTANCE:
        return False, "light_inside"
    c = to_camera(obj)
    if -c[2] < SPHERE_RADIUS + NEAR:
        return False, "behind_camera"
    ndc = project_ndc(c)
    if np.abs(ndc).max() > 1.0:
        return False, "out_of_view"
    return True, "ok"


def sample_scene(rng: np.random.Generator, constrained: bool = True,
                 stats: dict | None = None) -> SceneParams:
    while True:
        obj = rng.uniform(-POS_RANGE, POS_RANGE, 3)
        light = rng.uniform(-POS_RANGE, POS_RANGE, 3)
        ok, reason = is_valid(obj, light)
        if stats is not None:
            stats[reason] = stats.get(reason, 0) + 1
        if ok or not constrained:
            break
    return SceneParams(
        model_translation=[float(v) for v in obj],
        material_diffuse=[int(v) for v in rng.integers(0, 256, 3)],
        material_shininess=float(rng.uniform(*SHININESS_RANGE)),
        light_position=[float(v) for v in light],
    )


def phong_reference(p: SceneParams, size: int = IMAGE_SIZE, specular: bool = True) -> np.ndarray:
    """Ray-traced Phong of an ideal sphere (numpy). Used only for sanity checks
    of the camera conventions, the dataset itself comes from the OpenGL renderer."""
    obj = np.asarray(p.model_translation)
    c = to_camera(obj)
    lc = to_camera(np.asarray(p.light_position))
    d = pixel_rays(size)
    b = d @ c
    disc = b ** 2 - (c @ c - SPHERE_RADIUS ** 2)
    hit = disc > 0
    t = b - np.sqrt(np.maximum(disc, 0))
    x = d * t[..., None]
    n = (x - c) / SPHERE_RADIUS
    lv = lc - x
    lv /= np.linalg.norm(lv, axis=-1, keepdims=True)
    v = -x / np.linalg.norm(x, axis=-1, keepdims=True)
    ndl = np.clip((n * lv).sum(-1), 0, 1)
    r = 2 * (n * lv).sum(-1, keepdims=True) * n - lv
    rv = np.clip((r * v).sum(-1), 0, 1)
    kd = np.asarray(p.material_diffuse) / 255.0
    col = (LIGHT_AMBIENT * MATERIAL_AMBIENT + kd * ndl[..., None]
           + specular * (rv ** p.material_shininess)[..., None])
    col = np.clip(col, 0, 1)
    return np.where(hit[..., None], col, 0.0)
