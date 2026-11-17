"""
A small software renderer for previewing PBR materials (metallic/roughness workflow).

It draws the material on a sphere or a flat tile, lit by one point light (adjustable direction)
and a simple studio environment, with GGX specular, Schlick Fresnel and Smith geometry. It's
meant for judging maps by eye, not for physical accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ..utils.utils import get_h_w_c


class PreviewShape(Enum):
    SPHERE = "sphere"
    FLAT = "flat"


@dataclass(frozen=True)
class PreviewSettings:
    shape: PreviewShape = PreviewShape.SPHERE
    size: int = 512
    tiling: float = 2.0
    light_azimuth: float = 135.0
    """Degrees, counter-clockwise from the right (90 = from the top)."""
    light_elevation: float = 45.0
    """Degrees above the surface plane of a flat tile / the image plane of the sphere."""
    light_intensity: float = 1.0
    normal_directx: bool = False


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _rgb(img: np.ndarray) -> np.ndarray:
    """Any image -> float RGB (3 channels)."""
    c = get_h_w_c(img)[2]
    if c == 1:
        g = img.reshape(img.shape[:2])
        return np.dstack([g, g, g]).astype(np.float32)
    return img[:, :, 2::-1].astype(np.float32) if c >= 3 else img


def _gray(img: np.ndarray) -> np.ndarray:
    c = get_h_w_c(img)[2]
    if c == 1:
        return img.reshape(img.shape[:2]).astype(np.float32)
    return cv2.cvtColor(img[:, :, :3].astype(np.float32), cv2.COLOR_BGR2GRAY)


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear, wrap-around sampling at texture coordinates (u right, v down, 1 = one tile)."""
    h, w = img.shape[:2]
    mx = ((u % 1) * w - 0.5).astype(np.float32)
    my = ((v % 1) * h - 0.5).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _environment(direction: np.ndarray) -> np.ndarray:
    """A simple studio environment: bright sky, darker ground, soft horizon (RGB)."""
    y = direction[..., 1:2]
    sky = np.array([0.55, 0.6, 0.7], np.float32)
    ground = np.array([0.12, 0.11, 0.1], np.float32)
    t = np.clip(y * 2 + 0.5, 0, 1)
    return ground * (1 - t) + sky * t


def _geometry(settings: PreviewSettings):
    """Per-pixel position, geometric normal, tangent frame, texture coordinates and mask."""
    n = settings.size
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    px = (xx + 0.5) / n * 2 - 1
    py = 1 - (yy + 0.5) / n * 2

    if settings.shape == PreviewShape.SPHERE:
        r2 = px * px + py * py
        mask = r2 < 1
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        normal = np.dstack([px, py, z])
        position = normal.copy()
        # longitude/latitude texture coordinates
        u = (np.arctan2(px, z) / math.pi * 0.5 + 0.5) * settings.tiling
        v = (0.5 - np.arcsin(np.clip(py, -1, 1)) / math.pi) * settings.tiling * 0.5
        tangent = _normalize(np.dstack([z, np.zeros_like(z), -px]))
        bitangent = np.cross(normal, tangent)  # points up (towards +y)
        return position, normal, tangent, bitangent, u, v, mask

    # flat tile, seen from a camera above at a slight angle, filling the frame
    mask = np.ones((n, n), bool)
    position = np.dstack([px, py, np.zeros_like(px)])
    normal = np.dstack([np.zeros_like(px), np.zeros_like(px), np.ones_like(px)])
    tangent = np.dstack([np.ones_like(px), np.zeros_like(px), np.zeros_like(px)])
    bitangent = np.dstack([np.zeros_like(px), np.ones_like(px), np.zeros_like(px)])
    u = (px * 0.5 + 0.5) * settings.tiling
    v = (0.5 - py * 0.5) * settings.tiling
    return position, normal, tangent, bitangent, u, v, mask


def render_preview(
    albedo: np.ndarray,
    roughness: np.ndarray,
    normal: np.ndarray | None = None,
    metallic: np.ndarray | None = None,
    settings: PreviewSettings = PreviewSettings(),  # noqa: B008
) -> np.ndarray:
    """Returns a BGR float32 image of size `settings.size`."""
    position, geo_normal, tangent, bitangent, u, v, mask = _geometry(settings)

    base = _srgb_to_linear(np.clip(_sample(_rgb(albedo), u, v), 0, 1))
    rough = np.clip(_sample(_gray(roughness), u, v), 0.03, 1)[..., None]
    metal = (
        np.clip(_sample(_gray(metallic), u, v), 0, 1)[..., None]
        if metallic is not None
        else np.zeros_like(rough)
    )

    n = geo_normal
    if normal is not None:
        tn = _sample(normal.astype(np.float32), u, v)
        x = tn[:, :, 2] * 2 - 1
        y = tn[:, :, 1] * 2 - 1
        if settings.normal_directx:
            y = -y
        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
        n = _normalize(
            tangent * x[..., None]
            + bitangent * y[..., None]
            + geo_normal * z[..., None]
        )

    # camera and light
    if settings.shape == PreviewShape.SPHERE:
        view = np.broadcast_to(np.array([0, 0, 1], np.float32), n.shape)
        az, el = (
            math.radians(settings.light_azimuth),
            math.radians(settings.light_elevation),
        )
        to_light = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        light_dir = np.broadcast_to(to_light.astype(np.float32), n.shape)
        falloff = 1.0
    else:
        camera = np.array([0, -0.6, 2.6], np.float32)
        view = _normalize(camera - position)
        az, el = (
            math.radians(settings.light_azimuth),
            math.radians(settings.light_elevation),
        )
        light_pos = 2.2 * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        offset = light_pos - position
        dist2 = np.sum(offset * offset, axis=-1, keepdims=True)
        light_dir = _normalize(offset)
        falloff = 4.0 / (dist2 + 1e-3)

    half = _normalize(light_dir + view)
    ndl = np.clip(np.sum(n * light_dir, -1, keepdims=True), 0, 1)
    ndv = np.clip(np.sum(n * view, -1, keepdims=True), 1e-3, 1)
    ndh = np.clip(np.sum(n * half, -1, keepdims=True), 0, 1)
    vdh = np.clip(np.sum(view * half, -1, keepdims=True), 0, 1)

    alpha = rough * rough
    a2 = alpha * alpha
    distribution = a2 / (math.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
    k = (rough + 1) ** 2 / 8
    geometry = (ndl / (ndl * (1 - k) + k + 1e-6)) * (ndv / (ndv * (1 - k) + k))
    f0 = 0.04 * (1 - metal) + base * metal
    fresnel = f0 + (1 - f0) * (1 - vdh) ** 5
    specular = distribution * geometry * fresnel / (4 * ndl * ndv + 1e-4)
    diffuse = (1 - fresnel) * (1 - metal) * base / math.pi
    direct = (diffuse + specular) * ndl * falloff * settings.light_intensity * 3.0

    # environment: diffuse sky light and a roughness-blurred reflection
    reflected = _normalize(2 * ndv * n - view)
    env_spec = _environment(reflected) * (1 - rough) + _environment(n) * rough
    f_env = f0 + (np.maximum(1 - rough, f0) - f0) * (1 - ndv) ** 5
    ambient = base * (1 - metal) * _environment(n) * 0.35 + f_env * env_spec * 0.6

    color = direct + ambient
    color = color / (1 + color)  # tone map
    out = _linear_to_srgb(color)
    background = np.full_like(out, 0.18)
    out = np.where(mask[..., None], out, background)
    return np.ascontiguousarray(out[:, :, ::-1]).astype(np.float32)
