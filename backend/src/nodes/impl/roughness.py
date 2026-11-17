"""
Synthetic roughness maps from an albedo, with optional curvature, ambient occlusion, metal mask
and color picks.

There is no way to measure roughness from a single diffuse photo, so this is an artistic tool:
 1. A base map from the albedo's brightness, remapped to a chosen average and contrast. Whether
    darker or brighter areas are rougher depends on the material, so it's a setting.
 2. Optional layers that make edges, cavities, occluded areas or busy fine detail rougher
    (positive weight) or smoother (negative weight).
 3. Color picks that give regions of a chosen color their own roughness (e.g. rust on paint),
    and a metal mask with its own roughness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..utils.utils import get_h_w_c
from .image_utils import fast_gaussian_blur


class BrightnessReference(Enum):
    GLOBAL = "global"
    LOCAL = "local"


@dataclass(frozen=True)
class ColorPick:
    color: tuple[float, float, float]
    """BGR in [0, 1]."""
    roughness: float
    tolerance: float = 0.12
    """Color distance (in Lab, as a fraction of 100) that is fully selected."""
    softness: float = 0.05
    """Width of the fade-out beyond the tolerance."""


@dataclass(frozen=True)
class RoughnessParams:
    average: float = 0.6
    contrast: float = 0.1
    """Spread (standard deviation) of the base map."""
    brighter_is_rougher: bool = False
    reference: BrightnessReference = BrightnessReference.GLOBAL
    local_radius: float = 0.05
    """For the local reference: blur radius as a fraction of the image size."""
    edges: float = 0.0
    cavities: float = 0.0
    occlusion: float = 0.0
    fine_detail: float = 0.0
    metal_roughness: float = 0.3
    picks: tuple[ColorPick, ...] = field(default_factory=tuple)


def _gray(img: np.ndarray) -> np.ndarray:
    c = get_h_w_c(img)[2]
    if c == 1:
        return img.reshape(img.shape[:2]).astype(np.float32)
    return cv2.cvtColor(img[:, :, :3].astype(np.float32), cv2.COLOR_BGR2GRAY)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    if sigma > 8:
        return fast_gaussian_blur(img, sigma).reshape(img.shape)
    return cv2.GaussianBlur(img, (0, 0), sigma).reshape(img.shape)


def _resized(mask: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    if mask is None:
        return None
    m = _gray(mask)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0, 1)


def color_mask(
    albedo: np.ndarray, pick: ColorPick, blur_sigma: float = 1.0
) -> np.ndarray:
    """Soft mask (0..1) of the pixels close to the picked color, measured in Lab."""
    a = albedo
    if get_h_w_c(a)[2] == 1:
        a = np.dstack([a.reshape(a.shape[:2])] * 3)
    a = _blur(np.clip(a[:, :, :3], 0, 1).astype(np.float32), blur_sigma)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    ref = cv2.cvtColor(np.array([[pick.color]], np.float32), cv2.COLOR_BGR2LAB)[0, 0]
    distance = np.linalg.norm(lab - ref, axis=2) / 100
    t = np.clip(
        (pick.tolerance + pick.softness - distance) / max(pick.softness, 1e-6), 0, 1
    )
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _fine_detail(lum: np.ndarray) -> np.ndarray:
    """Local strength of fine detail (local std of a high-pass), scaled to about 0..1."""
    high = lum - _blur(lum, 2.0)
    energy = np.sqrt(np.maximum(_blur(high * high, 6.0), 0))
    scale = float(np.percentile(energy, 99))
    return np.clip(energy / scale, 0, 1) if scale > 1e-6 else np.zeros_like(energy)


def generate_roughness(
    albedo: np.ndarray,
    params: RoughnessParams,
    curvature: np.ndarray | None = None,
    ao: np.ndarray | None = None,
    metal: np.ndarray | None = None,
) -> np.ndarray:
    """
    `curvature` follows the usual convention: 0.5 = flat, brighter = convex, darker = concave.
    `ao`: 1 = open, 0 = fully occluded. `metal`: 1 = metal. All optional maps are resized to
    the albedo's size. Returns a single-channel roughness map in [0, 1].
    """
    h, w, _ = get_h_w_c(albedo)
    lum = _gray(albedo)

    # 1. base map from brightness, remapped to the chosen average and contrast
    if params.reference == BrightnessReference.LOCAL:
        cue = lum - _blur(lum, params.local_radius * max(h, w))
    else:
        cue = lum
    if not params.brighter_is_rougher:
        cue = -cue
    std = float(cue.std())
    z = (cue - cue.mean()) / std if std > 1e-6 else np.zeros_like(cue)
    rough = params.average + params.contrast * z

    # 2. layers
    curv = _resized(curvature, h, w)
    if curv is not None:
        if params.edges:
            rough = rough + params.edges * np.clip(curv * 2 - 1, 0, 1)
        if params.cavities:
            rough = rough + params.cavities * np.clip(1 - curv * 2, 0, 1)
    occ = _resized(ao, h, w)
    if occ is not None and params.occlusion:
        rough = rough + params.occlusion * (1 - occ)
    if params.fine_detail:
        rough = rough + params.fine_detail * _fine_detail(lum)

    # 3. regions with their own roughness
    for pick in params.picks:
        m = color_mask(albedo, pick)
        rough = rough * (1 - m) + pick.roughness * m
    met = _resized(metal, h, w)
    if met is not None:
        rough = rough * (1 - met) + params.metal_roughness * met

    return np.clip(rough, 0, 1).astype(np.float32)
