"""
Curvature maps from normal maps or height maps.

Curvature is positive on convex areas (edges, ridges) and negative on concave areas (cavities,
gaps). It is accumulated over several blur scales, from 1 pixel up to a radius, so both fine
detail and broader shapes contribute (similar to Substance's "Curvature Smooth").
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from ..utils.utils import get_h_w_c

# Raw curvature is divided by these so that 100% strength without auto level gives about the
# same contrast as auto level on a typical texture (measured on 40 Poly Haven materials).
NORMAL_GAIN = 9.0
HEIGHT_GAIN = 30.0


class CurvatureSource(Enum):
    NORMAL_GL = "normal_gl"
    NORMAL_DX = "normal_dx"
    HEIGHT = "height"


def _scales(radius: float) -> list[float]:
    """1, 2, 4, ... up to the radius (always including the radius itself)."""
    scales: list[float] = []
    s = 1.0
    while s < radius:
        scales.append(s)
        s *= 2
    scales.append(max(radius, 0.5))
    return scales


def _blur(x: np.ndarray, sigma: float, tileable: bool) -> np.ndarray:
    p = max(1, int(np.ceil(3 * sigma)))
    mode = "wrap" if tileable else "reflect"
    padded = np.pad(x, p, mode=mode)
    return cv2.GaussianBlur(padded, (0, 0), sigma)[p:-p, p:-p]


def _shift(x: np.ndarray, dy: int, dx: int, tileable: bool) -> np.ndarray:
    """x[y + dy, x + dx], with wrap-around or edge clamping."""
    if tileable:
        return np.roll(x, (-dy, -dx), axis=(0, 1))
    padded = np.pad(x, 1, mode="edge")
    h, w = x.shape
    return padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]


def raw_curvature(
    img: np.ndarray, source: CurvatureSource, radius: float, tileable: bool
) -> np.ndarray:
    """Signed curvature, roughly in [-1, 1] after the source's gain (convex > 0)."""
    _, _, c = get_h_w_c(img)
    scales = _scales(radius)
    total = None

    if source == CurvatureSource.HEIGHT:
        h = img if c == 1 else img[:, :, :3].mean(axis=2)
        h = h.reshape(h.shape[:2]).astype(np.float32)
        total = np.zeros_like(h)
        for s in scales:
            b = _blur(h, s, tileable)
            laplacian = (
                _shift(b, -1, 0, tileable)
                + _shift(b, 1, 0, tileable)
                + _shift(b, 0, -1, tileable)
                + _shift(b, 0, 1, tileable)
                - 4 * b
            )
            # the Laplacian of a blurred image shrinks with 1/s^2; weight it back up
            total += -laplacian * s * s
        return total / len(scales) * HEIGHT_GAIN

    if c < 3:
        raise ValueError("A normal map needs at least 3 channels (RGB).")
    x = img[:, :, 2].astype(np.float32) * 2 - 1
    y = img[:, :, 1].astype(np.float32) * 2 - 1
    if source == CurvatureSource.NORMAL_DX:
        y = -y
    y_down = -y  # image rows point down, OpenGL green points up
    total = np.zeros_like(x)
    for s in scales:
        bx = _blur(x, s, tileable)
        by = _blur(y_down, s, tileable)
        divergence = (_shift(bx, 0, 1, tileable) - _shift(bx, 0, -1, tileable)) / 2 + (
            _shift(by, 1, 0, tileable) - _shift(by, -1, 0, tileable)
        ) / 2
        # slopes of a blurred normal field shrink with 1/s; weight them back up
        total += divergence * s
    return total / len(scales) * NORMAL_GAIN


def curvature_maps(
    img: np.ndarray,
    source: CurvatureSource,
    radius: float,
    strength: float,
    auto_level: bool,
    tileable: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (curvature, convex, concave):
    curvature is 0.5 on flat areas, brighter on convex and darker on concave areas;
    convex and concave are masks (0 = none) of the two halves.
    """
    curv = raw_curvature(img, source, radius, tileable)
    if auto_level:
        scale = float(np.percentile(np.abs(curv), 99))
        if scale > 1e-8:
            curv = curv / scale
    curv = np.clip(curv * strength, -1, 1).astype(np.float32)
    return (
        (curv * 0.5 + 0.5).astype(np.float32),
        np.maximum(curv, 0).astype(np.float32),
        np.maximum(-curv, 0).astype(np.float32),
    )
