from __future__ import annotations

import pytest

# The plain backend test job doesn't install the image dependencies.
pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("chainner_ext")

import numpy as np

from nodes.impl.curvature import CurvatureSource, curvature_maps


def bump(size: int = 128, cx: float = 64, cy: float = 64, r: float = 14) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)).astype(np.float32)


def normal_from_height(height: np.ndarray, directx: bool = False) -> np.ndarray:
    dy, dx = np.gradient(height * 8)
    x, y = -dx, dy  # rows grow downwards: "up" is -row
    if directx:
        y = -y
    n = np.dstack([np.ones_like(x), y, x])  # B=z, G=y, R=x
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


@pytest.mark.parametrize(
    ("source", "make"),
    [
        (CurvatureSource.HEIGHT, lambda h: h),
        (CurvatureSource.NORMAL_GL, lambda h: normal_from_height(h)),
        (CurvatureSource.NORMAL_DX, lambda h: normal_from_height(h, directx=True)),
    ],
    ids=["height", "normal_gl", "normal_dx"],
)
def test_bump_is_convex_and_pit_is_concave(source, make):
    for sign, expect_convex in ((1, True), (-1, False)):
        height = 0.5 + 0.4 * sign * bump()
        curv, convex, concave = curvature_maps(
            make(height), source, 8, 1.0, True, False
        )
        centre = curv[60:68, 60:68].mean()
        if expect_convex:
            assert centre > 0.7, centre
            assert convex[64, 64] > 0.3 and concave[64, 64] == 0
        else:
            assert centre < 0.3, centre
            assert concave[64, 64] > 0.3 and convex[64, 64] == 0


def test_flat_is_mid_gray_and_masks_are_empty():
    flat = np.full((64, 64, 3), [1.0, 0.5, 0.5], np.float32)  # straight-up normal
    curv, convex, concave = curvature_maps(
        flat, CurvatureSource.NORMAL_GL, 16, 1.0, True, True
    )
    assert np.allclose(curv, 0.5)
    assert convex.max() == 0 and concave.max() == 0


def test_normal_and_height_sources_agree():
    rng = np.random.default_rng(0)
    height = np.zeros((128, 128), np.float32)
    for _ in range(12):
        cx, cy = (float(v) for v in rng.uniform(16, 112, 2))
        height += rng.uniform(-1, 1) * bump(cx=cx, cy=cy, r=rng.uniform(5, 12))
    from_height, _, _ = curvature_maps(
        height, CurvatureSource.HEIGHT, 8, 1.0, True, False
    )
    from_normal, _, _ = curvature_maps(
        normal_from_height(height), CurvatureSource.NORMAL_GL, 8, 1.0, True, False
    )
    corr = np.corrcoef(from_height.ravel(), from_normal.ravel())[0, 1]
    assert corr > 0.8, corr


def test_tileable_has_no_seam():
    # a seamless input: a bump centred on the corner, using wrapped distances
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = np.minimum(xx, size - xx)
    dy = np.minimum(yy, size - yy)
    height = (0.5 + 0.4 * np.exp(-(dx * dx + dy * dy) / (2 * 10 * 10))).astype(
        np.float32
    )
    curv, _, _ = curvature_maps(height, CurvatureSource.HEIGHT, 8, 1.0, False, True)
    jump = np.abs(curv[:, 0] - curv[:, -1]).mean()
    step = np.abs(np.diff(curv, axis=1)).mean()
    assert jump < 3 * step
    # without tileable, the corner bump is cut off at the edge
    cut, _, _ = curvature_maps(height, CurvatureSource.HEIGHT, 8, 1.0, False, False)
    assert np.abs(cut[:, 0] - cut[:, -1]).mean() > jump


def test_strength_and_auto_level():
    height = 0.5 + 0.2 * bump()
    a, _, _ = curvature_maps(height, CurvatureSource.HEIGHT, 8, 1.0, True, False)
    b, _, _ = curvature_maps(height, CurvatureSource.HEIGHT, 8, 0.5, True, False)
    assert abs(a.max() - 0.5) > abs(b.max() - 0.5)
    # without auto level, the same shape at half the height gives half the curvature
    full, _, _ = curvature_maps(height, CurvatureSource.HEIGHT, 8, 1.0, False, False)
    half, _, _ = curvature_maps(
        0.5 + 0.1 * bump(), CurvatureSource.HEIGHT, 8, 1.0, False, False
    )
    assert abs((half.max() - 0.5) / (full.max() - 0.5) - 0.5) < 0.05


def test_normal_source_needs_rgb():
    with pytest.raises(ValueError):
        curvature_maps(bump(), CurvatureSource.NORMAL_GL, 8, 1.0, True, False)
