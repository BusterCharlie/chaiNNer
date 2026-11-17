from __future__ import annotations

import math

import pytest

# The plain backend test job doesn't install the image dependencies.
pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("chainner_ext")

import cv2
import numpy as np

from nodes.impl.perspective import (
    _axis_lattice,
    _lattice_strength,
    apply_perspective,
    output_size,
    plan_corners,
    plan_lines,
    plan_pattern,
)
from nodes.impl.seamless import MapType


def panels(h: int, w: int, period: int, seed: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    groove = ((xx % period) < 3) | ((yy % period) < 3)
    rivet = np.hypot((xx % period) - 12, (yy % period) - 12) < 3
    base = np.where(groove, 0.2, 0.55).astype(np.float32)
    base[rivet] = 0.8
    noise = np.random.default_rng(seed).normal(0, 0.02, (h, w)).astype(np.float32)
    return np.repeat(np.clip(base + noise, 0, 1)[:, :, np.newaxis], 3, axis=2)


def tilt(img: np.ndarray, g: float, h_: float) -> np.ndarray:
    """Applies a known perspective (in coordinates centred on the image)."""
    h, w = img.shape[:2]
    s = max(w, h) / 2
    n = np.array([[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]])
    persp = np.array([[1, 0, 0], [0, 1, 0], [g, h_, 1.0]])
    pixel = np.linalg.inv(n) @ persp @ n
    return cv2.warpPerspective(
        img, pixel, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )


def centre(img: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    return img[(h - ch) // 2 : (h + ch) // 2, (w - cw) // 2 : (w + cw) // 2]


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def segment_angles(img: np.ndarray) -> np.ndarray:
    """Deviation (degrees) of long line segments from the nearest axis."""
    g8 = (np.clip(gray(img), 0, 1) * 255).astype(np.uint8)
    segs = cv2.createLineSegmentDetector().detect(g8)[0].reshape(-1, 4)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[length > 0.1 * max(img.shape[:2])]
    angles = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    angles = angles % 90
    return np.minimum(angles, 90 - angles)


@pytest.mark.parametrize("params", [(0.25, 0.0), (0.0, -0.3), (0.2, 0.15)])
def test_pattern_mode_recovers_tilt(params: tuple[float, float]):
    flat = panels(384, 384, 48)
    tilted = tilt(flat, *params)
    plan = plan_pattern(tilted)
    out = apply_perspective(plan, tilted, MapType.COLOR)
    reference = _lattice_strength(gray(centre(flat)))
    before = _lattice_strength(gray(centre(tilted)))
    after = _lattice_strength(gray(centre(out)))
    assert after > before
    assert after > 0.85 * reference, (before, after, reference)


@pytest.mark.parametrize("params", [(0.0, 0.3), (0.25, 0.0), (0.15, -0.2)])
def test_lines_mode_straightens_lines(params: tuple[float, float]):
    tilted = tilt(panels(480, 480, 60), *params)
    assert np.percentile(segment_angles(tilted), 90) > 1.0
    plan = plan_lines(tilted)
    out = apply_perspective(plan, tilted, MapType.COLOR)
    deviation = segment_angles(out)
    assert np.percentile(deviation, 90) < 1.0, np.percentile(deviation, 90)


def test_corners_mode_recovers_rectangle():
    flat = panels(300, 400, 50)
    # a known quadrilateral: where the corners of `flat` end up after a perspective warp
    src = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype=np.float32)
    dst = np.array([[60, 40], [350, 20], [390, 290], [30, 260]], dtype=np.float32)
    warp = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(flat, warp, (420, 320), flags=cv2.INTER_CUBIC)
    corners = [(float(x) / 420, float(y) / 320) for x, y in dst.tolist()]
    plan = plan_corners(corners, 420, 320)
    out = apply_perspective(plan, photo, MapType.COLOR)
    restored = cv2.resize(out, (400, 300), interpolation=cv2.INTER_AREA)
    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(restored[inner] - flat[inner]).mean() < 0.05


def test_same_plan_at_half_resolution():
    tilted = tilt(panels(400, 400, 50), 0.2, 0.1)
    plan = plan_lines(tilted)
    full = apply_perspective(plan, tilted, MapType.COLOR)
    half = apply_perspective(
        plan,
        cv2.resize(tilted, (200, 200), interpolation=cv2.INTER_AREA),
        MapType.COLOR,
    )
    fw, fh = output_size(plan, 400)
    hw, hh = output_size(plan, 200)
    assert abs(fw / 2 - hw) <= 1 and abs(fh / 2 - hh) <= 1
    down = cv2.resize(
        full, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_AREA
    )
    assert np.abs(down - half).mean() < 0.03


def normal_from_height(height: np.ndarray, map_type: MapType) -> np.ndarray:
    dy, dx = np.gradient(height * 8)
    x, y = -dx, dy
    if map_type == MapType.NORMAL_DX:
        y = -y
    n = np.dstack([np.ones_like(x), y, x])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


@pytest.mark.parametrize("map_type", [MapType.NORMAL_GL, MapType.NORMAL_DX], ids=str)
def test_normal_maps_are_rotated_with_the_warp(map_type: MapType):
    """Warping a normal map must match the normals of the warped height map."""
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    height = np.exp(-(((xx - 110) / 18) ** 2) - ((yy - 80) / 9) ** 2)
    # corners of a rectangle rotated by 20 degrees -> the correction is a rotation
    a = math.radians(20)
    c, s = math.cos(a), math.sin(a)
    pts = [(-60, -60), (60, -60), (60, 60), (-60, 60)]
    corners = [
        ((100 + c * x - s * y) / 200, (100 + s * x + c * y) / 200) for x, y in pts
    ]
    plan = plan_corners(corners, 200, 200)

    def warp_height(img: np.ndarray) -> np.ndarray:
        return apply_perspective(plan, img[:, :, np.newaxis], MapType.COLOR)[:, :, 0]

    expected = normal_from_height(warp_height(height), map_type)
    actual = apply_perspective(plan, normal_from_height(height, map_type), map_type)
    inner = (slice(15, -15), slice(15, -15))
    error = np.abs(expected[inner][:, :, 1:] - actual[inner][:, :, 1:]).mean()
    unrotated = apply_perspective(
        plan, normal_from_height(height, map_type), MapType.COLOR
    )
    wrong = np.abs(expected[inner][:, :, 1:] - unrotated[inner][:, :, 1:]).mean()
    assert error < 0.01
    assert wrong > 3 * error


def test_masks_stay_binary():
    tilted = tilt(panels(300, 300, 50), 0.2, 0.0)
    mask = (gray(tilted) > 0.4).astype(np.float32)
    out = apply_perspective(plan_lines(tilted), mask, MapType.MASK)
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


def test_impossible_corners_raise():
    with pytest.raises(ValueError):
        plan_corners([(0.5, 0.5)] * 4, 100, 100)


@pytest.mark.parametrize(("rotation", "shear"), [(4.0, 0.0), (0.0, 3.0), (-3.0, 2.5)])
def test_pattern_mode_aligns_lattice_to_axes(rotation: float, shear: float):
    flat = panels(400, 400, 40)
    a, k = math.radians(rotation), math.tan(math.radians(shear))
    affine = np.array(
        [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    ) @ np.array([[1, k], [0, 1]])
    matrix = np.c_[affine, np.array([200, 200]) - affine @ np.array([200, 200])]
    skewed = cv2.warpAffine(
        flat, matrix, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )

    def off_axis(img: np.ndarray) -> float:
        lattice = _axis_lattice(gray(centre(img, 0.6)))
        assert lattice is not None
        return float(max(abs(lattice[1, 0]), abs(lattice[0, 1])))

    assert off_axis(skewed) > 1.0
    out = apply_perspective(plan_pattern(skewed, deskew=True), skewed, MapType.COLOR)
    assert off_axis(out) < 0.5, off_axis(out)


def test_corners_without_crop_refuse_huge_outputs():
    # a strongly converging quad: its horizon lies inside the image, so the uncropped warp
    # would be infinitely large
    corners = [(0.49, 0.4), (0.51, 0.4), (0.9, 0.6), (0.1, 0.6)]
    with pytest.raises(ValueError):
        plan_corners(corners, 400, 400, crop=False)
    # cropped to the rectangle it's fine
    plan = plan_corners(corners, 400, 400, crop=True)
    assert max(plan.out_size) < 400
