from __future__ import annotations

import pytest

# The plain backend test job doesn't install the image dependencies.
pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("chainner_ext")

import cv2
import numpy as np

from nodes.impl.seamless import (
    MapType,
    QuincunxParams,
    SeamlessMethod,
    SplatParams,
    apply_plan,
    create_plan,
    equalize_lighting,
    rotate_normal_vectors,
    seam_ratio,
)

ALL_METHODS = list(SeamlessMethod)
SPLATS_WITH_ROTATION = SplatParams(count=16, rotation_variation=60)


def smooth_random(h: int, w: int, c: int, seed: int) -> np.ndarray:
    """A smooth, clearly non-tileable test image with a lighting gradient."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(0, 1, (8, 8, c)).astype(np.float32)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(h, w, c)
    gradient = np.linspace(0.6, 1.0, w, dtype=np.float32)[np.newaxis, :, np.newaxis]
    return np.clip(img * gradient, 0, 1).astype(np.float32)


def normal_from_height(height: np.ndarray, map_type: MapType) -> np.ndarray:
    """BGR normal map from a height map. OpenGL: +Y (green) points up the image."""
    dy, dx = np.gradient(height * 8)
    x, y = -dx, dy  # rows grow downwards, so "up" is -row
    if map_type == MapType.NORMAL_DX:
        y = -y
    n = np.dstack([np.ones_like(x), y, x])  # B=z, G=y, R=x
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def run(img, method, map_type=MapType.COLOR, guide=None, seed=1, **kwargs):
    plan = create_plan(img if guide is None else guide, method, seed, **kwargs)
    return apply_plan(plan, img, map_type)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_output_tiles(method: SeamlessMethod):
    img = smooth_random(128, 96, 3, seed=0)
    assert seam_ratio(img) > 5

    out = run(img, method)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1

    if method == SeamlessMethod.EDGES_LINEAR_X:
        jump = np.abs(out[:, 0] - out[:, -1]).mean()
        inner = np.abs(np.diff(out, axis=1)).mean()
        assert jump / inner < 1.6
    elif method == SeamlessMethod.EDGES_LINEAR_Y:
        jump = np.abs(out[0] - out[-1]).mean()
        inner = np.abs(np.diff(out, axis=0)).mean()
        assert jump / inner < 1.6
    else:
        ratio = seam_ratio(out)
        assert ratio < 1.6, f"{method}: seam ratio {ratio}"


@pytest.mark.parametrize("method", ALL_METHODS)
def test_deterministic(method: SeamlessMethod):
    img = smooth_random(64, 64, 3, seed=1)
    a = run(img, method, seed=5)
    b = run(img, method, seed=5)
    np.testing.assert_array_equal(a, b)


def test_seed_changes_result():
    img = smooth_random(64, 64, 3, seed=1)
    a = run(img, SeamlessMethod.SPLAT_RANDOM, seed=1)
    b = run(img, SeamlessMethod.SPLAT_RANDOM, seed=2)
    assert np.abs(a - b).mean() > 1e-3


@pytest.mark.parametrize(
    "method",
    [
        SeamlessMethod.EDGES_LINEAR,
        SeamlessMethod.EDGES_QUINCUNX,
        SeamlessMethod.MIN_ERROR_SEAM,
        SeamlessMethod.SPLAT_RANDOM,
    ],
)
def test_stack_sync_across_resolutions(method: SeamlessMethod):
    """The same plan on a half-resolution copy gives (nearly) the same result."""
    guide = smooth_random(128, 128, 1, seed=2)
    full = smooth_random(128, 128, 3, seed=3)
    half = cv2.resize(full, (64, 64), interpolation=cv2.INTER_AREA)

    plan = create_plan(guide, method, 7)
    out_full = apply_plan(plan, full, MapType.COLOR)
    out_half = apply_plan(plan, half, MapType.COLOR)

    downscaled = cv2.resize(out_full, (64, 64), interpolation=cv2.INTER_AREA)
    assert np.abs(downscaled - out_half).mean() < 0.02
    assert seam_ratio(out_half) < 1.6


@pytest.mark.parametrize("map_type", [MapType.NORMAL_GL, MapType.NORMAL_DX])
@pytest.mark.parametrize("method", ALL_METHODS)
def test_normals_stay_normalized(map_type: MapType, method: SeamlessMethod):
    height = smooth_random(96, 96, 1, seed=4)[:, :, 0]
    normal = normal_from_height(height, map_type)
    out = apply_plan(
        create_plan(height, method, 3, splat=SPLATS_WITH_ROTATION), normal, map_type
    )
    n = out[:, :, :3] * 2 - 1
    length = np.linalg.norm(n, axis=2)
    assert np.allclose(length, 1, atol=1e-3)


@pytest.mark.parametrize("map_type", [MapType.NORMAL_GL, MapType.NORMAL_DX])
@pytest.mark.parametrize("angle", [30.0, -75.0, 90.0])
def test_rotated_normals_match_rotated_height(map_type: MapType, angle: float):
    """Rotating a height map and deriving normals == rotating the normal map + its vectors."""
    yy, xx = np.mgrid[0:129, 0:129].astype(np.float32)
    height = np.exp(-(((xx - 80) / 14) ** 2) - ((yy - 50) / 9) ** 2)

    center = (64, 64)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)

    def rotate_pixels(img: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(img, rot, (129, 129), flags=cv2.INTER_LINEAR)

    expected = normal_from_height(rotate_pixels(height), map_type)
    actual = rotate_normal_vectors(
        rotate_pixels(normal_from_height(height, map_type)), angle, map_type
    )

    inner = (slice(20, -20), slice(20, -20))
    error = np.abs(expected[inner] - actual[inner]).mean()
    assert error < 0.01
    # Without rotating the vectors, the result is clearly worse.
    wrong = rotate_pixels(normal_from_height(height, map_type))
    assert np.abs(expected[inner] - wrong[inner]).mean() > 3 * error


@pytest.mark.parametrize("method", ALL_METHODS)
def test_masks_stay_binary(method: SeamlessMethod):
    guide = smooth_random(96, 96, 1, seed=5)
    mask = (guide > 0.5).astype(np.float32)
    out = apply_plan(
        create_plan(guide, method, 3, splat=SPLATS_WITH_ROTATION), mask, MapType.MASK
    )
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


def test_content_influence_keeps_tiling():
    guide = smooth_random(128, 128, 1, seed=6)
    img = smooth_random(128, 128, 3, seed=7)
    out = run(
        img,
        SeamlessMethod.EDGES_QUINCUNX,
        guide=guide,
        quincunx=QuincunxParams(content_influence=1.0),
    )
    assert seam_ratio(out) < 2.0


def test_alpha_and_grayscale_supported():
    rgba = smooth_random(64, 64, 4, seed=8)
    assert run(rgba, SeamlessMethod.SPLAT_RANDOM).shape == (64, 64, 4)
    gray = smooth_random(64, 64, 1, seed=9)[:, :, 0]
    assert run(gray, SeamlessMethod.EDGES_QUINCUNX).shape == (64, 64, 1)


def test_equalize_lighting_removes_gradient():
    flat = np.full((128, 128, 3), 0.5, np.float32)
    ramp = np.linspace(0.3, 1.0, 128, dtype=np.float32)[np.newaxis, :, np.newaxis]
    out = equalize_lighting(flat * ramp, 0.1)
    left = out[:, :32].mean()
    right = out[:, -32:].mean()
    assert abs(left - right) < 0.05
