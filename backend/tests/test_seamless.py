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
    crop_margin_fraction,
    equalize_lighting,
    height_from_normals,
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


@pytest.mark.parametrize(
    ("w", "h", "percent", "expected"),
    [
        (1000, 750, 7.5, (638, 850)),
        (2049, 1024, 12.3, (774, 1545)),
        (1, 1, 25, (1, 1)),
        (512, 512, 0, (512, 512)),
    ],
)
def test_crop_margin_size(w: int, h: int, percent: float, expected: tuple[int, int]):
    # These sizes must match the node's output type expression (checked with navi).
    img = np.zeros((h, w, 3), np.float32)
    assert crop_margin_fraction(img, percent).shape[:2] == expected


def test_crop_margin_same_area_across_resolutions():
    full = smooth_random(200, 200, 3, seed=10)
    half = cv2.resize(full, (100, 100), interpolation=cv2.INTER_AREA)
    a = crop_margin_fraction(full, 10)
    b = crop_margin_fraction(half, 10)
    assert a.shape[:2] == (160, 160)
    assert b.shape[:2] == (80, 80)
    downscaled = cv2.resize(a, (80, 80), interpolation=cv2.INTER_AREA)
    assert np.abs(downscaled - b).mean() < 0.01


def test_quincunx_tips_do_not_leak():
    # Smooth, low-detail images used to show a seam at the 4 diamond tips.
    ratios = []
    for seed in range(6):
        small = np.random.default_rng(seed).random((8, 8, 3)).astype(np.float32)
        img = cv2.resize(small, (400, 300), interpolation=cv2.INTER_CUBIC)
        out = run(img, SeamlessMethod.EDGES_QUINCUNX, seed=seed)
        ratios.append(seam_ratio(out))
    assert max(ratios) < 1.1, ratios


def test_min_error_seam_has_no_hard_step_on_smooth_images():
    # Only large-scale brightness, no detail: a sharp cut would show as a visible step.
    small = np.random.default_rng(3).random((4, 4, 3)).astype(np.float32)
    img = cv2.resize(small, (256, 256), interpolation=cv2.INTER_CUBIC)
    out = run(img, SeamlessMethod.MIN_ERROR_SEAM)
    max_step_in = max(np.abs(np.diff(img, axis=a)).max() for a in (0, 1))
    max_step_out = max(np.abs(np.diff(out, axis=a)).max() for a in (0, 1))
    assert max_step_out < 3 * max_step_in


@pytest.mark.parametrize("method", ALL_METHODS)
def test_normals_ignore_blue_channel(method: SeamlessMethod):
    """Two-channel (BC5-style) normal maps with an empty blue channel give the same result."""
    height = smooth_random(96, 96, 1, seed=11)[:, :, 0]
    normal = normal_from_height(height, MapType.NORMAL_GL)
    no_blue = normal.copy()
    no_blue[:, :, 0] = 0
    plan = create_plan(height, method, 3, splat=SPLATS_WITH_ROTATION)
    a = apply_plan(plan, normal, MapType.NORMAL_GL)
    b = apply_plan(plan, no_blue, MapType.NORMAL_GL)
    assert np.abs(a - b).max() < 0.02


def panels(h: int, w: int, period: int, seed: int) -> np.ndarray:
    """A strictly regular grid of panels with grooves and rivets, plus grain."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    groove = ((xx % period) < 3) | ((yy % period) < 3)
    rivet = np.hypot((xx % period) - 10, (yy % period) - 10) < 3
    base = np.where(groove, 0.2, 0.55).astype(np.float32)
    base[rivet] = 0.8
    noise = np.random.default_rng(seed).normal(0, 0.02, (h, w)).astype(np.float32)
    return np.repeat(np.clip(base + noise, 0, 1)[:, :, np.newaxis], 3, axis=2)


def test_pattern_align_shifts_by_whole_repeats():
    img = panels(300, 384, 70, seed=0)  # neither side is a multiple of 70
    plan = create_plan(img, SeamlessMethod.MIN_ERROR_SEAM, 1, pattern_align=True)
    # After fitting, the image holds a whole number of repeats and the shift is a whole number
    # of repeats too. In output pixels, one repeat is 70 / fit.
    for fraction, fit, size in (
        (plan.shift_x, plan.fit_x, 384),
        (plan.shift_y, plan.fit_y, 300),
    ):
        repeat = 70 / fit
        for length in (size, round(fraction * size)):
            off = length % repeat
            assert min(off, repeat - off) <= 1.5, (length, repeat)
    assert all("fitted to" in line for line in plan.info), plan.info

    aligned = apply_plan(plan, img, MapType.COLOR)
    plain = run(img, SeamlessMethod.MIN_ERROR_SEAM)

    # A correctly aligned result is still a regular pattern: it matches itself one repeat away.
    # Measured inside the image, above the noise floor of the (perfectly regular) input.
    def irregularity(out: np.ndarray, repeat: float) -> float:
        r = round(repeat)
        return float(np.abs(out[:, r:] - out[:, :-r]).mean())

    floor = irregularity(img, 70)
    excess_aligned = irregularity(aligned, 70 / plan.fit_x) - floor
    excess_plain = irregularity(plain, 70) - floor
    assert excess_aligned < 0.25 * excess_plain, (excess_aligned, excess_plain)
    assert seam_ratio(aligned) < 1.6


def test_pattern_align_keeps_half_shift_without_pattern():
    img = smooth_random(128, 160, 3, seed=12)
    plan = create_plan(img, SeamlessMethod.MIN_ERROR_SEAM, 1, pattern_align=True)
    assert plan.shift_x == 0.5
    assert plan.shift_y == 0.5
    assert not any("aligned to the pattern" in line for line in plan.info), plan.info


@pytest.mark.parametrize(
    "map_type", [MapType.COLOR, MapType.NORMAL_GL, MapType.MASK], ids=str
)
@pytest.mark.parametrize(
    "method",
    [SeamlessMethod.EDGES_LINEAR, SeamlessMethod.MIN_ERROR_SEAM],
    ids=str,
)
def test_aligned_shift_still_tiles(method: SeamlessMethod, map_type: MapType):
    guide = panels(300, 384, 70, seed=1)
    plan = create_plan(guide, method, 1, pattern_align=True)
    assert plan.fit_x < 1 and plan.fit_y < 1, plan.info
    if map_type == MapType.NORMAL_GL:
        img = normal_from_height(smooth_random(300, 384, 1, seed=2)[:, :, 0], map_type)
    elif map_type == MapType.MASK:
        img = (smooth_random(300, 384, 3, seed=3) > 0.5).astype(np.float32)
    else:
        img = smooth_random(300, 384, 3, seed=3)
    out = apply_plan(plan, img, map_type)
    assert seam_ratio(out) < 1.6
    # the same plan at half resolution (stack sync)
    half = cv2.resize(img, (192, 150), interpolation=cv2.INTER_AREA)
    if map_type == MapType.MASK:
        half = (half > 0.5).astype(np.float32)
    assert seam_ratio(apply_plan(plan, half, map_type)) < 1.6


@pytest.mark.parametrize("map_type", [MapType.NORMAL_GL, MapType.NORMAL_DX], ids=str)
def test_height_from_normals_recovers_shapes(map_type: MapType):
    height = smooth_random(128, 128, 1, seed=13)[:, :, 0]
    normal = normal_from_height(height, map_type)
    recovered = height_from_normals(normal, map_type)
    detail = height - cv2.GaussianBlur(height, (0, 0), 8)
    corr = np.corrcoef(detail.ravel(), recovered.ravel())[0, 1]
    assert corr > 0.6, corr


def test_x_only_tiles_horizontally_and_keeps_rows():
    from nodes.impl.seamless import TileAxes

    img = smooth_random(128, 160, 3, seed=14)
    plan = create_plan(
        img, SeamlessMethod.MIN_ERROR_SEAM, 1, pattern_align=True, axes=TileAxes.X
    )
    assert plan.masks[1] is None
    assert any("Y: not tiled" in line for line in plan.info)
    out = apply_plan(plan, img, MapType.COLOR)
    jump_x = np.abs(out[:, 0] - out[:, -1]).mean()
    step_x = 0.5 * (
        np.abs(out[:, 1] - out[:, 0]).mean() + np.abs(out[:, -1] - out[:, -2]).mean()
    )
    assert jump_x / step_x < 1.6
    # nothing is shifted vertically: the top and bottom rows still come from the original's
    # top and bottom rows (possibly from the shifted copy, but never from the middle)
    assert np.abs(out[0] - out[-1]).mean() > 3 * step_x
