from __future__ import annotations

import pytest

# The plain backend test job doesn't install the image dependencies.
pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("chainner_ext")

import numpy as np

from nodes.impl.material_preview import PreviewSettings, PreviewShape, render_preview

GRAY = np.full((32, 32, 3), 0.5, np.float32)


def flat(roughness: float, **kwargs) -> np.ndarray:
    settings = PreviewSettings(shape=PreviewShape.FLAT, size=128, **kwargs)
    rough = np.full((32, 32), roughness, np.float32)
    return render_preview(GRAY, rough, settings=settings)


def brightness(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=2)


@pytest.mark.parametrize("shape", list(PreviewShape))
def test_output_format(shape: PreviewShape):
    out = render_preview(
        GRAY,
        np.full((16, 16), 0.5, np.float32),
        settings=PreviewSettings(shape=shape, size=96),
    )
    assert out.shape == (96, 96, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_smooth_highlight_is_brighter_and_tighter_than_rough():
    smooth, rough = brightness(flat(0.15)), brightness(flat(0.8))
    assert smooth.max() > rough.max() + 0.1

    # area clearly brighter than the median: small for a smooth, sharp highlight
    def hot(b: np.ndarray) -> float:
        median = float(np.median(b))
        return float((b > median + 0.5 * (b.max() - median)).mean())

    assert hot(smooth) < hot(rough)


def test_light_direction_moves_the_highlight():
    right = brightness(flat(0.2, light_azimuth=0))
    left = brightness(flat(0.2, light_azimuth=180))
    x_right = np.unravel_index(np.argmax(right), right.shape)[1]
    x_left = np.unravel_index(np.argmax(left), left.shape)[1]
    assert x_right > 64 > x_left


def test_metallic_tints_the_highlight():
    red = np.zeros((16, 16, 3), np.float32)
    red[:, :, 2] = 0.9  # BGR: red
    rough = np.full((16, 16), 0.2, np.float32)
    settings = PreviewSettings(shape=PreviewShape.FLAT, size=128, light_azimuth=90)
    dielectric = render_preview(red, rough, settings=settings)
    metal = render_preview(
        red, rough, metallic=np.ones((16, 16), np.float32), settings=settings
    )

    def highlight_color(img: np.ndarray) -> np.ndarray:
        y, x = np.unravel_index(np.argmax(img.mean(2)), img.shape[:2])
        return img[y, x]

    _, g, r = highlight_color(dielectric)
    assert abs(r - g) < 0.15  # white-ish highlight on plastic
    _, g, r = highlight_color(metal)
    assert r > g + 0.2  # colored highlight on metal


def bump_normal_map(directx: bool = False) -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    height = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 10**2))
    dy, dx = np.gradient(height * 10)
    x, y = -dx, dy
    if directx:
        y = -y
    n = np.dstack([np.ones_like(x), y, x])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def test_normal_map_shading_and_directx_flag():
    rough = np.full((64, 64), 0.6, np.float32)

    def settings_for(directx: bool) -> PreviewSettings:
        return PreviewSettings(
            shape=PreviewShape.FLAT,
            size=128,
            tiling=1.0,
            light_azimuth=90,
            light_elevation=20,
            normal_directx=directx,
        )

    settings = settings_for(directx=False)
    lit = brightness(render_preview(GRAY, rough, bump_normal_map(), settings=settings))
    # light from the top: the upper side of the bump faces it
    upper, lower = lit[34:58, 54:74].mean(), lit[70:94, 54:74].mean()
    assert upper > lower + 0.05

    # a DirectX map with the DirectX flag looks the same as the OpenGL map
    dx_settings = settings_for(directx=True)
    dx = brightness(
        render_preview(GRAY, rough, bump_normal_map(directx=True), settings=dx_settings)
    )
    assert np.abs(dx - lit).mean() < 0.01


def test_grayscale_albedo_and_mixed_sizes():
    out = render_preview(
        np.full((40, 40), 0.5, np.float32),
        np.full((10, 20), 0.5, np.float32),
        metallic=np.zeros((7, 7), np.float32),
        settings=PreviewSettings(size=64),
    )
    assert out.shape == (64, 64, 3)
