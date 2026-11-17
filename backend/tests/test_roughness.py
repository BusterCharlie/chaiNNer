from __future__ import annotations

import pytest

# The plain backend test job doesn't install the image dependencies.
pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("chainner_ext")

import cv2
import numpy as np

from nodes.impl.roughness import (
    BrightnessReference,
    ColorPick,
    RoughnessParams,
    color_mask,
    generate_roughness,
)


def albedo(seed: int = 0, size: int = 96) -> np.ndarray:
    small = np.random.default_rng(seed).uniform(0.2, 0.8, (8, 8, 3)).astype(np.float32)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def lum(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_average_and_contrast():
    r = generate_roughness(albedo(), RoughnessParams(average=0.4, contrast=0.08))
    assert r.shape == (96, 96)
    assert r.dtype == np.float32
    assert abs(r.mean() - 0.4) < 0.01
    assert abs(r.std() - 0.08) < 0.01


def test_direction_setting():
    a = albedo()
    darker = generate_roughness(a, RoughnessParams())
    brighter = generate_roughness(a, RoughnessParams(brighter_is_rougher=True))
    assert np.corrcoef(darker.ravel(), lum(a).ravel())[0, 1] < -0.99
    assert np.corrcoef(brighter.ravel(), lum(a).ravel())[0, 1] > 0.99


def test_local_reference_ignores_large_scale_shading():
    a = albedo()
    gradient = np.linspace(0.5, 1.0, 96, dtype=np.float32)[np.newaxis, :, np.newaxis]
    shaded = a * gradient
    params = RoughnessParams(reference=BrightnessReference.LOCAL, local_radius=0.05)
    r = generate_roughness(shaded, params)
    # the left/right halves get about the same average roughness despite the shading
    assert abs(r[:, :48].mean() - r[:, 48:].mean()) < 0.03
    g = generate_roughness(shaded, RoughnessParams())
    assert abs(g[:, :48].mean() - g[:, 48:].mean()) > 0.05


def test_layers_only_act_where_they_should():
    a = np.full(
        (64, 64, 3), 0.5, np.float32
    )  # flat albedo: base map is exactly the average
    curvature = np.full((64, 64), 0.5, np.float32)
    curvature[:, :16] = 1.0  # edges
    curvature[:, 48:] = 0.0  # cavities
    params = RoughnessParams(average=0.5, edges=-0.3, cavities=0.4)
    r = generate_roughness(a, params, curvature=curvature)
    assert np.allclose(r[:, :16], 0.2)
    assert np.allclose(r[:, 16:48], 0.5)
    assert np.allclose(r[:, 48:], 0.9)

    ao = np.ones((32, 32), np.float32)  # different size: gets resized
    ao[:, 16:] = 0.0
    r = generate_roughness(a, RoughnessParams(average=0.5, occlusion=0.3), ao=ao)
    assert np.allclose(r[:, :24], 0.5, atol=0.01)
    assert np.allclose(r[:, 40:], 0.8, atol=0.01)


def test_color_pick_and_metal_mask_override_regions():
    a = np.full((64, 64, 3), [0.25, 0.35, 0.25], np.float32)  # green paint (BGR)
    rust = (0.15, 0.35, 0.6)
    a[20:30, 20:30] = rust
    pick = ColorPick(color=rust, roughness=0.95, tolerance=0.05, softness=0.02)
    m = color_mask(a, pick)
    assert m[25, 25] > 0.99 and m[5, 5] < 0.01
    r = generate_roughness(a, RoughnessParams(average=0.4, picks=(pick,)))
    assert r[25, 25] > 0.9 and abs(r[5, 5] - 0.4) < 0.05

    metal = np.zeros((64, 64), np.float32)
    metal[40:, 40:] = 1
    r = generate_roughness(
        a, RoughnessParams(average=0.6, metal_roughness=0.2), metal=metal
    )
    assert np.allclose(r[50, 50], 0.2) and abs(r[5, 5] - 0.6) < 0.05


def test_grayscale_albedo_and_range():
    a = lum(albedo())
    r = generate_roughness(a, RoughnessParams(average=0.95, contrast=0.5))
    assert r.min() >= 0 and r.max() <= 1
