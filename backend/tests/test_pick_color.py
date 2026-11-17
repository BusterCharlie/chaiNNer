from __future__ import annotations

import numpy as np
import pytest

from nodes.properties.inputs.point_input import parse_point
from packages.chaiNNer_standard.image_utility.miscellaneous.pick_color import (
    PREVIEW_SIZE,
    CoordinateType,
    pick_color_node,
)


def _image() -> np.ndarray:
    img = np.zeros((40, 60, 3), np.float32)
    img[:20, :30] = (0.1, 0.2, 0.3)  # BGR top-left quadrant
    img[20:, 30:] = (1.0, 1.0, 1.0)
    return img


def test_parse_point() -> None:
    assert parse_point("0.5,0.25") == (0.5, 0.25)
    assert parse_point(" 0 , 1 ") == (0.0, 1.0)
    assert parse_point("1.5,-1") == (1.0, 0.0)  # clamped
    for bad in ["", "0.5", "a,b", "0.1,0.2,0.3", "nan,0"]:
        with pytest.raises(ValueError):
            parse_point(bad)


def test_click_mode_picks_pixel_under_point() -> None:
    img = _image()
    color, _ = pick_color_node(img, CoordinateType.CLICK, 0, 0, 0, 0, (0.1, 0.1), 0)
    assert color.channels == 3
    np.testing.assert_allclose(color.value, (0.1, 0.2, 0.3), atol=1e-6)
    color, _ = pick_color_node(img, CoordinateType.CLICK, 0, 0, 0, 0, (1.0, 1.0), 0)
    np.testing.assert_allclose(color.value, (1.0, 1.0, 1.0))


def test_sample_radius_averages() -> None:
    img = _image()
    # on the boundary between black and white quadrants, averaging mixes them
    color, _ = pick_color_node(img, CoordinateType.ABSOLUTE, 0, 0, 30, 30, (0, 0), 5)
    v = np.asarray(color.value)
    assert 0.2 < v.mean() < 0.8
    # radius 0 is the old exact-pixel behavior
    color, _ = pick_color_node(img, CoordinateType.ABSOLUTE, 0, 0, 30, 30, (0, 0), 0)
    np.testing.assert_allclose(color.value, (1.0, 1.0, 1.0))


def test_existing_modes_unchanged() -> None:
    img = _image()
    color, preview = pick_color_node(
        img, CoordinateType.RELATIVE, 0, 0, 0, 0, (0, 0), 0
    )
    np.testing.assert_allclose(color.value, (0.1, 0.2, 0.3), atol=1e-6)
    assert preview.shape[:2] == (1, 1)  # no preview needed outside click mode


def test_gray_and_alpha() -> None:
    gray = np.full((8, 8), 0.25, np.float32)
    color, _ = pick_color_node(gray, CoordinateType.CLICK, 0, 0, 0, 0, (0.5, 0.5), 2)
    assert color.channels == 1
    assert color.value[0] == pytest.approx(0.25)
    rgba = np.full((8, 8, 4), 0.5, np.float32)
    color, _ = pick_color_node(rgba, CoordinateType.CLICK, 0, 0, 0, 0, (0.5, 0.5), 0)
    assert color.channels == 4


def test_preview_is_downscaled() -> None:
    img = np.zeros((PREVIEW_SIZE * 2, PREVIEW_SIZE, 3), np.float32)
    _, preview = pick_color_node(img, CoordinateType.CLICK, 0, 0, 0, 0, (0.5, 0.5), 0)
    assert preview.shape[:2] == (PREVIEW_SIZE, PREVIEW_SIZE // 2)
