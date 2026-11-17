from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from nodes.groups import if_enum_group
from nodes.impl.color.color import Color
from nodes.properties.inputs import (
    EnumInput,
    ImageInput,
    NumberInput,
    Point,
    PointInput,
    SliderInput,
)
from nodes.properties.outputs import ColorOutput, InputPreviewOutput
from nodes.utils.utils import get_h_w_c, round_half_up

from .. import miscellaneous_group


class CoordinateType(Enum):
    RELATIVE = 1
    ABSOLUTE = 2
    CLICK = 3


# the image preview sent to the position editor never needs more detail than this
PREVIEW_SIZE = 1024


@miscellaneous_group.register(
    schema_id="chainner:image:pick_color",
    name="Pick Color",
    description=[
        "Returns the color of the pixel at the given coordinate.",
        "This node supports 2 coordinate types:",
        "- Relative: The coordinates are relative to the image size. So (X=0, Y=0) is the top-left pixel, ((X=50%, Y=50%) is the center pixel, and (X=100%, Y=100%) is the bottom-right pixel.",
        "- Absolute: The coordinates are absolute pixel coordinates. So (X=0, Y=0) is the top-left pixel and (X=width-1, Y=height-1) is the bottom-right pixel. X-Y coordinates outside the image are not allowed and will result in an error.",
        "- Click: Click on the image inside the node to pick the position. The image appears as soon as it is available: immediately when it comes straight from *Load Image*, otherwise after running the chain once.",
        "**Sample Radius** averages the pixels around the position, which gives a more representative color for noisy photos.",
    ],
    icon="MdColorize",
    inputs=[
        ImageInput(channels=[1, 3, 4]).with_id(0),
        EnumInput(
            CoordinateType,
            default=CoordinateType.ABSOLUTE,
            preferred_style="tabs",
            option_labels={CoordinateType.CLICK: "Click"},
        ).with_id(3),
        if_enum_group(3, CoordinateType.RELATIVE)(
            SliderInput("X", max=100, default=0, unit="%")
            .with_docs("Relative X coordinate.")
            .with_id(4),
            SliderInput("Y", max=100, default=0, unit="%")
            .with_docs("Relative Y coordinate.")
            .with_id(5),
        ),
        if_enum_group(3, CoordinateType.ABSOLUTE)(
            NumberInput("X", default=0, min=0, unit="px")
            .with_docs("Absolute X coordinate.")
            .with_id(1),
            NumberInput("Y", default=0, min=0, unit="px")
            .with_docs("Absolute Y coordinate.")
            .with_id(2),
        ),
        if_enum_group(3, CoordinateType.CLICK)(
            PointInput("Position", image_input=0, preview_output=1)
            .with_docs(
                "Click or drag on the image to pick the position, then use the arrow keys to nudge it by one pixel (hold Shift for 10 pixels)."
            )
            .with_id(6),
        ),
        SliderInput("Sample Radius", min=0, max=50, default=0, unit="px")
        .with_docs(
            "Averages the pixels within this radius around the position. 0 picks the single pixel."
        )
        .with_id(7),
    ],
    outputs=[
        ColorOutput(
            "Color",
            color_type="""
                let image = Input0;
                let coords = Input3;
                let x = Input1;
                let y = Input2;

                let valid = match coords {
                    CoordinateType::Relative => true,
                    CoordinateType::Absolute => x < image.width and y < image.height,
                    CoordinateType::Click => true,
                };

                if valid {
                    Color { channels: Input0.channels }
                } else {
                    never
                }
            """,
        ).with_never_reason("The given coordinates (X, Y) are outside the image."),
        InputPreviewOutput(),
    ],
)
def pick_color_node(
    orig_img: np.ndarray,
    coord_type: CoordinateType,
    x_rel: int,
    y_rel: int,
    x: int,
    y: int,
    position: Point,
    sample_radius: int,
) -> tuple[Color, np.ndarray]:
    h, w, c = get_h_w_c(orig_img)

    if coord_type == CoordinateType.RELATIVE:
        x = round_half_up((w - 1) * x_rel / 100)
        y = round_half_up((h - 1) * y_rel / 100)
    elif coord_type == CoordinateType.CLICK:
        x = min(int(position[0] * w), w - 1)
        y = min(int(position[1] * h), h - 1)

    assert x < w and y < h, (
        "The given coordinates (X, Y) are outside the image."
        f" Expect X={x} to be to 0 to {w - 1} and Y={y} to be to 0 to {h - 1}."
    )

    r = max(0, sample_radius)
    if r == 0:
        value = orig_img[y, x]
    else:
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        value = orig_img[y0:y1, x0:x1][inside].mean(axis=0)

    if c == 1:
        color = Color.gray(float(np.asarray(value).reshape(-1)[0]))
    elif c == 3:
        color = Color.bgr(np.asarray(value).reshape(-1))
    elif c == 4:
        color = Color.bgra(np.asarray(value).reshape(-1))
    else:
        raise ValueError(f"Unsupported number of channels: {c}")

    return color, _preview(orig_img, coord_type)


def _preview(img: np.ndarray, coord_type: CoordinateType) -> np.ndarray:
    """A downscaled copy of the image for the position editor (tiny when not needed)."""
    if coord_type != CoordinateType.CLICK:
        return img[:1, :1]
    h, w, _ = get_h_w_c(img)
    scale = PREVIEW_SIZE / max(h, w)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
