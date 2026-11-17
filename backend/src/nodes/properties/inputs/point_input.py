from __future__ import annotations

import math

from api import BaseInput

Point = tuple[float, float]
"""A position as fractions of an image's width and height (0..1)."""


def parse_point(value: str) -> Point:
    """Parses "x,y" (fractions of the image size); values are clamped to the image."""
    try:
        x, y = (float(v) for v in value.split(","))
    except ValueError:
        raise ValueError(f"Invalid position: {value!r}") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"Invalid position: {value!r}")
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


class PointInput(BaseInput[Point]):
    """
    A position on an image, picked by clicking on a preview of the image inside the node.
    The value is stored as "x,y" (fractions of the image size).
    """

    def __init__(
        self,
        label: str = "Position",
        *,
        image_input: int = 0,
        preview_output: int | None = None,
        default: str = "0.5,0.5",
    ):
        super().__init__(
            input_type="string",
            label=label,
            has_handle=False,
            kind="point",
        )
        parse_point(default)
        self.default = default
        self.image_input = image_input
        self.preview_output = preview_output
        self.associated_type = Point

    def enforce(self, value: object) -> Point:
        return parse_point(str(value))

    def to_dict(self):
        return {
            **super().to_dict(),
            "def": self.default,
            "imageInput": self.image_input,
            "previewOutput": self.preview_output,
        }
