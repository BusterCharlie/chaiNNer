from __future__ import annotations

import cv2
import numpy as np

from nodes.groups import if_enum_group
from nodes.impl.perspective import (
    PerspectiveMode,
    apply_perspective,
    plan_corners,
    plan_lines,
    plan_pattern,
)
from nodes.impl.seamless import MapType, height_from_normals
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
    Quad,
    QuadInput,
    SliderInput,
)
from nodes.properties.outputs import ImageOutput, InputPreviewOutput, TextOutput
from nodes.utils.utils import get_h_w_c

from .. import tiling_group

# the input preview sent to the corner editor never needs more detail than this
PREVIEW_SIZE = 1024


@tiling_group.register(
    schema_id="chainner:image:correct_perspective",
    name="Correct Perspective",
    description=[
        "Removes the perspective of a photo taken at an angle, so the surface looks like it was photographed straight on. Use it before *Make Seamless*.",
        "### Modes",
        "- **Automatic (Lines)**: straightens converging lines, so they become vertical and horizontal. For walls, bricks, tiles, panels, planks, ...",
        "- **Automatic (Pattern)**: makes a repeating pattern as regular as possible, for surfaces without long straight lines, such as tread plate, mesh, or fabric. It can also remove a slight rotation or shear of the pattern.",
        "- **Four Corners**: drag the four corner points onto a rectangle on the surface (e.g. the corners of a tile or a panel), and it becomes a rectangle. The image appears in the node as soon as it is available: immediately when it comes straight from *Load Image*, otherwise after running the chain once.",
        "### Processing a whole material",
        "Connect **the same Guide** to one *Correct Perspective* node per map, with the same settings, and every map gets exactly the same correction, even at different resolutions.",
    ],
    icon="MdCropRotate",
    inputs=[
        ImageInput("Image", channels=[1, 3, 4]),
        ImageInput("Guide", channels=[1, 3, 4])
        .make_optional()
        .with_docs(
            "The image the correction is detected on. If not connected, the image itself is used. Not used by the *Four Corners* mode."
        ),
        EnumInput(
            MapType,
            label="Map Type",
            default=MapType.COLOR,
            option_labels={
                MapType.COLOR: "Color / Grayscale",
                MapType.NORMAL_GL: "Normal Map (OpenGL)",
                MapType.NORMAL_DX: "Normal Map (DirectX)",
                MapType.MASK: "Mask (hard edges)",
            },
        ).with_id(2),
        EnumInput(
            PerspectiveMode,
            label="Mode",
            default=PerspectiveMode.LINES,
            option_labels={
                PerspectiveMode.LINES: "Automatic (Lines)",
                PerspectiveMode.PATTERN: "Automatic (Pattern)",
                PerspectiveMode.CORNERS: "Four Corners",
            },
        ).with_id(3),
        if_enum_group(3, PerspectiveMode.PATTERN)(
            SliderInput("Max Tilt", min=5, max=100, default=60, precision=0, unit="%")
            .with_docs("How strong a perspective the search considers.")
            .with_id(4),
            BoolInput("Align Pattern to Axes", default=True)
            .with_docs(
                "Also removes rotation and shear (up to 15 degrees), so the pattern repeats exactly horizontally and vertically. *Make Seamless* needs that to align patterns."
            )
            .with_id(5),
        ),
        if_enum_group(3, PerspectiveMode.CORNERS)(
            QuadInput("Corners", image_input=0, preview_output=2)
            .with_docs(
                "Drag the corners onto a rectangle on the surface. Click a corner and use the arrow keys for fine adjustment (hold Shift for bigger steps)."
            )
            .with_id(15),
        ),
        BoolInput("Crop to Valid Area", default=True)
        .with_docs(
            "Crops away the empty areas the correction creates. In *Four Corners* mode, the output is exactly the rectangle."
        )
        .with_id(14),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                Image {
                    width: int(1..),
                    height: int(1..),
                    channels: Input0.channels,
                }
            """,
        ),
        TextOutput("Info"),
        InputPreviewOutput(),
    ],
)
def correct_perspective_node(
    img: np.ndarray,
    guide: np.ndarray | None,
    map_type: MapType,
    mode: PerspectiveMode,
    max_tilt: int,
    align_to_axes: bool,
    corners: Quad,
    crop: bool,
) -> tuple[np.ndarray, str, np.ndarray]:
    h, w, _ = get_h_w_c(img)
    info: list[str] = []

    if mode == PerspectiveMode.CORNERS:
        plan = plan_corners(list(corners), w, h, crop=crop)
    else:
        if guide is not None:
            gh, gw, _ = get_h_w_c(guide)
            if abs(gw / gh - w / h) > 0.01:
                raise ValueError(
                    f"The guide ({gw}x{gh}) must have the same aspect ratio as the image ({w}x{h})."
                )
        elif map_type.is_normal:
            guide = height_from_normals(img, map_type)
            info.append("Guide: height reconstructed from the normal map.")
        else:
            guide = img
        if mode == PerspectiveMode.LINES:
            plan = plan_lines(guide, crop=crop)
        else:
            plan = plan_pattern(
                guide,
                max_tilt=max_tilt / 100,
                deskew=align_to_axes,
                crop=crop,
            )
    info.extend(plan.info)
    return apply_perspective(plan, img, map_type), "\n".join(info), _preview(img)


def _preview(img: np.ndarray) -> np.ndarray:
    """A downscaled copy of the input for the corner editor."""
    h, w, _ = get_h_w_c(img)
    scale = PREVIEW_SIZE / max(h, w)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
