from __future__ import annotations

import numpy as np

from nodes.impl.curvature import CurvatureSource, curvature_maps
from nodes.properties.inputs import BoolInput, EnumInput, ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import normal_map_group


@normal_map_group.register(
    schema_id="chainner:image:curvature_map",
    name="Curvature Map",
    description=[
        "Generates a curvature map from a normal map or a height map. Convex areas (edges, ridges) become brighter and concave areas (cavities, gaps) darker; flat areas are 50% gray.",
        "The **Convex** and **Concave** outputs split the two halves into separate masks, e.g. for edge wear and dirt in cavities.",
        "A normal map usually gives the crispest result. Height maps often contain large, gentle undulations that show up as broad blotches.",
    ],
    icon="MdOutlineAutoFixHigh",
    inputs=[
        ImageInput("Image", channels=[1, 3, 4]),
        EnumInput(
            CurvatureSource,
            label="Source",
            default=CurvatureSource.NORMAL_GL,
            option_labels={
                CurvatureSource.NORMAL_GL: "Normal Map (OpenGL)",
                CurvatureSource.NORMAL_DX: "Normal Map (DirectX)",
                CurvatureSource.HEIGHT: "Height Map",
            },
        ),
        SliderInput(
            "Radius",
            min=1,
            max=128,
            default=16,
            precision=1,
            scale="log",
            unit="px",
        ).with_docs(
            "The size of the largest shapes that count. Curvature is accumulated from 1 pixel up to this radius, so small values pick up only fine detail and large values also include broad shapes."
        ),
        SliderInput(
            "Strength", min=0, max=400, default=100, precision=0, unit="%"
        ).with_docs("Contrast of the result."),
        BoolInput("Auto Level", default=True).with_docs(
            "Stretches the result so its strongest curvature reaches black and white. Turn it off to get the same scale for every image (e.g. when processing several tiles of one material)."
        ),
        BoolInput("Tileable", default=False).with_docs(
            "Treats the image as tileable, so the edges wrap around without a seam."
        ),
    ],
    outputs=[
        ImageOutput("Curvature", size_as=0, channels=1),
        ImageOutput("Convex", size_as=0, channels=1),
        ImageOutput("Concave", size_as=0, channels=1),
    ],
)
def curvature_map_node(
    img: np.ndarray,
    source: CurvatureSource,
    radius: float,
    strength: int,
    auto_level: bool,
    tileable: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source != CurvatureSource.HEIGHT and get_h_w_c(img)[2] < 3:
        raise ValueError(
            "A normal map needs at least 3 channels (RGB). For a grayscale height map, choose *Height Map* as the source."
        )
    return curvature_maps(img, source, radius, strength / 100, auto_level, tileable)
