from __future__ import annotations

import numpy as np

from nodes.condition import Condition
from nodes.groups import if_enum_group, if_group
from nodes.impl.color.color import Color
from nodes.impl.roughness import (
    BrightnessReference,
    ColorPick,
    RoughnessParams,
    generate_roughness,
)
from nodes.properties.inputs import (
    BoolInput,
    ColorInput,
    EnumInput,
    ImageInput,
    SliderInput,
)
from nodes.properties.outputs import ImageOutput

from .. import conversion_group


def weight_slider(label: str, input_id: int, docs: str):
    return (
        SliderInput(label, min=-100, max=100, default=0, precision=0, unit="%")
        .with_docs(docs)
        .with_id(input_id)
    )


def pick_inputs(n: int, first_id: int):
    """Inputs of one color pick: enable, color, roughness, tolerance, softness."""
    return (
        BoolInput(f"Color Pick {n}", default=False)
        .with_docs(
            "Gives areas of one color their own roughness (e.g. rust on painted metal). The color can be set here or connected from another node."
        )
        .with_id(first_id),
        if_group(Condition.bool(first_id, True))(
            ColorInput(f"Color {n}", channels=[3]).with_id(first_id + 1),
            SliderInput(
                f"Roughness {n}", min=0, max=100, default=80, precision=0, unit="%"
            ).with_id(first_id + 2),
            SliderInput(
                f"Tolerance {n}", min=0, max=50, default=12, precision=1, unit="%"
            )
            .with_docs(
                "How different a color may be and still count as the picked color."
            )
            .with_id(first_id + 3),
            SliderInput(
                f"Softness {n}", min=0, max=50, default=5, precision=1, unit="%"
            )
            .with_docs("Width of the fade between picked and other areas.")
            .with_id(first_id + 4),
        ),
    )


def to_pick(color: Color, roughness: int, tolerance: float, softness: float):
    bgr = color.value[:3] if color.channels >= 3 else (color.value[0],) * 3
    return ColorPick(
        color=(float(bgr[0]), float(bgr[1]), float(bgr[2])),
        roughness=roughness / 100,
        tolerance=tolerance / 100,
        softness=softness / 100,
    )


@conversion_group.register(
    schema_id="chainner:image:generate_roughness",
    name="Generate Roughness",
    description=[
        "Generates a roughness map from an albedo (color) texture, with optional curvature, ambient occlusion and metal maps.",
        "Roughness can't be measured from a photo, so this is an artistic tool: start from the albedo's brightness, set the **Average Roughness** and **Contrast**, then add layers and color picks until it looks right.",
        "Whether darker or brighter areas should be rougher depends on the material (e.g. dirt vs. dust), so it's a setting.",
        "Save the result as a 16-bit PNG to keep smooth gradients for fine-tuning in a game engine, and import it as linear (non-color) data.",
    ],
    icon="MdGrain",
    inputs=[
        ImageInput("Albedo", channels=[1, 3, 4]),
        ImageInput("Curvature", channels=1)
        .make_optional()
        .with_docs(
            "A curvature map: 50% gray = flat, brighter = convex (edges), darker = concave (cavities). Enables the *Edges* and *Cavities* layers."
        ),
        ImageInput("Ambient Occlusion", channels=1)
        .make_optional()
        .with_docs("White = open, black = occluded. Enables the *Occlusion* layer."),
        ImageInput("Metal Mask", channels=1)
        .make_optional()
        .with_docs("White = metal. Metal areas get the *Metal Roughness*."),
        ImageInput("Texture Mask", channels=1)
        .make_optional()
        .with_docs(
            "White = texture, black = background to ignore, e.g. the empty space between the islands of a UV atlas. The *Average Roughness* and *Contrast* are then measured on the texture only.",
            "Without a mask, the albedo's alpha channel is used if it has one.",
        )
        .with_id(25),
        SliderInput(
            "Average Roughness", min=0, max=100, default=60, precision=0, unit="%"
        ).with_id(4),
        SliderInput("Contrast", min=0, max=50, default=4, precision=1, unit="%")
        .with_docs(
            "How much the roughness varies with the albedo's brightness (its standard deviation). Scanned materials are usually subtle: about 2-8%, typically 4%."
        )
        .with_id(5),
        BoolInput("Brighter Is Rougher", default=False)
        .with_docs(
            "Off: darker areas are rougher (dirt, stains). On: brighter areas are rougher (dust, dry or matte spots)."
        )
        .with_id(6),
        EnumInput(
            BrightnessReference,
            label="Brightness Compared To",
            default=BrightnessReference.GLOBAL,
            option_labels={
                BrightnessReference.GLOBAL: "Whole Image",
                BrightnessReference.LOCAL: "Local Area",
            },
        )
        .with_docs(
            "*Local Area* compares each pixel with its surroundings instead of the whole image, which ignores large-scale lighting and shading in a photo."
        )
        .with_id(7),
        if_enum_group(7, BrightnessReference.LOCAL)(
            SliderInput("Local Radius", min=1, max=50, default=5, precision=1, unit="%")
            .with_docs("Size of the local area, relative to the image size.")
            .with_id(8),
        ),
        if_group(Condition.type(1, "Image"))(
            weight_slider(
                "Edges", 9, "Makes convex edges rougher (+) or smoother (-, e.g. worn)."
            ),
            weight_slider(
                "Cavities",
                10,
                "Makes cavities and gaps rougher (+, e.g. dirt) or smoother (-).",
            ),
        ),
        if_group(Condition.type(2, "Image"))(
            weight_slider(
                "Occlusion",
                11,
                "Makes occluded areas rougher (+, e.g. dust) or smoother (-).",
            ),
        ),
        weight_slider(
            "Fine Detail",
            12,
            "Makes areas with busy fine detail rougher (+) or smoother (-).",
        ),
        if_group(Condition.type(3, "Image"))(
            SliderInput(
                "Metal Roughness", min=0, max=100, default=30, precision=0, unit="%"
            ).with_id(13),
        ),
        *pick_inputs(1, 14),
        *pick_inputs(2, 19),
        BoolInput("Output Smoothness", default=False)
        .with_docs("Outputs smoothness (1 - roughness) instead, e.g. for Unity.")
        .with_id(24),
    ],
    outputs=[
        ImageOutput("Roughness", size_as=0, channels=1),
    ],
)
def generate_roughness_node(
    albedo: np.ndarray,
    curvature: np.ndarray | None,
    ao: np.ndarray | None,
    metal: np.ndarray | None,
    texture_mask: np.ndarray | None,
    average: int,
    contrast: float,
    brighter_is_rougher: bool,
    reference: BrightnessReference,
    local_radius: float,
    edges: int,
    cavities: int,
    occlusion: int,
    fine_detail: int,
    metal_roughness: int,
    use_pick_1: bool,
    color_1: Color,
    roughness_1: int,
    tolerance_1: float,
    softness_1: float,
    use_pick_2: bool,
    color_2: Color,
    roughness_2: int,
    tolerance_2: float,
    softness_2: float,
    output_smoothness: bool,
) -> np.ndarray:
    picks = []
    if use_pick_1:
        picks.append(to_pick(color_1, roughness_1, tolerance_1, softness_1))
    if use_pick_2:
        picks.append(to_pick(color_2, roughness_2, tolerance_2, softness_2))

    params = RoughnessParams(
        average=average / 100,
        contrast=contrast / 100,
        brighter_is_rougher=brighter_is_rougher,
        reference=reference,
        local_radius=local_radius / 100,
        edges=edges / 100,
        cavities=cavities / 100,
        occlusion=occlusion / 100,
        fine_detail=fine_detail / 100,
        metal_roughness=metal_roughness / 100,
        picks=tuple(picks),
    )
    rough = generate_roughness(albedo, params, curvature, ao, metal, texture_mask)
    return 1 - rough if output_smoothness else rough
