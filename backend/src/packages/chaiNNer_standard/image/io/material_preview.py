from __future__ import annotations

import numpy as np

from nodes.condition import Condition
from nodes.groups import if_group
from nodes.impl.material_preview import PreviewSettings, PreviewShape, render_preview
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
    NumberInput,
    SliderInput,
)
from nodes.properties.outputs import LargeImageOutput

from .. import io_group


@io_group.register(
    schema_id="chainner:image:material_preview",
    name="Material Preview",
    description=[
        "Renders a quick preview of a PBR material (metallic/roughness) on a sphere or a flat tile, lit by one adjustable light and a simple studio environment.",
        "Use it to judge a roughness, normal or metallic map by eye before taking the textures into a game engine. It's an approximation, not a match for any particular engine.",
    ],
    icon="MdLightbulbOutline",
    inputs=[
        ImageInput("Albedo", channels=[1, 3, 4]),
        ImageInput("Roughness", channels=1),
        ImageInput("Normal Map", channels=[3, 4]).make_optional(),
        ImageInput("Metallic", channels=1).make_optional(),
        if_group(Condition.type(2, "Image"))(
            BoolInput("DirectX Normal Map", default=False)
            .with_docs(
                "Turn on if the normal map uses the DirectX convention (Y down)."
            )
            .with_id(4),
        ),
        EnumInput(
            PreviewShape,
            label="Shape",
            default=PreviewShape.SPHERE,
            option_labels={
                PreviewShape.SPHERE: "Sphere",
                PreviewShape.FLAT: "Flat Tile",
            },
        ).with_id(5),
        SliderInput(
            "Light Direction", min=0, max=360, default=135, precision=0, unit="°"
        )
        .with_docs(
            "Direction the light comes from: 0° = right, 90° = top, 180° = left."
        )
        .with_id(6),
        SliderInput("Light Height", min=5, max=90, default=45, precision=0, unit="°")
        .with_docs("Angle of the light above the surface. Low angles show bumps best.")
        .with_id(7),
        SliderInput(
            "Light Intensity", min=0, max=400, default=100, precision=0, unit="%"
        ).with_id(8),
        SliderInput("Tiling", min=1, max=8, default=2, precision=1)
        .with_docs("How many times the texture repeats across the preview.")
        .with_id(9),
        NumberInput("Size", min=64, max=2048, default=512, unit="px").with_id(10),
    ],
    outputs=[
        LargeImageOutput(
            "Preview",
            image_type="""
                Image {
                    width: Input10,
                    height: Input10,
                    channels: 3,
                }
            """,
            assume_normalized=True,
        ),
    ],
    side_effects=True,
)
def material_preview_node(
    albedo: np.ndarray,
    roughness: np.ndarray,
    normal: np.ndarray | None,
    metallic: np.ndarray | None,
    normal_directx: bool,
    shape: PreviewShape,
    light_direction: int,
    light_height: int,
    light_intensity: int,
    tiling: float,
    size: int,
) -> np.ndarray:
    settings = PreviewSettings(
        shape=shape,
        size=size,
        tiling=tiling,
        light_azimuth=light_direction,
        light_elevation=light_height,
        light_intensity=light_intensity / 100,
        normal_directx=normal_directx,
    )
    return render_preview(albedo, roughness, normal, metallic, settings)
