from __future__ import annotations

import numpy as np

from nodes.condition import Condition
from nodes.groups import if_enum_group, if_group, seed_group
from nodes.impl.seamless import (
    LinearParams,
    MapType,
    QuincunxParams,
    SeamlessMethod,
    SeamParams,
    SplatParams,
    apply_plan,
    create_plan,
    crop_margin_fraction,
    equalize_lighting,
)
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
    NumberInput,
    SeedInput,
    SliderInput,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.seed import Seed
from nodes.utils.utils import get_h_w_c

from .. import tiling_group

LINEAR_METHODS = (
    SeamlessMethod.EDGES_LINEAR,
    SeamlessMethod.EDGES_LINEAR_X,
    SeamlessMethod.EDGES_LINEAR_Y,
)
QUINCUNX_METHODS = (SeamlessMethod.EDGES_QUINCUNX, SeamlessMethod.SPLAT_RANDOM)


@tiling_group.register(
    schema_id="chainner:image:make_seamless",
    name="Make Seamless",
    description=[
        "Makes an image seamlessly tileable by blending it with shifted copies of itself.",
        "### Methods",
        "- **Edges Linear** (and **X**/**Y** only): blends the image with a copy shifted by half its size, using straight transitions. Works well for patterns, but blended areas can look doubled.",
        "- **Edges Quincunx**: blends with a diagonally shifted copy using a warped diamond mask, and patches the 4 tips of the diamond with copies that are seamless there. Doesn't depend on the image content, so maps stay in sync even without a Guide.",
        "- **Min-Error Seam** (default): cuts along the path where the image and its shifted copy look most alike, so seams follow the content (e.g. the gaps between stones). Large-scale brightness is blended smoothly so the cut doesn't show on smooth textures. Gave the best results on average in our tests. Seams depend on the content, so connect a Guide when processing a whole material.",
        "- **Splat Random**: starts from Edges Quincunx and stamps soft, randomly placed patches of the image on top. Different seeds give different variations that all tile.",
        "### Processing a whole material",
        "To process several maps of a material (albedo, normal, height, roughness, ...) in exactly the same way, use one *Make Seamless* node per map and connect **the same Guide and Seed** to all of them, with the same settings. A height map makes a good guide. Set **Map Type** for each map so that normal maps and masks are handled correctly.",
        "The maps may have different resolutions, as long as they have the same aspect ratio as the guide.",
        "Normal maps are read from their red and green channels only (blue is rebuilt), so two-channel normal maps work too.",
        "### Regular patterns",
        "Strictly regular patterns (bricks, floor tiles, fabric) only tile if the image contains a whole number of pattern repeats. Crop or resize the image to a whole number of repeats first.",
        "### Bad edges",
        "The methods mostly use the center of the image already. If the edges of a photo are bad (vignetting, lens blur, a ruler in frame, ...), use **Crop Margin** to remove them. Upscaling the cropped image back to its original size does not help: it adds no real detail, and AI upscalers invent different detail for each map of a material.",
    ],
    icon="MdGridOn",
    inputs=[
        ImageInput("Image", channels=[1, 3, 4]),
        ImageInput("Guide", channels=[1, 3, 4])
        .make_optional()
        .with_docs(
            "The image that decides where seams go. If not connected, the image itself is used.",
            "Connect the same guide to every map of a material so that all maps get identical seams.",
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
        )
        .with_docs(
            "- **Color / Grayscale**: albedo, roughness, height, AO, ...",
            "- **Normal Map**: normals are re-normalized after blending, and rotated patches also get their normal vectors rotated.",
            "- **Mask**: metalness masks, IDs, opacity, ... Transitions are hard so no in-between values are created.",
        )
        .with_id(2),
        EnumInput(
            SeamlessMethod,
            label="Method",
            default=SeamlessMethod.MIN_ERROR_SEAM,
            option_labels={
                SeamlessMethod.EDGES_LINEAR: "Edges Linear",
                SeamlessMethod.EDGES_LINEAR_X: "Edges Linear X",
                SeamlessMethod.EDGES_LINEAR_Y: "Edges Linear Y",
                SeamlessMethod.EDGES_QUINCUNX: "Edges Quincunx",
                SeamlessMethod.MIN_ERROR_SEAM: "Min-Error Seam",
                SeamlessMethod.SPLAT_RANDOM: "Splat Random",
            },
        ).with_id(3),
        seed_group(SeedInput().with_id(4)),
        SliderInput("Crop Margin", min=0, max=25, default=0, precision=1, unit="%")
        .with_docs(
            "Crops this much from every side of the image (and the guide) before making it seamless. Use it to drop bad edges of a photo, e.g. vignetting, lens blur, or a ruler in frame.",
            "The margin is a percentage, so maps of different resolutions are cropped to the same area.",
        )
        .with_id(21),
        if_enum_group(3, LINEAR_METHODS)(
            SliderInput(
                "Transition Width", min=1, max=100, default=50, precision=1, unit="%"
            ).with_id(5),
        ),
        if_enum_group(3, QUINCUNX_METHODS)(
            SliderInput(
                "Transition Softness", min=1, max=100, default=15, precision=1, unit="%"
            ).with_id(6),
            SliderInput(
                "Mask Warp", min=0, max=50, default=8, precision=1, unit="%"
            ).with_id(7),
            NumberInput("Warp Detail", min=1, max=64, default=6).with_id(8),
            SliderInput(
                "Content Influence", min=0, max=100, default=0, precision=1, unit="%"
            )
            .with_docs(
                "Lets the guide move the transition, so that brighter/higher areas of the guide win. With a height map as guide, transitions follow the gaps between stones, bricks, etc."
            )
            .with_id(9),
        ),
        if_enum_group(3, SeamlessMethod.MIN_ERROR_SEAM)(
            SliderInput(
                "Search Width", min=10, max=100, default=80, precision=1, unit="%"
            )
            .with_docs("How far the seams may wander from their default position.")
            .with_id(10),
            SliderInput(
                "Seam Feather", min=0, max=20, default=2, precision=1, unit="px"
            ).with_id(11),
            SliderInput("Smooth Blend", min=0, max=10, default=3, precision=1, unit="%")
            .with_docs(
                "Large-scale brightness is blended over a transition of this size, so the cut doesn't show as a step on smooth textures. 0 disables it."
            )
            .with_id(22),
        ),
        if_enum_group(3, SeamlessMethod.SPLAT_RANDOM)(
            NumberInput("Splat Count", min=1, max=500, default=24).with_id(12),
            SliderInput(
                "Splat Size", min=1, max=50, default=16, precision=1, unit="%"
            ).with_id(13),
            SliderInput(
                "Size Variation", min=0, max=100, default=30, precision=1, unit="%"
            ).with_id(14),
            SliderInput(
                "Rotation Variation", min=0, max=180, default=0, precision=1, unit="°"
            ).with_id(15),
            SliderInput(
                "Splat Softness", min=1, max=100, default=35, precision=1, unit="%"
            ).with_id(16),
            SliderInput(
                "Splat Warp", min=0, max=100, default=25, precision=1, unit="%"
            ).with_id(17),
        ),
        if_enum_group(2, MapType.COLOR)(
            BoolInput("Equalize Lighting", default=False)
            .with_docs(
                "Removes large-scale lighting differences (e.g. a photo that is brighter on one side) before tiling. Without this, such gradients turn into a visible repeating pattern."
            )
            .with_id(18),
            if_group(Condition.bool(18, True))(
                SliderInput(
                    "Equalize Radius", min=1, max=50, default=15, precision=1, unit="%"
                ).with_id(19),
            ),
            BoolInput("Preserve Contrast", default=False)
            .with_docs(
                "Keeps blended areas from looking washed out. Works best for noisy, stochastic textures such as sand, gravel, or grass."
            )
            .with_id(20),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                let margin = Input21;
                let mx = floor(Input0.width * margin / 100);
                let my = floor(Input0.height * margin / 100);
                // The clamps never change the result (the margin is at most 25%),
                // they only help the type system when the size is unknown.
                Image {
                    width: min(Input0.width, max(1, Input0.width - 2 * mx)),
                    height: min(Input0.height, max(1, Input0.height - 2 * my)),
                    channels: Input0.channels,
                }
            """,
        ),
    ],
)
def make_seamless_node(
    img: np.ndarray,
    guide: np.ndarray | None,
    map_type: MapType,
    method: SeamlessMethod,
    seed: Seed,
    crop_margin: float,
    transition_width: float,
    transition_softness: float,
    mask_warp: float,
    warp_detail: int,
    content_influence: float,
    search_width: float,
    seam_feather: float,
    smooth_blend: float,
    splat_count: int,
    splat_size: float,
    size_variation: float,
    rotation_variation: float,
    splat_softness: float,
    splat_warp: float,
    equalize: bool,
    equalize_radius: float,
    preserve_contrast: bool,
) -> np.ndarray:
    img = crop_margin_fraction(img, crop_margin)
    if guide is not None:
        guide = crop_margin_fraction(guide, crop_margin)

    h, w, _ = get_h_w_c(img)

    if guide is not None:
        gh, gw, _ = get_h_w_c(guide)
        if abs(gw / gh - w / h) > 0.01:
            raise ValueError(
                f"The guide ({gw}x{gh}) must have the same aspect ratio as the image ({w}x{h})."
            )

    is_color = map_type == MapType.COLOR
    if is_color and equalize:
        img = equalize_lighting(img, equalize_radius / 100)

    if guide is None:
        guide = img

    plan = create_plan(
        guide,
        method,
        seed.to_u32(),
        linear=LinearParams(width=transition_width / 100),
        quincunx=QuincunxParams(
            softness=transition_softness / 100,
            warp=mask_warp / 100,
            warp_detail=warp_detail,
            content_influence=content_influence / 100,
        ),
        seam=SeamParams(
            search_width=search_width / 100,
            feather=seam_feather,
            low_freq_blend=smooth_blend / 100,
        ),
        splat=SplatParams(
            count=splat_count,
            size=splat_size / 100,
            size_variation=size_variation / 100,
            rotation_variation=rotation_variation,
            softness=splat_softness / 100,
            warp=splat_warp / 100,
        ),
        preserve_contrast=is_color and preserve_contrast,
    )

    return apply_plan(plan, img, map_type)
