"""
Make an image seamlessly tileable.

Every method is split in two phases:

1. `create_plan(guide, ...)` computes a resolution-independent `SeamlessPlan` (blend masks and
   a list of splats in UV space) from a guide image and a seed.
2. `apply_plan(plan, img, map_type)` applies that plan to an image at the image's own
   resolution.

Because the plan only depends on the guide, the seed, and the parameters, applying the same plan
to every map of a material (albedo, normal, height, roughness, ...) produces perfectly aligned
results, even if the maps have different resolutions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..utils.utils import get_h_w_c
from .image_utils import fast_gaussian_blur

# Plans are computed at most at this resolution (long side) and resized for each map.
MAX_PLAN_SIZE = 2048


class SeamlessMethod(Enum):
    EDGES_LINEAR = "edges_linear"
    EDGES_LINEAR_X = "edges_linear_x"
    EDGES_LINEAR_Y = "edges_linear_y"
    EDGES_QUINCUNX = "edges_quincunx"
    MIN_ERROR_SEAM = "min_error_seam"
    SPLAT_RANDOM = "splat_random"


class MapType(Enum):
    COLOR = "color"
    NORMAL_GL = "normal_gl"
    NORMAL_DX = "normal_dx"
    MASK = "mask"

    @property
    def is_normal(self) -> bool:
        return self in (MapType.NORMAL_GL, MapType.NORMAL_DX)


@dataclass(frozen=True)
class LinearParams:
    width: float = 0.5
    """Fraction of the half-size used for the transition (0..1)."""


@dataclass(frozen=True)
class QuincunxParams:
    softness: float = 0.15
    """Width of the diamond transition (fraction of the diamond radius)."""
    warp: float = 0.08
    """Strength of the noise warping of the diamond mask."""
    warp_detail: int = 6
    """Number of noise cells across the image."""
    content_influence: float = 0.0
    """How strongly the guide moves the transition (0 = purely geometric mask)."""


@dataclass(frozen=True)
class SeamParams:
    search_width: float = 0.8
    """Fraction of the available band in which the seam may wander (0..1)."""
    feather: float = 2.0
    """Blur radius of the seam, in pixels of the plan resolution."""
    low_freq_blend: float = 0.03
    """
    Large-scale brightness is blended over a wide transition of this size (fraction of the
    short side), so the sharp cut doesn't show as a step on smooth textures. 0 disables it.
    """


@dataclass(frozen=True)
class SplatParams:
    count: int = 24
    size: float = 0.16
    """Splat radius as a fraction of the short image side."""
    size_variation: float = 0.3
    rotation_variation: float = 0.0
    """Maximum random rotation in degrees."""
    softness: float = 0.35
    warp: float = 0.25


@dataclass(frozen=True)
class Splat:
    cx: float
    """Destination centre in UV."""
    cy: float
    sx: float
    """Source centre in UV."""
    sy: float
    radius: float
    """Radius as a fraction of the short image side."""
    angle: float
    """Rotation in degrees (counter-clockwise)."""
    seed: int


@dataclass
class SeamlessPlan:
    method: SeamlessMethod
    masks: list[np.ndarray | None] = field(default_factory=list)
    """
    Blend masks at plan resolution (1 = keep the current image, 0 = take the shifted copy).
    """
    splats: list[Splat] = field(default_factory=list)
    splat_softness: float = 0.35
    splat_warp: float = 0.25
    preserve_contrast: bool = False
    low_masks: list[np.ndarray | None] = field(default_factory=list)
    """Optional wide masks for the low-frequency band (two-band blending)."""
    band_sigma: float = 0.0
    """Blur sigma separating the two bands, as a fraction of the short side."""


# --------------------------------------------------------------------------- utilities


def _as3(img: np.ndarray) -> np.ndarray:
    return img[:, :, np.newaxis] if img.ndim == 2 else img


def _roll(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(img, (dy, dx), axis=(0, 1))


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _luminance(img: np.ndarray) -> np.ndarray:
    """Luminance of a BGR(A) or grayscale image."""
    img = _as3(img)
    c = img.shape[2]
    if c >= 3:
        return (
            img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
        ).astype(np.float32)
    return img[:, :, 0].astype(np.float32)


def _tileable_noise(h: int, w: int, cells: int, seed: int) -> np.ndarray:
    """Periodic value noise in [-1, 1]. It wraps around, so masks warped by it stay tileable."""
    rng = np.random.default_rng(seed)
    cells = max(1, cells)
    grid = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
    # Repeat the grid so that bicubic interpolation sees the wrap-around on both sides.
    big = np.tile(grid, (3, 3))
    up = cv2.resize(big, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    return up[h : 2 * h, w : 2 * w]


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[:2] == (h, w):
        return mask
    interpolation = (
        cv2.INTER_AREA if mask.shape[0] * mask.shape[1] > h * w else cv2.INTER_LINEAR
    )
    return cv2.resize(mask, (w, h), interpolation=interpolation)


def _wrap_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that treats the image as periodic."""
    if sigma <= 0:
        return mask
    p = max(1, math.ceil(sigma * 3))
    padded = np.pad(mask, p, mode="wrap")
    return cv2.GaussianBlur(padded, (0, 0), sigma)[p:-p, p:-p]


def _normal_xy(img: np.ndarray, map_type: MapType) -> tuple[np.ndarray, np.ndarray]:
    """Returns the X and Y components of a BGR normal map in OpenGL convention."""
    x = img[:, :, 2] * 2 - 1
    y = img[:, :, 1] * 2 - 1
    if map_type == MapType.NORMAL_DX:
        y = -y
    return x, y


def _decode_normals(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    XYZ of a BGR normal map, read from the red and green channels only. Z is rebuilt, so maps
    with an empty or junk blue channel (e.g. BC5/two-channel normal maps) still work.
    """
    x = img[:, :, 2] * 2 - 1
    y = img[:, :, 1] * 2 - 1
    z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
    return x, y, z


def _blend_normals(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Blends two BGR normal maps as vectors and re-normalizes the result."""
    ax, ay, az = _decode_normals(a)
    bx, by, bz = _decode_normals(b)
    m2 = m[:, :, 0]
    x = ax * m2 + bx * (1 - m2)
    y = ay * m2 + by * (1 - m2)
    z = az * m2 + bz * (1 - m2)
    length = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-6)
    out = a * m + b * (1 - m)  # keeps any extra (alpha) channel
    out[:, :, 0] = (z / length + 1) * 0.5
    out[:, :, 1] = (y / length + 1) * 0.5
    out[:, :, 2] = (x / length + 1) * 0.5
    return out.astype(np.float32)


def rotate_normal_vectors(
    img: np.ndarray, angle_deg: float, map_type: MapType
) -> np.ndarray:
    """
    Rotates the XY vectors of a normal map by the given angle (counter-clockwise, as the image
    is displayed). This must be done in addition to rotating the pixels, otherwise lighting
    on a rotated patch points the wrong way.
    """
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    x, y = _normal_xy(img, map_type)
    nx = x * cos - y * sin
    ny = x * sin + y * cos
    if map_type == MapType.NORMAL_DX:
        ny = -ny
    out = img.copy()
    out[:, :, 2] = (nx + 1) * 0.5
    out[:, :, 1] = (ny + 1) * 0.5
    return out


def _blend(
    a: np.ndarray,
    b: np.ndarray,
    m: np.ndarray,
    map_type: MapType,
    preserve_contrast: bool,
) -> np.ndarray:
    """`m * a + (1 - m) * b`, with handling that depends on the map type."""
    m = _as3(m)
    if map_type == MapType.MASK:
        m = (m >= 0.5).astype(np.float32)
        return a * m + b * (1 - m)

    if map_type.is_normal:
        return _blend_normals(a, b, m)

    out = a * m + b * (1 - m)

    if preserve_contrast:
        # Variance-preserving blend (Heitz & Neyret 2018): blending two uncorrelated textures
        # reduces their contrast by a factor of sqrt(m^2 + (1-m)^2). Undo that around the mean.
        mean = (a.mean(axis=(0, 1)) + b.mean(axis=(0, 1))) * 0.5
        k = np.sqrt(m * m + (1 - m) * (1 - m))
        out = (out - mean) / k + mean
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------- plans


def _ramp(n: int, width: float) -> np.ndarray:
    """1D mask: 0 at both borders, 1 in the middle."""
    x = np.abs(np.arange(n, dtype=np.float32) + 0.5 - n / 2) / (n / 2)
    return 1 - _smoothstep(1 - max(width, 1e-3), 1.0, x)


def _plan_linear(
    h: int, w: int, x: bool, y: bool, p: LinearParams
) -> list[np.ndarray | None]:
    mx = np.tile(_ramp(w, p.width)[np.newaxis, :], (h, 1)) if x else None
    my = np.tile(_ramp(h, p.width)[:, np.newaxis], (1, w)) if y else None
    return [mx, my]


def _quincunx_mask(guide: np.ndarray, seed: int, p: QuincunxParams) -> np.ndarray:
    """
    Mask for blending the image with a copy shifted by half its size in both directions.

    The original keeps a centred diamond and the shifted copy fills the 4 corners. The
    original's own seam (its border) lies outside the diamond and the copy's seam (the centre
    cross) lies inside it, so only the 4 tips of the diamond can show a seam. Warping the mask
    hides those.
    """
    h, w = guide.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.abs(xx + 0.5 - w / 2) / (w / 2) + np.abs(yy + 0.5 - h / 2) / (h / 2)
    if p.warp > 0:
        d = d + p.warp * _tileable_noise(h, w, p.warp_detail, seed)
    s = max(p.softness, 1e-3)
    m = 1 - _smoothstep(1 - s, 1 + s, d)

    if p.content_influence > 0:
        # Height-blend style: where both layers overlap, the brighter/higher one wins, so the
        # transition follows the content (e.g. the mortar between stones). The bias is scaled
        # by 4m(1-m), so m == 0 and m == 1 never change and the result still tiles.
        g = _luminance(guide)
        diff = g - _roll(g, h // 2, w // 2)
        t = m + diff * (p.content_influence * 4) * 4 * m * (1 - m)
        sharp = _smoothstep(0.5 - s, 0.5 + s, t)
        m = np.where(m <= 0, 0, np.where(m >= 1, 1, sharp)).astype(np.float32)

    return m.astype(np.float32)


def _tip_masks(h: int, w: int, seed: int, p: QuincunxParams) -> list[np.ndarray]:
    """
    Masks for the 4 tips of the quincunx diamond, where the original's seam (its border) and the
    diagonal copy's seam (the centre cross) meet, so neither layer is clean there.

    A copy shifted only vertically has its seams on the left/right border and the horizontal
    centre line, so it is clean at the top/bottom tips (w/2, 0). Likewise, a copy shifted only
    horizontally is clean at the left/right tips (0, h/2). Blending those in hides the tips with
    real, continuous content. Both masks wrap around, so the result still tiles.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Along the border, the diamond mask is non-zero up to (softness + warp) * half-size away
    # from a tip. The patches' fully-opaque core must cover that whole stretch, even after the
    # noise shrinks it by up to 15%.
    reach = (p.softness + p.warp) / 0.85 + 0.03
    ax = min(reach * w / 2, 0.45 * w)
    ay = min(reach * h / 2, 0.45 * h)

    def ellipse(dx: np.ndarray, dy: np.ndarray, rx: float, ry: float, noise_seed: int):
        d = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
        d = d + 0.15 * _tileable_noise(h, w, max(2, p.warp_detail), noise_seed)
        return (1 - _smoothstep(1.0, 1.4, d)).astype(np.float32)

    # top/bottom tips (one patch, wrapping across the horizontal border). It must stay clear of
    # the vertical copy's own seams (left/right border and the horizontal centre line).
    rx_top, ry_top = min(ax, 0.3 * w), min(ay, 0.3 * h)
    dy_top = np.minimum(yy + 0.5, h - yy - 0.5)
    top = ellipse(np.abs(xx + 0.5 - w / 2), dy_top, rx_top, ry_top, seed + 1)
    # left/right tips (one patch, wrapping across the vertical border)
    rx_left, ry_left = min(ax, 0.3 * w), min(ay, 0.3 * h)
    dx_left = np.minimum(xx + 0.5, w - xx - 0.5)
    left = ellipse(dx_left, np.abs(yy + 0.5 - h / 2), rx_left, ry_left, seed + 2)
    return [top, left]


def _dp_path(cost: np.ndarray, fixed_start: int | None) -> np.ndarray:
    """Top-to-bottom minimum-cost path (8-connected). Returns the column index per row."""
    h, bw = cost.shape
    acc = cost.astype(np.float64)
    if fixed_start is not None:
        acc[0] = np.inf
        acc[0, fixed_start] = cost[0, fixed_start]
    back = np.zeros((h, bw), np.int8)
    cols = np.arange(bw)
    inf = np.array([np.inf])
    for y in range(1, h):
        prev = acc[y - 1]
        stacked = np.stack(
            [np.concatenate([inf, prev[:-1]]), prev, np.concatenate([prev[1:], inf])]
        )
        idx = np.argmin(stacked, axis=0)
        acc[y] += stacked[idx, cols]
        back[y] = idx - 1

    path = np.empty(h, np.int64)
    if fixed_start is not None:
        path[-1] = fixed_start
    else:
        path[-1] = int(np.argmin(acc[-1]))
    for y in range(h - 1, 0, -1):
        path[y - 1] = min(max(path[y] + back[y, path[y]], 0), bw - 1)
    return path


def _cyclic_min_cut(cost: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """
    Vertical minimum-cost path within columns [lo, hi) whose first and last rows are in the
    same column, so the cut itself wraps around vertically.
    """
    band = cost[:, lo:hi]
    first = _dp_path(band, None)
    return _dp_path(band, int(first[-1])) + lo


def _seam_mask_x(a: np.ndarray, b: np.ndarray, p: SeamParams) -> np.ndarray:
    """Mask (1 = take `a`) bounded left and right by minimum-error cuts between a and b."""
    w = a.shape[1]
    cost = np.square(a - b)
    q = w // 4
    half = max(1, int(p.search_width * q))
    left = _cyclic_min_cut(cost, max(0, q - half), q + half)
    right = _cyclic_min_cut(cost, 3 * q - half, min(w, 3 * q + half))
    xx = np.arange(w)[np.newaxis, :]
    m = ((xx >= left[:, np.newaxis]) & (xx < right[:, np.newaxis])).astype(np.float32)
    return _wrap_blur(m, p.feather)


def _plan_min_error_seam(guide: np.ndarray, p: SeamParams) -> list[np.ndarray | None]:
    g = _luminance(guide)
    h, w = g.shape
    shifted_x = _roll(g, 0, w // 2)
    mx = _seam_mask_x(g, shifted_x, p)
    # The second pass works on the result of the first pass, transposed.
    gx = g * mx + shifted_x * (1 - mx)
    my = _seam_mask_x(gx.T, _roll(gx, h // 2, 0).T, p).T
    return [mx, np.ascontiguousarray(my)]


def _wide_mask_x(m: np.ndarray, sigma: float) -> np.ndarray:
    """
    A blurred version of an X-pass mask for blending large-scale brightness. It is forced to 0
    at the left/right border (the original's seam) and to 1 at the vertical centre line (the
    shifted copy's seam), so the result still tiles.
    """
    w = m.shape[1]
    wide = np.clip(_wrap_blur(m, sigma), 0, 1)
    x = np.arange(w, dtype=np.float32) + 0.5
    dist_border = np.minimum(x, w - x) / w
    dist_centre = np.abs(x - w / 2) / w
    allow = _smoothstep(0.0, 0.04, dist_border)[np.newaxis, :]
    force = (1 - _smoothstep(0.0, 0.04, dist_centre))[np.newaxis, :]
    return np.maximum(np.minimum(wide, allow), force).astype(np.float32)


def _plan_low_masks(
    masks: list[np.ndarray | None], sigma: float
) -> list[np.ndarray | None]:
    mx, my = masks
    return [
        _wide_mask_x(mx, sigma) if mx is not None else None,
        np.ascontiguousarray(_wide_mask_x(my.T, sigma).T) if my is not None else None,
    ]


def _plan_splats(seed: int, p: SplatParams) -> list[Splat]:
    rng = np.random.default_rng(seed)
    splats: list[Splat] = []
    for _ in range(p.count):
        cx, cy = rng.uniform(0, 1, 2)
        radius = p.size * (1 + rng.uniform(-p.size_variation, p.size_variation))
        sx, sy = rng.uniform(0, 1, 2)
        angle = rng.uniform(-p.rotation_variation, p.rotation_variation)
        splats.append(
            Splat(
                cx=float(cx % 1),
                cy=float(cy % 1),
                sx=float(sx),
                sy=float(sy),
                radius=float(max(radius, 0.005)),
                angle=float(angle),
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return splats


def _plan_guide(guide: np.ndarray) -> np.ndarray:
    h, w, _ = get_h_w_c(guide)
    scale = MAX_PLAN_SIZE / max(h, w)
    if scale < 1:
        guide = cv2.resize(
            guide,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return _as3(guide).astype(np.float32)


def create_plan(
    guide: np.ndarray,
    method: SeamlessMethod,
    seed: int,
    linear: LinearParams = LinearParams(),  # noqa: B008
    quincunx: QuincunxParams = QuincunxParams(),  # noqa: B008
    seam: SeamParams = SeamParams(),  # noqa: B008
    splat: SplatParams = SplatParams(),  # noqa: B008
    preserve_contrast: bool = False,
) -> SeamlessPlan:
    guide = _plan_guide(guide)
    h, w = guide.shape[:2]
    plan = SeamlessPlan(method, preserve_contrast=preserve_contrast)

    if method == SeamlessMethod.EDGES_LINEAR:
        plan.masks = _plan_linear(h, w, True, True, linear)
    elif method == SeamlessMethod.EDGES_LINEAR_X:
        plan.masks = _plan_linear(h, w, True, False, linear)
    elif method == SeamlessMethod.EDGES_LINEAR_Y:
        plan.masks = _plan_linear(h, w, False, True, linear)
    elif method == SeamlessMethod.EDGES_QUINCUNX:
        plan.masks = [
            _quincunx_mask(guide, seed, quincunx),
            *_tip_masks(h, w, seed, quincunx),
        ]
    elif method == SeamlessMethod.MIN_ERROR_SEAM:
        plan.masks = _plan_min_error_seam(guide, seam)
        if seam.low_freq_blend > 0:
            plan.band_sigma = seam.low_freq_blend
            plan.low_masks = _plan_low_masks(
                plan.masks, seam.low_freq_blend * min(h, w) * 2
            )
    elif method == SeamlessMethod.SPLAT_RANDOM:
        plan.masks = [
            _quincunx_mask(guide, seed, quincunx),
            *_tip_masks(h, w, seed, quincunx),
        ]
        plan.splats = _plan_splats(seed, splat)
        plan.splat_softness = splat.softness
        plan.splat_warp = splat.warp
    else:
        raise AssertionError(f"Unknown method {method}")

    return plan


# --------------------------------------------------------------------------- apply


def _wrap_blur_image(img: np.ndarray, sigma: float) -> np.ndarray:
    p = max(1, math.ceil(sigma * 3))
    padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="wrap")
    return _as3(cv2.GaussianBlur(padded, (0, 0), sigma))[p:-p, p:-p]


def _blend_two_band(
    a: np.ndarray,
    b: np.ndarray,
    sharp: np.ndarray,
    wide: np.ndarray,
    sigma: float,
    map_type: MapType,
) -> np.ndarray:
    """
    Fine detail follows the sharp mask, large-scale brightness follows the wide mask
    (two-band version of Burt & Adelson's multi-band blending).
    """
    low_a = _wrap_blur_image(a, sigma)
    low_b = _wrap_blur_image(b, sigma)
    s = _as3(sharp)
    wd = _as3(wide)
    out = (a - low_a) * s + (b - low_b) * (1 - s) + low_a * wd + low_b * (1 - wd)
    out = np.clip(out, 0, 1).astype(np.float32)
    if map_type.is_normal:
        ones = np.ones_like(s)
        return _blend_normals(out, out, ones)
    return out


def _apply_two_pass(plan: SeamlessPlan, img: np.ndarray, map_type: MapType):
    h, w = img.shape[:2]
    mx, my = plan.masks
    low_x, low_y = plan.low_masks if plan.low_masks else (None, None)
    two_band = plan.band_sigma > 0 and map_type != MapType.MASK
    sigma = plan.band_sigma * min(h, w)

    for mask, low, shift in ((mx, low_x, (0, w // 2)), (my, low_y, (h // 2, 0))):
        if mask is None:
            continue
        shifted = _roll(img, *shift)
        sharp = _resize_mask(mask, h, w)
        if two_band and low is not None:
            wide = _resize_mask(low, h, w)
            img = _blend_two_band(img, shifted, sharp, wide, sigma, map_type)
        else:
            img = _blend(img, shifted, sharp, map_type, plan.preserve_contrast)
    return img


def _apply_quincunx(plan: SeamlessPlan, img: np.ndarray, map_type: MapType):
    h, w = img.shape[:2]
    mask, top, left = plan.masks
    assert mask is not None and top is not None and left is not None
    pc = plan.preserve_contrast
    out = _blend(
        img, _roll(img, h // 2, w // 2), _resize_mask(mask, h, w), map_type, pc
    )
    # patch the 4 diamond tips with copies that are seamless there
    out = _blend(_roll(img, h // 2, 0), out, _resize_mask(top, h, w), map_type, pc)
    return _blend(_roll(img, 0, w // 2), out, _resize_mask(left, h, w), map_type, pc)


def _apply_splats(plan: SeamlessPlan, img: np.ndarray, map_type: MapType):
    h, w = img.shape[:2]
    short = min(h, w)
    interpolation = cv2.INTER_NEAREST if map_type == MapType.MASK else cv2.INTER_LINEAR
    source = img  # already tileable, so sampling with wrap-around is seamless
    out = img.copy()

    for sp in plan.splats:
        r_px = sp.radius * short
        rp = max(1, math.ceil(r_px * (1 + plan.splat_warp)))
        size = 2 * rp + 1

        # Inverse map: output pixel p of the patch samples the source at M @ p.
        rotation = cv2.getRotationMatrix2D((rp, rp), sp.angle, 1.0)
        rotation[0, 2] += sp.sx * w - rp
        rotation[1, 2] += sp.sy * h - rp
        patch = cv2.warpAffine(
            source,
            rotation,
            (size, size),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_WRAP,
        )
        patch = _as3(patch)
        if map_type.is_normal and sp.angle != 0:
            # The inverse map rotates the content by -angle on screen.
            patch = rotate_normal_vectors(patch, -sp.angle, map_type)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        dist = np.hypot(xx - rp, yy - rp) / r_px
        if plan.splat_warp > 0:
            dist = dist + plan.splat_warp * _tileable_noise(size, size, 4, sp.seed)
        mask = 1 - _smoothstep(1 - max(plan.splat_softness, 1e-3), 1, dist)

        # Paste with wrap-around, so splats crossing the border keep the image seamless.
        x0 = round(sp.cx * w) - rp
        y0 = round(sp.cy * h) - rp
        ys = (np.arange(size) + y0) % h
        xs = (np.arange(size) + x0) % w
        index = np.ix_(ys, xs)
        out[index] = _blend(patch, out[index], mask, map_type, plan.preserve_contrast)

    return out


def apply_plan(plan: SeamlessPlan, img: np.ndarray, map_type: MapType) -> np.ndarray:
    img = _as3(img).astype(np.float32)
    if map_type.is_normal:
        assert img.shape[2] >= 3, "Normal maps must have at least 3 channels."

    if plan.method in (
        SeamlessMethod.EDGES_LINEAR,
        SeamlessMethod.EDGES_LINEAR_X,
        SeamlessMethod.EDGES_LINEAR_Y,
        SeamlessMethod.MIN_ERROR_SEAM,
    ):
        out = _apply_two_pass(plan, img, map_type)
    elif plan.method == SeamlessMethod.EDGES_QUINCUNX:
        out = _apply_quincunx(plan, img, map_type)
    elif plan.method == SeamlessMethod.SPLAT_RANDOM:
        out = _apply_splats(plan, _apply_quincunx(plan, img, map_type), map_type)
    else:
        raise AssertionError(f"Unknown method {plan.method}")

    return out


# --------------------------------------------------------------------------- pre-pass


def crop_margin_fraction(img: np.ndarray, percent: float) -> np.ndarray:
    """
    Crops `percent`% of the width/height from every side. The margin in pixels is
    `floor(size * percent / 100)`, which must match the node's output type expression.
    """
    if percent <= 0:
        return img
    h, w, _ = get_h_w_c(img)
    mx = math.floor(w * percent / 100)
    my = math.floor(h * percent / 100)
    return img[my : h - my, mx : w - mx]


def equalize_lighting(img: np.ndarray, radius: float) -> np.ndarray:
    """
    Removes low-frequency lighting (e.g. a baked gradient) by dividing out a heavy blur while
    keeping the average color. `radius` is a fraction of the long image side.
    """
    img = _as3(img).astype(np.float32)
    h, w, c = get_h_w_c(img)
    color_channels = min(c, 3)
    color = img[:, :, :color_channels]
    sigma = max(1.0, radius * max(h, w))
    low = _as3(fast_gaussian_blur(color, sigma))
    mean = color.mean(axis=(0, 1))
    corrected = np.clip(color / np.maximum(low, 1e-3) * mean, 0, 1)
    if c > color_channels:
        corrected = np.concatenate([corrected, img[:, :, color_channels:]], axis=2)
    return corrected.astype(np.float32)


def seam_ratio(img: np.ndarray) -> float:
    """
    How visible the wrap-around seam is: the mean jump across the border divided by the mean
    step between the pixels right next to the border. About 1 means seamless.
    """
    img = _as3(img)

    def axis_ratio(
        first: np.ndarray,
        second: np.ndarray,
        last: np.ndarray,
        before_last: np.ndarray,
    ):
        jump = np.abs(first - last).mean()
        inner = 0.5 * (
            np.abs(second - first).mean() + np.abs(last - before_last).mean()
        )
        return jump / max(float(inner), 1e-8)

    rx = axis_ratio(img[:, 0], img[:, 1], img[:, -1], img[:, -2])
    ry = axis_ratio(img[0], img[1], img[-1], img[-2])
    return float(0.5 * (rx + ry))
