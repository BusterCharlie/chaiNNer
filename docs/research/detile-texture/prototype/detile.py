"""
Prototype: make a set of (PBR) maps seamlessly tileable with one shared plan.

Every method is split into two phases:
  1. plan(guide, seed, ...)   -> a resolution-independent "TilingPlan"
                                 (masks + splat list), computed ONCE from a guide map.
  2. apply(plan, image, kind) -> the same operation applied to any map of the stack,
                                 at that map's own resolution.

Images are float32 HxWxC in [0, 1] (chaiNNer convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class MapKind(Enum):
    COLOR = "color"  # albedo, roughness, height, AO, ... (linear blend OK)
    NORMAL = "normal"  # tangent-space normal map: rotate vectors + renormalize
    DISCRETE = "discrete"  # IDs / masks / metal: hard (binary) transitions only


# ---------------------------------------------------------------- utilities


def as3(img: np.ndarray) -> np.ndarray:
    return img[:, :, None] if img.ndim == 2 else img


def roll(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(img, (dy, dx), axis=(0, 1))


def luminance(img: np.ndarray) -> np.ndarray:
    img = as3(img)
    if img.shape[2] >= 3:
        return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    return img[:, :, 0]


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def tileable_noise(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Periodic value noise in [-1, 1] (wraps at the borders, so masks stay tileable)."""
    grid = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
    grid = np.pad(grid, ((0, 1), (0, 1)), mode="wrap")
    up = cv2.resize(grid, (w + w // cells, h + h // cells), interpolation=cv2.INTER_CUBIC)
    return up[:h, :w]


def fit(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resample a plan mask to another map's resolution (stacks may mix 2k/4k)."""
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- blending


def blend(a: np.ndarray, b: np.ndarray, m: np.ndarray, kind: MapKind, preserve_contrast: bool):
    """out = m*a + (1-m)*b with map-kind aware handling."""
    m = as3(m)
    if kind == MapKind.DISCRETE:
        m = (m >= 0.5).astype(np.float32)
        return a * m + b * (1 - m)
    if kind == MapKind.NORMAL:
        na, nb = a[:, :, :3] * 2 - 1, b[:, :, :3] * 2 - 1
        n = na * m + nb * (1 - m)
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
        out = n * 0.5 + 0.5
        return np.concatenate([out, a[:, :, 3:] * m + b[:, :, 3:] * (1 - m)], axis=2)
    out = a * m + b * (1 - m)
    if preserve_contrast:
        # Variance-preserving blend (Heitz & Neyret 2018, without the Gaussianisation step):
        # linear blending of two uncorrelated textures drops contrast by sqrt(m^2+(1-m)^2).
        mean = 0.5 * (a.mean(axis=(0, 1)) + b.mean(axis=(0, 1)))
        k = np.sqrt(m * m + (1 - m) * (1 - m))
        out = (out - mean) / k + mean
    return np.clip(out, 0, 1)


def height_blend(m: np.ndarray, ha: np.ndarray, hb: np.ndarray, strength: float, sharp: float):
    """Bias a soft mask by the guide heights so the higher material 'wins' (content-aware seam)."""
    # The bias is scaled by 4m(1-m) so m==0 / m==1 are never changed: the regions that
    # guarantee tileability (borders of the original, seams of the offset copy) stay intact.
    t = m + (ha - hb) * strength * 4 * m * (1 - m)
    return np.where(m <= 0, 0.0, np.where(m >= 1, 1.0, smoothstep(0.5 - sharp, 0.5 + sharp, t))).astype(np.float32)


# ---------------------------------------------------------------- plans


@dataclass
class Splat:
    cx: float  # destination centre, UV in [0,1)
    cy: float
    sx: float  # source centre, UV (kept away from the borders)
    sy: float
    radius: float  # in UV units (relative to min(w,h))
    angle: float  # degrees
    seed: int


@dataclass
class TilingPlan:
    method: str
    # Masks are stored at guide resolution; resampled per map.
    masks: list[np.ndarray] = field(default_factory=list)
    splats: list[Splat] = field(default_factory=list)
    preserve_contrast: bool = False
    splat_softness: float = 0.3
    splat_warp: float = 0.25


# --- 1) Edges Linear (separable X then Y offset-blend) -----------------------------


def _ramp(n: int, width: float) -> np.ndarray:
    """0 at the borders, 1 in the middle; `width` = fraction of half-size used for the fade."""
    x = np.abs(np.arange(n, dtype=np.float32) + 0.5 - n / 2) / (n / 2)  # 0 centre, 1 border
    return 1 - smoothstep(1 - width, 1.0, x)


def plan_edges_linear(guide, width=0.5, axes="xy", **_):
    h, w = guide.shape[:2]
    mx = np.tile(_ramp(w, width)[None, :], (h, 1)) if "x" in axes else None
    my = np.tile(_ramp(h, width)[:, None], (1, w)) if "y" in axes else None
    return TilingPlan("edges_linear", masks=[mx, my])


def apply_edges_linear(plan, img, kind):
    h, w = img.shape[:2]
    mx, my = plan.masks
    if mx is not None:
        img = blend(img, roll(img, 0, w // 2), fit(mx, h, w), kind, plan.preserve_contrast)
    if my is not None:
        img = blend(img, roll(img, h // 2, 0), fit(my, h, w), kind, plan.preserve_contrast)
    return img


# --- 2) Edges Quincunx (diamond mask, original vs. half/half offset copy) -------------


def plan_edges_quincunx(
    guide, seed=0, softness=0.15, warp=0.08, warp_scale=6, height_strength=0.0, **_
):
    """
    The original keeps a centred diamond; the (w/2,h/2)-offset copy fills the 4 corner
    triangles. The original's wrap seam lies only outside the diamond, the copy's seam (centre
    cross) only inside it, so the result tiles. Seams can only leak at the 4 diamond tips, which
    is why the mask is warped (B2M: "warp the mask to adapt it to the image information").
    """
    h, w = guide.shape[:2]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.abs(xx + 0.5 - w / 2) / (w / 2) + np.abs(yy + 0.5 - h / 2) / (h / 2)  # 0 centre, 1 tips
    if warp > 0:
        d = d + warp * tileable_noise(h, w, warp_scale, rng)
    m = 1 - smoothstep(1 - softness, 1 + softness, d)
    if height_strength > 0:
        g = luminance(guide)
        m = height_blend(m, g, roll(g, h // 2, w // 2), height_strength, softness)
    return TilingPlan("edges_quincunx", masks=[m])


def apply_edges_quincunx(plan, img, kind):
    h, w = img.shape[:2]
    return blend(img, roll(img, h // 2, w // 2), fit(plan.masks[0], h, w), kind, plan.preserve_contrast)


# --- 3) Min-error seam (image-quilting style cut, separable) --------------------------


def _min_cut_columns(cost: np.ndarray, lo: int, hi: int, cyclic: bool = True) -> np.ndarray:
    """
    Vertical min-cost path inside columns [lo, hi). Returns the x of the path per row.
    cyclic=True forces first row == last row so the cut itself wraps (needed for tiling).
    """
    path = _dp_path(cost[:, lo:hi].astype(np.float64), None)
    if cyclic:
        path = _dp_path(cost[:, lo:hi].astype(np.float64), int(path[-1]))
    return path + lo


def _dp_path(c: np.ndarray, fixed: int | None) -> np.ndarray:
    h, bw = c.shape
    acc = c.copy()
    if fixed is not None:
        acc[0] = np.inf
        acc[0, fixed] = c[0, fixed]
    back = np.zeros((h, bw), np.int8)
    for y in range(1, h):
        prev = acc[y - 1]
        left = np.r_[np.inf, prev[:-1]]
        right = np.r_[prev[1:], np.inf]
        stack = np.stack([left, prev, right])
        idx = np.argmin(stack, axis=0)
        acc[y] += stack[idx, np.arange(bw)]
        back[y] = idx - 1
    path = np.empty(h, np.int64)
    path[-1] = fixed if fixed is not None else int(np.argmin(acc[-1]))
    for y in range(h - 1, 0, -1):
        path[y - 1] = np.clip(path[y] + back[y, path[y]], 0, bw - 1)
    return path


def _cut_mask_x(a: np.ndarray, b: np.ndarray, band: float, feather: int) -> np.ndarray:
    """Mask (1 = take a) whose left/right boundaries follow min-error cuts between a and b."""
    h, w = a.shape[:2]
    cost = np.sum((as3(a) - as3(b)) ** 2, axis=2)
    q = w // 4
    half = int(band * q)
    left = _min_cut_columns(cost, q - half, q + half)
    right = _min_cut_columns(cost, 3 * q - half, 3 * q + half)
    xx = np.arange(w)[None, :]
    m = ((xx >= left[:, None]) & (xx < right[:, None])).astype(np.float32)
    if feather > 0:  # blur with wrap-around so the feathered mask is itself periodic
        p = 4 * feather
        m = cv2.GaussianBlur(np.pad(m, p, mode="wrap"), (0, 0), feather)[p:-p, p:-p]
    return m


def plan_min_cut(guide, band=0.8, feather=2, **_):
    h, w = guide.shape[:2]
    g = luminance(guide)
    mx = _cut_mask_x(g, roll(g, 0, w // 2), band, feather)
    # Second pass runs on the X-tiled guide, transposed.
    gx = g * mx + roll(g, 0, w // 2) * (1 - mx)
    my = _cut_mask_x(gx.T, roll(gx, h // 2, 0).T, band, feather).T
    return TilingPlan("min_cut", masks=[mx, my])


apply_min_cut = apply_edges_linear  # same two-pass structure, different masks


# --- 4) Splat Random (B2M "Random", formerly "Splat") ---------------------------------


def plan_splat_random(
    guide,
    seed=0,
    count=24,
    radius=0.16,
    radius_var=0.3,
    angle_var=0.0,
    softness=0.35,
    warp=0.25,
    base="edges_quincunx",
    **kw,
):
    """
    Start from a tileable base, then stamp soft round patches cut from the interior of the
    source at random (wrapping) positions. Positions/sources/angles are stored in UV so the
    exact same splats can be replayed on every map of the stack. Changing `seed` gives new
    variations that all tile (B2M "Randomize").
    """
    rng = np.random.default_rng(seed)
    base_plan = PLANNERS[base](guide, seed=seed, **kw)
    splats = []
    # Guarantee coverage of the 4 quincunx tips + border midpoints, then fill randomly.
    anchors = [(0.0, 0.5), (0.5, 0.0), (0.0, 0.0), (0.5, 0.5)]
    for i in range(count):
        if i < len(anchors):
            cx, cy = anchors[i]
            cx += rng.uniform(-0.03, 0.03)
            cy += rng.uniform(-0.03, 0.03)
        else:
            cx, cy = rng.uniform(0, 1, 2)
        r = radius * (1 + rng.uniform(-radius_var, radius_var))
        margin = r * 1.5
        sx, sy = rng.uniform(margin, 1 - margin, 2)
        splats.append(Splat(cx % 1, cy % 1, sx, sy, r, rng.uniform(-angle_var, angle_var), int(rng.integers(1 << 31))))
    plan = TilingPlan("splat_random", masks=base_plan.masks, splats=splats, splat_softness=softness, splat_warp=warp)
    plan.base = base  # type: ignore[attr-defined]
    return plan


def _rotate_normals(patch: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotating the *pixels* of a normal map also requires rotating its XY vectors."""
    a = math.radians(angle_deg)
    n = patch[:, :, :3] * 2 - 1
    x, y = n[:, :, 0].copy(), n[:, :, 1].copy()
    # image rows grow downward; OpenGL green = up -> rotation direction is flipped for Y
    n[:, :, 0] = x * math.cos(a) + y * math.sin(a)
    n[:, :, 1] = -x * math.sin(a) + y * math.cos(a)
    return np.concatenate([n * 0.5 + 0.5, patch[:, :, 3:]], axis=2)


def apply_splat_random(plan, img, kind):
    img = APPLIERS[plan.base](plan, img, kind)  # type: ignore[attr-defined]
    h, w = img.shape[:2]
    s = min(h, w)
    src_full = img  # splats sample the (already tileable) base so patches near edges are fine too
    out = img.copy()
    for sp in plan.splats:
        rp = int(math.ceil(sp.radius * s * (1 + plan.splat_warp)))
        size = 2 * rp + 1
        # source patch (rotated around its centre)
        sx, sy = sp.sx * w, sp.sy * h
        M = cv2.getRotationMatrix2D((rp, rp), sp.angle, 1.0)
        M[0, 2] += sx - rp
        M[1, 2] += sy - rp
        patch = cv2.warpAffine(
            src_full, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP
        )
        patch = as3(patch)
        if kind == MapKind.NORMAL and sp.angle != 0:
            patch = _rotate_normals(patch, sp.angle)
        # soft, noise-warped round mask (deterministic per splat => identical across maps)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        r = np.hypot(xx - rp, yy - rp) / (sp.radius * s)
        if plan.splat_warp > 0:
            r = r + plan.splat_warp * tileable_noise(size, size, 4, np.random.default_rng(sp.seed))
        m = 1 - smoothstep(1 - plan.splat_softness, 1, r)
        # paste with wrap-around (toroidal) so splats crossing the border keep it seamless
        x0, y0 = int(round(sp.cx * w)) - rp, int(round(sp.cy * h)) - rp
        ys = (np.arange(size) + y0) % h
        xs = (np.arange(size) + x0) % w
        region = out[np.ix_(ys, xs)]
        out[np.ix_(ys, xs)] = blend(patch, region, m, kind, plan.preserve_contrast)
    return out


# --- 5) Offset + inpaint (OpenCV Telea/NS over the centre cross) ---------------------


def plan_offset_inpaint(guide, width=0.02, **_):
    h, w = guide.shape[:2]
    m = np.zeros((h, w), np.float32)
    bw, bh = max(2, int(width * w)), max(2, int(width * h))
    m[:, w // 2 - bw : w // 2 + bw] = 1
    m[h // 2 - bh : h // 2 + bh, :] = 1
    return TilingPlan("offset_inpaint", masks=[m])


def apply_offset_inpaint(plan, img, kind):
    h, w = img.shape[:2]
    r = roll(img, h // 2, w // 2)
    m = (fit(plan.masks[0], h, w) > 0.5).astype(np.uint8)
    u8 = (np.clip(as3(r), 0, 1) * 255).astype(np.uint8)
    chans = [cv2.inpaint(u8[:, :, c], m, 5, cv2.INPAINT_TELEA) for c in range(u8.shape[2])]
    out = np.dstack(chans).astype(np.float32) / 255
    if kind == MapKind.NORMAL:
        n = out[:, :, :3] * 2 - 1
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
        out[:, :, :3] = n * 0.5 + 0.5
    return out


PLANNERS = {
    "edges_linear": plan_edges_linear,
    "edges_quincunx": plan_edges_quincunx,
    "min_cut": plan_min_cut,
    "splat_random": plan_splat_random,
    "offset_inpaint": plan_offset_inpaint,
}
APPLIERS = {
    "edges_linear": apply_edges_linear,
    "edges_quincunx": apply_edges_quincunx,
    "min_cut": apply_min_cut,
    "splat_random": apply_splat_random,
    "offset_inpaint": apply_offset_inpaint,
}


def make_seamless_stack(
    maps: dict[str, tuple[np.ndarray, MapKind]], guide: np.ndarray, method: str, **kw
) -> dict[str, np.ndarray]:
    preserve = kw.pop("preserve_contrast", False)
    plan = PLANNERS[method](guide, **kw)
    plan.preserve_contrast = preserve
    return {k: APPLIERS[method](plan, as3(img).astype(np.float32), kind) for k, (img, kind) in maps.items()}


# --- 6) Stochastic re-tiling (de-repetition of an ALREADY tileable texture) ----------


def stochastic_hex_tile(
    img: np.ndarray, out_h: int, out_w: int, scale: float, seed: int, kind=MapKind.COLOR, preserve_contrast=True
) -> np.ndarray:
    """
    Heitz-Neyret / Mikkelsen-style hex tiling baked to an image: each vertex of a triangle
    grid gets a random UV offset; every output pixel blends 3 lookups with barycentric weights.
    Same seed -> same offsets for every map of the stack.
    """
    img = as3(img).astype(np.float32)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    u, v = xx / w * scale, yy / h * scale  # uv in "tiles"
    # skewed (triangle) grid
    sq3 = math.sqrt(3)
    su = u + v / sq3 * 1.0
    sv = 2 * v / sq3
    i0, j0 = np.floor(su), np.floor(sv)
    fu, fv = su - i0, sv - j0
    upper = (fu + fv) > 1
    # 3 vertices + barycentric weights
    v1 = (np.where(upper, i0 + 1, i0), np.where(upper, j0 + 1, j0))
    v2 = (i0 + 1, j0)
    v3 = (i0, j0 + 1)
    w1 = np.where(upper, fu + fv - 1, 1 - fu - fv)
    w2 = np.where(upper, 1 - fv, fu)
    w3 = np.where(upper, 1 - fu, fv)
    # sharpen weights (Mikkelsen's exponent trick) so most pixels come from 1 sample
    ws = np.stack([w1, w2, w3]) ** 6
    ws /= ws.sum(axis=0, keepdims=True)

    def hash2(i, j):
        k = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ (seed * 83492791)
        k = (k ^ (k >> 13)) * 0x5BD1E995 & 0xFFFFFFFF
        return (k & 0xFFFF) / 65535.0, ((k >> 16) & 0xFFFF) / 65535.0

    acc = np.zeros((out_h, out_w, img.shape[2]), np.float32)
    acc2 = np.zeros_like(acc)
    for (vi, vj), wgt in zip((v1, v2, v3), ws):
        ox, oy = hash2(vi, vj)
        mx = (((u + ox) % 1) * w).astype(np.float32)
        my = (((v + oy) % 1) * h).astype(np.float32)
        s = as3(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
        acc += s * wgt[:, :, None]
        acc2 += wgt[:, :, None] ** 2
    if kind == MapKind.NORMAL:
        n = acc[:, :, :3] * 2 - 1
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
        acc[:, :, :3] = n * 0.5 + 0.5
    elif preserve_contrast:
        mean = img.mean(axis=(0, 1))
        acc = (acc - mean) / np.sqrt(acc2) + mean
    return np.clip(acc, 0, 1)


# --- pre-pass: remove low-frequency lighting (B2M "Light Equalizer"-like) -------------


def equalize_low_freq(img: np.ndarray, radius_frac: float = 0.15, amount: float = 1.0) -> np.ndarray:
    """Divide out a heavy blur (multiplicative delight), keeping the global mean."""
    img = as3(img).astype(np.float32)
    h, w = img.shape[:2]
    sigma = radius_frac * max(h, w)
    low = as3(cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT))
    mean = img.mean(axis=(0, 1))
    corrected = img / np.maximum(low, 1e-3) * mean
    return np.clip(img + (corrected - img) * amount, 0, 1)
