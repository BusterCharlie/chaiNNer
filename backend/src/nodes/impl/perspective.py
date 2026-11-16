"""
Perspective correction for texture photos.

Like `seamless.py`, correction is split in two phases:

1. `plan_*(guide, ...)` computes a resolution-independent `PerspectivePlan` from a guide image.
2. `apply_perspective(plan, img, map_type)` warps any map with that plan at the map's own
   resolution, so every map of a material gets exactly the same correction.

Plans map normalized input coordinates (u = x / width, v = y / height) to output pixels at a
reference scale (the guide's width). A map that is k times wider than the guide gets an output
that is k times larger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..utils.utils import get_h_w_c
from .seamless import MapType, rotate_normal_vectors


class PerspectiveMode(Enum):
    LINES = "lines"
    PATTERN = "pattern"
    CORNERS = "corners"


@dataclass
class PerspectivePlan:
    homography: np.ndarray
    """Maps normalized input coords (u, v, 1) to reference output pixels."""
    ref_width: float
    """Width of the guide in pixels (reference scale of the output)."""
    out_size: tuple[float, float]
    """Output (width, height) in reference pixels."""
    crop: tuple[float, float, float, float] | None = None
    """(x0, y0, x1, y1) in reference output pixels, or None for the full output."""
    info: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def _jacobian(h: np.ndarray, x: float, y: float) -> np.ndarray:
    """2x2 Jacobian of a homography at (x, y). Rows: d(out x), d(out y)."""
    e = 1e-4
    c = _apply_h(h, np.array([[x, y], [x + e, y], [x, y + e]], np.float64))
    return np.array([(c[1] - c[0]) / e, (c[2] - c[0]) / e]).T


def _centred(w: int, h: int) -> np.ndarray:
    """Pixel coords -> coords centred on the image, scaled so the long half-side is 1."""
    s = max(w, h) / 2
    return np.array(
        [[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]], np.float64
    )


def _gray(img: np.ndarray) -> np.ndarray:
    c = get_h_w_c(img)[2]
    if c == 1:
        return img.reshape(img.shape[:2]).astype(np.float32)
    return cv2.cvtColor(img[:, :, :3].astype(np.float32), cv2.COLOR_BGR2GRAY)


def _downscale(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, size / max(h, w))
    if scale < 1:
        img = cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    return img, scale


def _finalize(
    h_centred: np.ndarray, w: int, h: int, crop: bool, info: list[str]
) -> PerspectivePlan:
    """
    Turns a rectifying homography in centred coordinates (see `_centred`) into a plan. The
    result keeps the size of the image centre, with equal X/Y scale there, and covers the whole
    warped image (optionally cropped to the largest valid rectangle).
    """
    jac = _jacobian(h_centred, 0, 0)
    scale = np.diag([1 / np.linalg.norm(jac[0]), 1 / np.linalg.norm(jac[1]), 1])
    n = _centred(w, h)
    pixel = np.linalg.inv(n) @ scale @ h_centred @ n

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    denominators = np.c_[corners, np.ones(4)] @ pixel[2]
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError(
            "The detected perspective is too strong: the horizon would cross the image."
        )
    warped = _apply_h(pixel, corners)
    if np.ptp(warped[:, 0]) * np.ptp(warped[:, 1]) > 16 * w * h:
        raise ValueError(
            "The detected perspective would enlarge the image more than 4x. "
            "Try the Four Corners mode."
        )
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    pixel = shift @ pixel
    quad = _apply_h(pixel, corners)

    # normalized input coords -> pixel input coords
    to_pixels = np.diag([w, h, 1.0])
    plan = PerspectivePlan(
        homography=pixel @ to_pixels,
        ref_width=float(w),
        out_size=(float(x1 - x0), float(y1 - y0)),
        info=info,
    )
    if crop:
        plan.crop = _inner_rect(quad)
    return plan


def _inner_rect(quad: np.ndarray) -> tuple[float, float, float, float]:
    """An axis-aligned rectangle inside a convex quad given as TL, TR, BR, BL."""
    tl, tr, br, bl = quad
    x0 = max(tl[0], bl[0])
    x1 = min(tr[0], br[0])
    y0 = max(tl[1], tr[1])
    y1 = min(bl[1], br[1])
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError("The corrected image has no valid area to crop to.")
    return float(x0), float(y0), float(x1), float(y1)


# --------------------------------------------------------------------------- guided


def plan_corners(
    corners: list[tuple[float, float]],
    width: int,
    height: int,
    crop: bool = True,
) -> PerspectivePlan:
    """
    `corners` are the top-left, top-right, bottom-right and bottom-left corners of a rectangle
    on the surface, as fractions of the image size. The output size follows the average lengths
    of opposite edges, so the rectangle keeps its apparent size.
    """
    c = np.array(corners, np.float64) * [width, height]
    out_w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    out_h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    if out_w < 2 or out_h < 2:
        raise ValueError("The four corners must span an area.")
    src = c.astype(np.float32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    pixel = np.asarray(cv2.getPerspectiveTransform(src, dst), dtype=np.float64)

    # Output covers the whole warped image; the rectangle itself is kept when cropping.
    img_corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], np.float64
    )
    warped = _apply_h(pixel, img_corners)
    if crop:
        x0, y0, x1, y1 = 0.0, 0.0, float(out_w), float(out_h)
    else:
        x0, y0 = warped.min(axis=0)
        x1, y1 = warped.max(axis=0)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    pixel = shift @ pixel
    return PerspectivePlan(
        homography=pixel @ np.diag([width, height, 1.0]),
        ref_width=float(width),
        out_size=(float(x1 - x0), float(y1 - y0)),
        info=[f"Corrected from 4 corners, {out_w:.0f} x {out_h:.0f} px rectangle."],
    )


# --------------------------------------------------------------------------- lines


def _segments(g: np.ndarray) -> np.ndarray:
    g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    detected = cv2.createLineSegmentDetector().detect(g8)[0]
    # OpenCV returns None when no lines are found (the type stubs don't say so)
    if detected is None:  # pyright: ignore[reportUnnecessaryComparison]
        return np.zeros((0, 4), np.float64)
    lines = np.asarray(detected, dtype=np.float64).reshape(-1, 4)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[length > 0.03 * max(g.shape)]


def _vanishing_point(
    segs: np.ndarray, rng: np.random.Generator, tol_deg: float = 1.0
) -> tuple[np.ndarray, int]:
    """RANSAC vanishing point (homogeneous, may be at infinity) of a family of segments."""
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    direction = segs[:, 2:] - segs[:, :2]
    length = np.linalg.norm(direction, axis=1)
    direction /= length[:, np.newaxis]
    tol = math.sin(math.radians(tol_deg))

    def inliers(v: np.ndarray) -> np.ndarray:
        to = v[np.newaxis, :2] - mid * v[2]
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
        cross = direction[:, 0] * to[:, 1] - direction[:, 1] * to[:, 0]
        return np.abs(cross) < tol

    best, best_score = None, -1.0
    for _ in range(min(2000, len(segs) * len(segs))):
        i, j = rng.choice(len(segs), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        v /= np.linalg.norm(v)
        score = float(length[inliers(v)].sum())
        if score > best_score:
            best, best_score = v, score
    assert best is not None
    mask = inliers(best)
    refined = np.linalg.svd(lines[mask] * length[mask, np.newaxis])[2][-1]
    return refined / np.linalg.norm(refined), int(mask.sum())


def plan_lines(guide: np.ndarray, crop: bool = True) -> PerspectivePlan:
    """
    Straightens converging lines: finds the vanishing points of the near-vertical and
    near-horizontal line segments and maps them to infinity, so those lines become parallel and
    axis-aligned. Works for walls, bricks, tiles, panels, planks, ...
    """
    h, w = guide.shape[:2]
    small, _ = _downscale(_gray(guide), 1000)
    sh, sw = small.shape
    segs = _segments(small)
    n = _centred(sw, sh)
    segs_c = np.c_[_apply_h(n, segs[:, :2]), _apply_h(n, segs[:, 2:])]
    angle = (
        np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    )
    vertical = np.abs(angle - 90) < 30
    horizontal = (angle < 30) | (angle > 150)

    rng = np.random.default_rng(0)
    info = []
    families = []
    for name, sel, fallback in (
        ("vertical", vertical, np.array([0.0, 1.0, 0.0])),
        ("horizontal", horizontal, np.array([1.0, 0.0, 0.0])),
    ):
        if sel.sum() >= 4:
            vp, count = _vanishing_point(segs_c[sel], rng)
            info.append(f"{count} {name} line segments used.")
        else:
            vp = fallback
            info.append(f"Not enough {name} lines found, {name} direction left as is.")
        families.append(vp)
    v_vert, v_horiz = families

    # projective part: send the line through both vanishing points to infinity
    horizon = np.cross(v_vert, v_horiz)
    if abs(horizon[2]) < 1e-9:
        projective = np.eye(3)
    else:
        horizon = horizon / horizon[2]
        projective = np.array([[1, 0, 0], [0, 1, 0], [horizon[0], horizon[1], 1.0]])
    # affine part: make the two directions exactly vertical / horizontal
    d_vert = (projective @ v_vert)[:2]
    d_horiz = (projective @ v_horiz)[:2]
    d_vert /= np.linalg.norm(d_vert)
    d_horiz /= np.linalg.norm(d_horiz)
    if d_horiz[0] < 0:
        d_horiz = -d_horiz
    if d_vert[1] < 0:
        d_vert = -d_vert
    basis = np.c_[d_horiz, d_vert]
    if abs(np.linalg.det(basis)) < 0.2:
        raise ValueError("The detected line directions are almost parallel.")
    affine = np.eye(3)
    affine[:2, :2] = np.linalg.inv(basis)
    return _finalize(affine @ projective, w, h, crop, info)


# --------------------------------------------------------------------------- pattern


def _lattice_strength(g: np.ndarray) -> float:
    """Mean height of the strongest self-match peaks of the detail (higher = more regular)."""
    d = g - cv2.GaussianBlur(g, (0, 0), 4)
    d = d - d.mean()
    window = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    spectrum = np.fft.rfft2(d * window)
    ac = np.fft.irfft2(np.abs(spectrum) ** 2, s=g.shape)
    ac = np.fft.fftshift(ac / max(float(ac[0, 0]), 1e-12)).astype(np.float32)
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    ring = (r > 8) & (r < min(g.shape) / 3)
    local_max = ac >= cv2.dilate(ac, np.ones((5, 5), np.uint8)) - 1e-9
    peaks = ac[ring & local_max]
    if len(peaks) == 0:
        return 0.0
    return float(np.sort(peaks)[-8:].mean())


def _axis_lattice(g: np.ndarray, max_deg: float = 15) -> np.ndarray | None:
    """
    The repeats of a pattern closest to the horizontal and vertical directions (as the
    columns of a 2x2 matrix, in pixels), from the strongest self-match peaks within `max_deg`
    of each axis. None if neither is found. A missing one is assumed perpendicular to the other.
    """
    d = g - cv2.GaussianBlur(g, (0, 0), 4)
    d = d - d.mean()
    window = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    ac = np.fft.irfft2(np.abs(np.fft.rfft2(d * window)) ** 2, s=g.shape)
    ac = np.fft.fftshift(ac / max(float(ac[0, 0]), 1e-12)).astype(np.float32)
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    ring = (r > 8) & (r < min(g.shape) / 3)
    local_max = ac >= cv2.dilate(ac, np.ones((5, 5), np.uint8)) - 1e-9
    ys, xs = np.nonzero(ring & local_max & (ac > 0.2))

    best: dict[str, tuple[float, np.ndarray]] = {}
    for y, x in zip(ys, xs, strict=True):
        vec = np.array([x - cx, y - cy], np.float64)
        if vec[0] < 0 or (vec[0] == 0 and vec[1] < 0):
            vec = -vec  # the autocorrelation is symmetric; keep one of each pair
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        axis = (
            "x"
            if abs(angle) <= max_deg
            else "y"
            if abs(abs(angle) - 90) <= max_deg
            else None
        )
        if axis is None:
            continue
        # prefer strong peaks, then short ones (the fundamental, not a multiple)
        score = float(ac[y, x]) - 1e-3 * float(np.hypot(*vec))
        if axis not in best or score > best[axis][0]:
            best[axis] = (score, vec)
    if not best:
        return None

    # sub-pixel refinement of each peak (parabola through neighbours)
    def refine(vec: np.ndarray) -> np.ndarray:
        x, y = round(float(vec[0])) + cx, round(float(vec[1])) + cy
        out = vec.copy()
        for k, (dx, dy) in enumerate(((1, 0), (0, 1))):
            a, b, c = ac[y - dy, x - dx], ac[y, x], ac[y + dy, x + dx]
            denom = a - 2 * b + c
            if abs(denom) > 1e-9:
                out[k] += 0.5 * (a - c) / denom
        return out

    vx = refine(best["x"][1]) if "x" in best else None
    vy = refine(best["y"][1]) if "y" in best else None
    if vy is not None and vy[1] < 0:
        vy = -vy
    if vx is None:
        assert vy is not None
        vx = np.array([vy[1], -vy[0]])
    if vy is None:
        vy = np.array([-vx[1], vx[0]])
    return np.c_[vx, vy]


def plan_pattern(
    guide: np.ndarray,
    max_tilt: float = 0.6,
    deskew: bool = True,
    crop: bool = True,
) -> PerspectivePlan:
    """
    Finds the perspective that makes a repeating pattern (tread plate, tiles, fabric, ...) most
    regular: in a tilted photo the pattern gets denser towards the far side, which blurs its
    self-match peaks. Optionally also rotates so that the pattern repeats along the axes.
    """
    h, w = guide.shape[:2]
    small, _ = _downscale(_gray(guide), 512)
    sh, sw = small.shape
    n = _centred(sw, sh)
    n_inv = np.linalg.inv(n)
    cw, ch = int(sw * 0.6), int(sh * 0.6)

    def corrected(params: tuple[float, float]) -> np.ndarray:
        persp = np.array([[1, 0, 0], [0, 1, 0], [params[0], params[1], 1.0]])
        jac = _jacobian(persp, 0, 0)
        scale = np.diag([1 / np.linalg.norm(jac[0]), 1 / np.linalg.norm(jac[1]), 1])
        pixel = n_inv @ scale @ persp @ n
        return cv2.warpPerspective(
            small,
            pixel,
            (sw, sh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def objective(params: tuple[float, float]) -> float:
        wv = corrected(params)
        y0, x0 = (sh - ch) // 2, (sw - cw) // 2
        return _lattice_strength(wv[y0 : y0 + ch, x0 : x0 + cw])

    best = (0.0, 0.0)
    best_value = before = objective(best)
    for step in (max_tilt / 5, max_tilt / 15, max_tilt / 45):
        improved = True
        while improved:
            improved = False
            for dg in np.arange(-2, 3) * step:
                for dh in np.arange(-2, 3) * step:
                    p = (best[0] + float(dg), best[1] + float(dh))
                    if max(abs(p[0]), abs(p[1])) > max_tilt:
                        continue
                    value = objective(p)
                    if value > best_value + 1e-6:
                        best, best_value, improved = p, value, True

    info = [
        f"Pattern regularity {before:.2f} -> {best_value:.2f} "
        f"(perspective terms {best[0]:+.3f}, {best[1]:+.3f})."
    ]
    persp = np.array([[1, 0, 0], [0, 1, 0], [best[0], best[1], 1.0]])
    jac = _jacobian(persp, 0, 0)
    persp = np.diag([1 / np.linalg.norm(jac[0]), 1 / np.linalg.norm(jac[1]), 1]) @ persp
    if deskew:
        y0, x0 = (sh - ch) // 2, (sw - cw) // 2
        lattice = _axis_lattice(corrected(best)[y0 : y0 + ch, x0 : x0 + cw])
        if lattice is not None:
            # Map the near-horizontal / near-vertical repeats exactly onto the axes, keeping
            # their lengths. This removes rotation and the shear left by an angled shot.
            lengths = np.linalg.norm(lattice, axis=0)
            align = np.eye(3)
            align[:2, :2] = np.diag(lengths) @ np.linalg.inv(lattice)
            rotation = math.degrees(math.atan2(lattice[1, 0], lattice[0, 0]))
            skew = math.degrees(math.atan2(lattice[0, 1], lattice[1, 1])) + rotation
            if abs(rotation) > 0.1 or abs(skew) > 0.1:
                persp = align @ persp
                info.append(
                    f"Aligned the pattern to the axes (rotation {rotation:+.1f}, "
                    f"shear {skew:+.1f} degrees)."
                )
    if best_value < before + 0.02 and len(info) == 1:
        info.append("No clear improvement: is this a repeating pattern?")
    return _finalize(persp, w, h, crop, info)


# --------------------------------------------------------------------------- apply


def output_size(plan: PerspectivePlan, width: int) -> tuple[int, int]:
    k = width / plan.ref_width
    if plan.crop is not None:
        x0, y0, x1, y1 = plan.crop
        return max(1, int((x1 - x0) * k)), max(1, int((y1 - y0) * k))
    return max(1, round(plan.out_size[0] * k)), max(1, round(plan.out_size[1] * k))


def _normal_rotation(pixel_h: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Local rotation of the warp at each output pixel (degrees, counter-clockwise as displayed),
    from a coarse grid.
    """
    inv = np.linalg.inv(pixel_h)
    gx = np.linspace(0, out_w, 33)
    gy = np.linspace(0, out_h, 33)
    angles = np.zeros((len(gy), len(gx)), np.float32)
    for j, y in enumerate(gy):
        for i, x in enumerate(gx):
            jac = _jacobian(pixel_h, *_apply_h(inv, np.array([[x, y]]))[0])
            u, _, vt = np.linalg.svd(jac)
            rot = u @ vt
            # image rows point down, so this angle is clockwise on screen; return CCW
            angles[j, i] = -math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return cv2.resize(angles, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def apply_perspective(
    plan: PerspectivePlan, img: np.ndarray, map_type: MapType
) -> np.ndarray:
    h, w, c = get_h_w_c(img)
    k = w / plan.ref_width
    pixel = np.diag([k, k, 1.0]) @ plan.homography @ np.diag([1 / w, 1 / h, 1.0])
    if plan.crop is not None:
        x0, y0, _, _ = plan.crop
        pixel = np.array([[1, 0, -x0 * k], [0, 1, -y0 * k], [0, 0, 1]]) @ pixel
    out_w, out_h = output_size(plan, w)

    interpolation = cv2.INTER_NEAREST if map_type == MapType.MASK else cv2.INTER_CUBIC
    out = cv2.warpPerspective(
        img.astype(np.float32),
        pixel,
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if out.ndim == 2:
        out = out[:, :, np.newaxis]
    if map_type == MapType.MASK:
        return out.astype(np.float32)
    out = np.clip(out, 0, 1).astype(np.float32)

    if map_type.is_normal and c >= 3:
        # The warp locally rotates the surface in the image plane; rotate the normals with it.
        angles = _normal_rotation(pixel, out_w, out_h)
        mean = float(np.mean(angles))
        if np.abs(angles).max() > 0.5:
            # rotate_normal_vectors takes one angle; apply the mean, then the residual per pixel
            out = rotate_normal_vectors(out, mean, map_type)
            residual = np.radians(angles - mean)
            x = out[:, :, 2] * 2 - 1
            y = out[:, :, 1] * 2 - 1
            if map_type == MapType.NORMAL_DX:
                y = -y
            nx = x * np.cos(residual) - y * np.sin(residual)
            ny = x * np.sin(residual) + y * np.cos(residual)
            if map_type == MapType.NORMAL_DX:
                ny = -ny
            out[:, :, 2] = (nx + 1) * 0.5
            out[:, :, 1] = (ny + 1) * 0.5
        # re-normalize (Z rebuilt from X/Y)
        x = out[:, :, 2] * 2 - 1
        y = out[:, :, 1] * 2 - 1
        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
        out[:, :, 0] = (z + 1) * 0.5
    return out
