"""Prototype: automatic perspective correction (line-based and pattern-based) + guided 4-point."""

from __future__ import annotations

import cv2
import numpy as np


# ------------------------------------------------------------------ shared helpers


def norm_matrix(w: int, h: int) -> np.ndarray:
    """Pixel -> normalized coords (centre at 0, half-size 1)."""
    s = max(w, h) / 2
    return np.array([[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]], np.float64)


def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def jacobian(H, x, y):
    e = 1e-3
    c = apply_h(H, np.array([[x, y], [x + e, y], [x, y + e]]))
    return np.array([(c[1] - c[0]) / e, (c[2] - c[0]) / e]).T  # columns: d/dx, d/dy


def finish(H_norm, w, h, keep_axes=True):
    """
    Given a rectifying homography in normalized coordinates, rescale so that the image centre keeps
    its size (and, if keep_axes, equal X/Y scale there), then return a pixel homography and the
    output size covering the warped image.
    """
    J = jacobian(H_norm, 0, 0)
    if keep_axes:
        sx, sy = np.linalg.norm(J[0]), np.linalg.norm(J[1])  # row norms: output x/y per input
        D = np.diag([1 / sx, 1 / sy, 1])
    else:
        D = np.eye(3) / np.sqrt(abs(np.linalg.det(J)))
        D[2, 2] = 1
    N = norm_matrix(w, h)
    Hn = D @ H_norm
    Hp = np.linalg.inv(N) @ Hn @ N
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    wc = apply_h(Hp, corners)
    x0, y0 = wc.min(0)
    x1, y1 = wc.max(0)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    return T @ Hp, (int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))), apply_h(T @ Hp, corners)


def inner_rect(quad: np.ndarray) -> tuple[int, int, int, int]:
    """Largest-ish axis-aligned rectangle inside a convex quad (TL, TR, BR, BL)."""
    tl, tr, br, bl = quad
    x0 = max(tl[0], bl[0])
    x1 = min(tr[0], br[0])
    y0 = max(tl[1], tr[1])
    y1 = min(bl[1], br[1])
    return int(np.ceil(x0)), int(np.ceil(y0)), int(np.floor(x1)), int(np.floor(y1))


def warp(img, H, size):
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


# ------------------------------------------------------------------ guided: 4 points


def from_corners(img, corners, out_w=None, out_h=None):
    """corners: TL, TR, BR, BL in pixels. Output size defaults to the average edge lengths."""
    c = np.float32(corners)
    if out_w is None:
        out_w = int(round((np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2))
    if out_h is None:
        out_h = int(round((np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2))
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(c, dst), (out_w, out_h), flags=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ automatic: lines


def detect_lines(gray8):
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray8)[0]
    if lines is None:
        return np.zeros((0, 4))
    lines = lines.reshape(-1, 4)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[length > 0.03 * max(gray8.shape)]


def vanishing_point(lines, rng, iters=2000, tol_deg=1.0):
    """RANSAC vanishing point (homogeneous) of a family of segments in normalized coords."""
    p1 = np.c_[lines[:, :2], np.ones(len(lines))]
    p2 = np.c_[lines[:, 2:], np.ones(len(lines))]
    L = np.cross(p1, p2)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    mid = (lines[:, :2] + lines[:, 2:]) / 2
    d = lines[:, 2:] - lines[:, :2]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    length = np.hypot(*(lines[:, 2:] - lines[:, :2]).T)
    tol = np.sin(np.radians(tol_deg))

    def inliers(v):
        # direction from midpoint to the VP (handles points at infinity)
        to = v[:2][None, :] - mid * v[2]
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
        return np.abs(d[:, 0] * to[:, 1] - d[:, 1] * to[:, 0]) < tol

    best, best_score = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(lines), 2, replace=False)
        v = np.cross(L[i], L[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        v /= np.linalg.norm(v)
        score = length[inliers(v)].sum()
        if score > best_score:
            best, best_score = v, score
    inl = inliers(best)
    # least squares refine: v minimizing sum (w * L.v)^2
    A = L[inl] * length[inl, None]
    v = np.linalg.svd(A)[2][-1]
    return v / np.linalg.norm(v), inl


def rectify_lines(img, seed=0):
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    g8 = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sh, sw = g8.shape
    segs = detect_lines(g8)
    N = norm_matrix(sw, sh)
    segs_n = np.c_[apply_h(N, segs[:, :2]), apply_h(N, segs[:, 2:])]
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    vert = np.abs(ang - 90) < 30
    horiz = (ang < 30) | (ang > 150)
    rng = np.random.default_rng(seed)
    vv, iv = vanishing_point(segs_n[vert], rng)
    vh, ih = vanishing_point(segs_n[horiz], rng)
    # projective part: send the vanishing line (through both VPs) to infinity
    l = np.cross(vv, vh)
    l = l / l[2]
    Hp = np.array([[1, 0, 0], [0, 1, 0], [l[0], l[1], 1.0]])
    # affine part: make the two families exactly vertical / horizontal
    dv = Hp @ vv
    dh = Hp @ vh
    dv = dv[:2] / np.linalg.norm(dv[:2])
    dh = dh[:2] / np.linalg.norm(dh[:2])
    if dh[0] < 0:
        dh = -dh
    if dv[1] < 0:
        dv = -dv
    A = np.eye(3)
    A[:2, :2] = np.linalg.inv(np.c_[dh, dv])
    H_norm = A @ Hp
    Hpix, size, quad = finish(H_norm, w, h)
    info = dict(vertical_lines=int(iv.sum()), horizontal_lines=int(ih.sum()), vp_vertical=vv, vp_horizontal=vh)
    return Hpix, size, quad, info, (segs / scale, vert, horiz)


# ------------------------------------------------------------------ automatic: pattern regularity


def lattice_strength(g):
    """Height of the strongest self-match peaks (away from the origin) of the detail image."""
    d = g - cv2.GaussianBlur(g, (0, 0), 4)
    d = d - d.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    f = np.fft.rfft2(d * win)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=g.shape)
    ac = np.fft.fftshift(ac / ac[0, 0])
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    ring = (r > 8) & (r < min(g.shape) / 3)
    vals = ac[ring]
    # local maxima only, mean of the top few
    mx = cv2.dilate(ac.astype(np.float32), np.ones((5, 5), np.uint8))
    peaks = ac[ring & (ac >= mx - 1e-9)]
    top = np.sort(peaks)[-8:] if len(peaks) else np.sort(vals)[-8:]
    return float(top.mean())


def perspective_h(g_, h_):
    return np.array([[1, 0, 0], [0, 1, 0], [g_, h_, 1.0]])


def rectify_pattern(img, search=0.6, analysis=512):
    """Find the perspective terms that make the repeating pattern most regular."""
    h, w = img.shape[:2]
    scale = analysis / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    sh, sw = gs.shape
    N = norm_matrix(sw, sh)
    Ninv = np.linalg.inv(N)
    # evaluate on a central crop of the warped image, so borders don't matter
    cw, ch = int(sw * 0.6), int(sh * 0.6)

    def objective(params):
        H = perspective_h(*params)
        J = jacobian(H, 0, 0)
        D = np.diag([1 / np.linalg.norm(J[0]), 1 / np.linalg.norm(J[1]), 1])
        Hp = Ninv @ D @ H @ N
        wv = cv2.warpPerspective(gs, Hp, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        crop = wv[(sh - ch) // 2 : (sh + ch) // 2, (sw - cw) // 2 : (sw + cw) // 2]
        return lattice_strength(crop)

    # coarse grid, then local refinement
    best = (0.0, 0.0)
    best_v = objective(best)
    base_v = best_v
    for step in (search / 5, search / 15, search / 45):
        improved = True
        grid = np.arange(-2, 3) * step
        while improved:
            improved = False
            for dg in grid:
                for dh in grid:
                    p = (best[0] + dg, best[1] + dh)
                    if max(abs(p[0]), abs(p[1])) > search:
                        continue
                    v = objective(p)
                    if v > best_v + 1e-6:
                        best, best_v, improved = p, v, True
    Hpix, size, quad = finish(perspective_h(*best), w, h)
    return Hpix, size, quad, dict(params=best, strength_before=base_v, strength_after=best_v)
