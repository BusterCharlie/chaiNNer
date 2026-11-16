"""Prototype: texture features, pattern period detection, and automatic method choice."""

from __future__ import annotations

import cv2
import numpy as np


def gray(img):
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 and img.shape[2] >= 3 else img.reshape(img.shape[:2])


def autocorr(g: np.ndarray) -> np.ndarray:
    """Normalized circular autocorrelation of the detail (high-pass) image, origin at [0,0]."""
    hp = g - cv2.GaussianBlur(g, (0, 0), max(g.shape) / 32)
    hp = hp - hp.mean()
    # window to suppress the (non-periodic) border
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    f = np.fft.rfft2(hp * win)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=g.shape)
    return ac / ac[0, 0]


def _peak(line: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """Most prominent autocorrelation peak in lag lo..hi: its height above the lowest value
    between lag 0 and the peak. A real repeat dips and rises again; smooth content only decays."""
    best, best_prom = 0, 0.0
    for lag in range(lo + 1, min(hi, len(line) - 1)):
        if line[lag] >= line[lag - 1] and line[lag] >= line[lag + 1]:
            prom = float(line[lag] - line[1 : lag + 1].min())
            if prom > best_prom:
                best, best_prom = lag, prom
    return best, best_prom


def _combine(p1: int, s1: float, p2: int, s2: float) -> tuple[int, float]:
    """Two period estimates of one axis (from vertical and horizontal edges). If one is a whole
    multiple of the other, the true period is the larger one (e.g. staggered brick rows)."""
    if p1 and p2:
        a, b = (p1, p2) if p1 >= p2 else (p2, p1)
        ratio = a / b
        if abs(ratio - round(ratio)) < 0.08 and min(s1, s2) > 0.1:
            return a, max(s1, s2)
    return (p1, s1) if s1 >= s2 else (p2, s2)


def periods(g: np.ndarray) -> tuple[int, int, float]:
    h, w = g.shape
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    est = {}
    for name, e in (("gx", gx), ("gy", gy)):
        ac = autocorr(e)
        est[name] = (_peak(ac[0, :], 4, w // 3), _peak(ac[:, 0], 4, h // 3))
    px, sx = _combine(*est["gx"][0], *est["gy"][0])
    py, sy = _combine(*est["gx"][1], *est["gy"][1])
    return px, py, min(sx, sy)


def features(img: np.ndarray) -> dict[str, float]:
    g = gray(img).astype(np.float32)
    h, w = g.shape
    px, py, prom = periods(g)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    wsum = mag.sum() + 1e-8
    # one dominant direction (wood, grass blades): resultant of doubled angles
    directional = float(np.abs((mag * np.exp(2j * theta)).sum()) / wsum)
    # two right-angle directions (bricks, panels, tiles): resultant of quadrupled angles
    rectilinear = float(np.abs((mag * np.exp(4j * theta)).sum()) / wsum)

    low = cv2.GaussianBlur(g, (0, 0), min(h, w) / 40)
    smoothness = float(np.var(low) / (np.var(g) + 1e-8))
    return {"periodicity": prom, "period_x": px, "period_y": py, "directional": directional,
            "rectilinear": rectilinear, "smoothness": smoothness}


def classify(f: dict[str, float]) -> str:
    if f["periodicity"] > 0.15:
        return "regular pattern"
    if f["directional"] > 0.3:
        return "directional"
    if f["smoothness"] > 0.8:
        return "smooth"
    return "organic"


def fit_to_period(img: np.ndarray, px: int, py: int, plan_w: int, plan_h: int) -> np.ndarray:
    """Resample so the image holds an EVEN whole number of pattern repeats in each axis
    (even, so the half-size shift used by the methods is itself a whole number of repeats)."""
    h, w = img.shape[:2]
    kx = max(2, int(round(w / (px * w / plan_w) / 2)) * 2)
    ky = max(2, int(round(h / (py * h / plan_h) / 2)) * 2)
    sx = kx * (px * w / plan_w) / w  # content scale needed
    sy = ky * (py * h / plan_h) / h
    # take the central sx*w x sy*h region and resample it to w x h
    cw, ch = sx * w, sy * h
    m = np.float32([[cw / w, 0, (w - cw) / 2], [0, ch / h, (h - ch) / 2]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
