"""
Prototype: curvature from normal / height maps, roughness cues, and a synthetic roughness model.
Images are float32 in [0, 1], BGR (OpenCV / chaiNNer convention).
"""

from __future__ import annotations

import cv2
import numpy as np

SCALES = (1, 2, 4, 8, 16)


def gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img.reshape(img.shape[:2]).astype(np.float32)


def blur(x: np.ndarray, sigma: float, wrap: bool = True) -> np.ndarray:
    if sigma <= 0:
        return x
    p = int(np.ceil(3 * sigma))
    mode = "wrap" if wrap else "reflect"
    pad = ((p, p), (p, p)) + (((0, 0),) if x.ndim == 3 else ())
    out = cv2.GaussianBlur(np.pad(x, pad, mode=mode), (0, 0), sigma)
    return out[p:-p, p:-p]


def _robust_normalize(c: np.ndarray) -> np.ndarray:
    """Signed curvature -> [-1, 1] using a robust scale (99th percentile of |c|)."""
    s = float(np.percentile(np.abs(c), 99)) + 1e-8
    return np.clip(c / s, -1, 1)


def curvature_from_normal(normal: np.ndarray, scales=SCALES, directx: bool = False) -> np.ndarray:
    """
    Divergence of the normal field (convex > 0), accumulated over several blur scales like
    Substance's "Curvature Smooth". Returns [-1, 1].
    """
    x = normal[:, :, 2] * 2 - 1
    y = normal[:, :, 1] * 2 - 1
    if directx:
        y = -y
    y_down = -y  # image rows point down, OpenGL green points up
    total = np.zeros_like(x)
    for s in scales:
        bx, by = blur(x, s), blur(y_down, s)
        # central differences with wrap-around (textures are usually tileable)
        div = (np.roll(bx, -1, 1) - np.roll(bx, 1, 1)) / 2 + (np.roll(by, -1, 0) - np.roll(by, 1, 0)) / 2
        total += div * s  # larger scales have smaller per-pixel slopes; weight them back up
    return _robust_normalize(total)


def curvature_from_height(height: np.ndarray, scales=SCALES) -> np.ndarray:
    """Negative Laplacian of the height (convex > 0), accumulated over scales. Returns [-1, 1]."""
    h = gray(height) if height.ndim == 3 else height
    total = np.zeros_like(h)
    for s in scales:
        b = blur(h, s)
        lap = np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1) - 4 * b
        total += -lap * s * s  # Laplacian scales with 1/s^2
    return _robust_normalize(total)


def saturation(albedo: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(albedo[:, :, :3], 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]


def detail_energy(img: np.ndarray, sigma: float = 2.0, window: float = 6.0) -> np.ndarray:
    """Local strength of fine detail (local std of a high-pass): busy micro-surface."""
    g = gray(img) if img.ndim == 3 else img
    hp = g - blur(g, sigma)
    return np.sqrt(blur(hp * hp, window))


def features(albedo, normal=None, height=None, ao=None, local_sigma_frac=0.05):
    """All candidate cues, each a 2D float map (bigger value = 'more of it')."""
    L = gray(albedo)
    h, w = L.shape
    f = {
        "luminance": L,
        "local deviation": L - blur(L, local_sigma_frac * max(h, w)),
        "abs deviation": np.abs(L - float(np.median(L))),
        "saturation": saturation(albedo),
        "albedo detail": detail_energy(albedo),
    }
    if normal is not None:
        c = curvature_from_normal(normal)
        f["convex (normal)"] = np.maximum(c, 0)
        f["concave (normal)"] = np.maximum(-c, 0)
        f["curvature (normal)"] = c
    if height is not None:
        hg = gray(height) if height.ndim == 3 else height
        c = curvature_from_height(hg)
        f["curvature (height)"] = c
        f["height"] = hg
        f["height detail"] = detail_energy(hg)
    if ao is not None:
        f["occlusion"] = 1 - gray(ao)
    return f


def synthesize(
    albedo,
    average: float,
    contrast: float,
    weights: dict[str, float],
    feats: dict[str, np.ndarray] | None = None,
    picks: list[tuple[np.ndarray, float, float, float]] = (),
):
    """
    Synthetic roughness:
      1. weighted sum of standardized cues (each cue z-scored, so weights are comparable),
      2. remapped so its mean is `average` and its spread is `contrast` (std),
      3. colour picks: (bgr, roughness, tolerance, softness) override regions softly.
    """
    feats = feats or features(albedo)
    acc = np.zeros(albedo.shape[:2], np.float32)
    for name, wgt in weights.items():
        if wgt == 0 or name not in feats:
            continue
        x = feats[name]
        acc += wgt * (x - x.mean()) / (x.std() + 1e-6)
    if acc.std() > 1e-6:
        acc = (acc - acc.mean()) / acc.std()
    r = average + contrast * acc
    for bgr, value, tol, soft in picks:
        m = color_mask(albedo, bgr, tol, soft)
        r = r * (1 - m) + value * m
    return np.clip(r, 0, 1).astype(np.float32)


def color_mask(albedo, bgr, tolerance=0.15, softness=0.1, blur_sigma=2.0):
    """Soft mask of pixels close to a picked colour (in Lab, on a slightly blurred albedo)."""
    a = blur(np.clip(albedo[:, :, :3], 0, 1).astype(np.float32), blur_sigma)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    ref = cv2.cvtColor(np.array([[bgr]], np.float32), cv2.COLOR_BGR2LAB)[0, 0]
    d = np.linalg.norm(lab - ref, axis=2) / 100.0  # ~[0, 1+]
    t = np.clip((tolerance + softness - d) / max(softness, 1e-6), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


# ------------------------------------------------------------------ preview rendering


def render_sphere(albedo, normal, rough, metal=None, size=320, light=(-0.5, -0.6, 0.62)):
    """
    A small software preview: the material wrapped on a sphere, lit by one key light plus a
    simple sky/ground ambient, GGX specular. Good enough to judge roughness by eye.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    u = (xx + 0.5) / size * 2 - 1
    v = (yy + 0.5) / size * 2 - 1
    r2 = u * u + v * v
    inside = r2 < 1
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    N = np.dstack([u, -v, z])  # sphere normal, y up
    # texture coordinates: longitude/latitude, tiled 2x
    lon = np.arctan2(u, z) / np.pi * 0.5 + 0.5
    lat = np.arcsin(np.clip(-v, -1, 1)) / np.pi + 0.5
    th, tw = albedo.shape[:2]
    mx = ((lon * 2) % 1 * tw).astype(np.float32)
    my = ((1 - lat) * 2 % 1 * th).astype(np.float32)

    def sample(img):
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    alb = sample(albedo[:, :, :3])[:, :, ::-1]  # RGB
    rough_s = sample(rough if rough.ndim == 2 else gray(rough))
    met = sample(metal) if metal is not None else np.zeros_like(rough_s)
    # tangent frame on the sphere (T along longitude, B along latitude)
    T = np.dstack([z, np.zeros_like(z), -u])
    T /= np.linalg.norm(T, axis=2, keepdims=True) + 1e-6
    B = np.cross(N, T)
    tn = sample(normal[:, :, ::-1] * 2 - 1)  # RGB -> xyz (OpenGL)
    n = T * tn[:, :, :1] + B * tn[:, :, 1:2] + N * tn[:, :, 2:3]
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-6

    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    V = np.array([0, 0, 1], np.float32)
    H = (L + V) / np.linalg.norm(L + V)
    ndl = np.clip(n @ L, 0, 1)[..., None]
    ndv = np.clip(n @ V, 1e-3, 1)[..., None]
    ndh = np.clip(n @ H, 0, 1)[..., None]
    a = np.clip(rough_s, 0.04, 1)[..., None] ** 2
    d = a * a / (np.pi * ((ndh * ndh) * (a * a - 1) + 1) ** 2)
    k = (a + 1) ** 2 / 8
    g = (ndl / (ndl * (1 - k) + k + 1e-6)) * (ndv / (ndv * (1 - k) + k))
    f0 = 0.04 * (1 - met[..., None]) + alb * met[..., None]
    fres = f0 + (1 - f0) * (1 - np.clip(H @ V, 0, 1)) ** 5
    spec = d * g * fres / (4 * ndl * ndv + 1e-4)
    diffuse = alb * (1 - met[..., None]) / np.pi
    sky = 0.35 + 0.25 * n[:, :, 1:2]
    color = (diffuse + spec) * ndl * 3.2 + alb * (1 - met[..., None]) * sky * 0.35
    color = color / (1 + color)  # tone map
    color = np.clip(color, 0, 1) ** (1 / 2.2)
    bg = np.full((size, size, 3), 0.18, np.float32)
    out = np.where(inside[..., None], color, bg)
    return out[:, :, ::-1].astype(np.float32)  # back to BGR
