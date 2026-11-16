# Usage: python demo.py <dir containing detile.py> <output dir>
"""Build a synthetic NON-tiling PBR stack, run every method, write previews + seam metrics."""

import sys

import cv2
import numpy as np

sys.path.insert(0, sys.argv[1])
from detile import MapKind, as3, make_seamless_stack, stochastic_hex_tile  # noqa: E402

OUT = sys.argv[2]
N = 512
rng = np.random.default_rng(7)

# --- synthetic cobblestone material (deliberately not periodic + a lighting gradient) ---
pts = rng.uniform(0, N, (90, 2))
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
d = np.full((N, N), 1e9, np.float32)
d2 = np.full((N, N), 1e9, np.float32)
cell = np.zeros((N, N), np.int32)
for i, (px, py) in enumerate(pts):
    di = np.hypot(xx - px, yy - py)
    closer = di < d
    d2 = np.where(closer, d, np.minimum(d2, di))
    cell = np.where(closer, i, cell)
    d = np.where(closer, di, d)
edge = np.clip((d2 - d) / 14, 0, 1)
fbm = sum(cv2.resize(rng.uniform(-1, 1, (s, s)).astype(np.float32), (N, N), interpolation=cv2.INTER_CUBIC) / (s / 4) for s in (4, 8, 16, 32, 64))
height = np.clip(np.sqrt(edge) * 0.8 + fbm * 0.08 + 0.1, 0, 1).astype(np.float32)
tint = rng.uniform(0.75, 1.15, (len(pts), 3)).astype(np.float32)[cell]
light = (0.75 + 0.35 * xx / N)[:, :, None]  # baked lighting gradient -> makes tiling obvious
albedo = np.clip(np.array([0.55, 0.5, 0.45], np.float32) * tint * (0.35 + 0.65 * height[:, :, None]) * light, 0, 1)
gy, gx = np.gradient(height * 6)
nrm = np.dstack([-gx, gy, np.ones_like(gx)])
nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
normal = (nrm * 0.5 + 0.5).astype(np.float32)
rough = np.clip(0.9 - 0.4 * height + fbm * 0.1, 0, 1).astype(np.float32)
# metal/ID style map at HALF resolution to prove mixed-resolution sync
metal = cv2.resize((tint[:, :, 0] > 1.05).astype(np.float32), (N // 2, N // 2), interpolation=cv2.INTER_NEAREST)

stack = {
    "albedo": (albedo, MapKind.COLOR),
    "normal": (normal, MapKind.NORMAL),
    "height": (height, MapKind.COLOR),
    "rough": (rough, MapKind.COLOR),
    "metal": (metal, MapKind.DISCRETE),
}


def seam_ratio(img):
    """|jump across wrap border| / |typical neighbour difference|. ~1 = seamless."""
    img = as3(img)
    jx = np.abs(img[:, 0] - img[:, -1]).mean()
    jy = np.abs(img[0] - img[-1]).mean()
    ix = np.abs(np.diff(img, axis=1)).mean()
    iy = np.abs(np.diff(img, axis=0)).mean()
    return 0.5 * (jx / ix + jy / iy)


def to_bgr8(img):
    img = as3(img)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return (np.clip(img[:, :, ::-1], 0, 1) * 255).astype(np.uint8)


def tile2(img):
    t = to_bgr8(img)
    t = cv2.resize(t, (N // 2, N // 2), interpolation=cv2.INTER_AREA)
    return np.vstack([np.hstack([t, t])] * 2)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


methods = [
    ("original (no fix)", None, {}),
    ("edges_linear", "edges_linear", {}),
    ("edges_quincunx", "edges_quincunx", {}),
    ("quincunx + height-guided", "edges_quincunx", {"height_strength": 1.5}),
    ("min_cut", "min_cut", {}),
    ("splat_random", "splat_random", {"count": 28, "angle_var": 25, "height_strength": 1.5}),
    ("offset_inpaint", "offset_inpaint", {}),
]

rows, report = [], []
for title, method, kw in methods:
    if method is None:
        res = {k: v for k, (v, _) in stack.items()}
    else:
        res = make_seamless_stack(stack, guide=height, method=method, seed=3, **kw)
    ratios = {k: seam_ratio(v) for k, v in res.items()}
    report.append(f"{title:28s} " + "  ".join(f"{k}={r:5.2f}" for k, r in ratios.items()))
    cells = [label(tile2(res[k]), f"{title} | {k} 2x2") for k in ("albedo", "normal", "height", "metal")]
    rows.append(np.hstack(cells))
    if method:
        cv2.imwrite(f"{OUT}/{method}{'_hg' if 'height_strength' in kw and method != 'splat_random' else ''}_albedo.png", to_bgr8(res["albedo"]))

cv2.imwrite(f"{OUT}/contact_sheet.png", np.vstack(rows))
print("\n".join(report))

# --- de-repetition of an already-tileable texture (use the quincunx result as input) ---
tiled = make_seamless_stack(stack, guide=height, method="edges_quincunx", seed=3, height_strength=1.5)
naive = np.tile(as3(tiled["albedo"]), (3, 3, 1))
hexed = stochastic_hex_tile(tiled["albedo"], 3 * N, 3 * N, scale=1.0, seed=11)
hexed_n = stochastic_hex_tile(tiled["normal"], 3 * N, 3 * N, scale=1.0, seed=11, kind=MapKind.NORMAL)
small = lambda a: cv2.resize(to_bgr8(a), (768, 768), interpolation=cv2.INTER_AREA)  # noqa: E731
cv2.imwrite(
    f"{OUT}/stochastic_tiling.png",
    np.hstack([label(small(naive), "plain 3x3 repeat"), label(small(hexed), "stochastic hex (seed 11)"), label(small(hexed_n), "same seed, normal map")]),
)
