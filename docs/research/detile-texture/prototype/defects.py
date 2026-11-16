"""
Prototype: detect internal edit defects in a texture.

Cues (each produces a 0..1 map):
  step   - "1-pixel step" anomaly: the jump between two neighbouring pixels is much larger than
           the jumps right next to it in the same direction. Natural edges in photos are spread
           over a few pixels by optics; hard paste/clone edges are a single-pixel step.
           Long straight runs are boosted (offset crosses, pasted rectangles).
  smooth - local loss of fine grain/noise relative to the rest of the image (heal / soft clone
           / smudge brush leave blurry patches).
  dup    - copy-move: many keypoint matches that share the same displacement (clone stamp).
"""

from __future__ import annotations

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img.reshape(img.shape[0], img.shape[1]).astype(np.float32)


def step_anomaly(img: np.ndarray, run: int = 24, strength: float = 4.0) -> np.ndarray:
    g = _gray(img).astype(np.float32)
    h, w = g.shape
    out = np.zeros((h, w), np.float32)
    for axis in (1, 0):
        d = np.abs(np.diff(g, axis=axis))  # step between pixel i and i+1
        # reference: the steps 2 pixels before/after in the same direction (not the adjacent ones,
        # which a 2-3px natural edge would also raise)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (3, 3)
        dp = np.pad(d, pad, mode="edge")
        sl = lambda o: dp[:, 3 + o : 3 + o + d.shape[1]] if axis == 1 else dp[3 + o : 3 + o + d.shape[0], :]  # noqa: E731
        neighbours = np.maximum.reduce([sl(-2), sl(-1), sl(1), sl(2)])
        noise = np.median(d) + 1e-4
        ratio = (d - neighbours) / (neighbours + 2 * noise)
        score = np.clip(ratio / strength, 0, 1)
        # boost long straight runs ALONG the seam (perpendicular to the step direction)
        k = (run, 1) if axis == 1 else (1, run)
        straight = cv2.blur(score, k[::-1] if axis == 0 else (1, run))
        score = np.maximum(score * 0.5, straight * 1.5)
        full = np.zeros((h, w), np.float32)
        if axis == 1:
            full[:, :-1] = np.maximum(full[:, :-1], score)
            full[:, 1:] = np.maximum(full[:, 1:], score)
        else:
            full[:-1] = np.maximum(full[:-1], score)
            full[1:] = np.maximum(full[1:], score)
        out = np.maximum(out, full)
    return np.clip(out, 0, 1)


def smooth_anomaly(img: np.ndarray, window: int = 15) -> np.ndarray:
    g = _gray(img).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)  # fine grain
    energy = np.sqrt(cv2.blur(hp * hp, (window, window)))
    typical = np.median(energy) + 1e-5
    rel = energy / typical
    return np.clip((0.45 - rel) / 0.3, 0, 1)  # < ~45% of the usual grain = suspicious


def duplicate_regions(img: np.ndarray, min_shift: int = 24, min_votes: int = 5) -> np.ndarray:
    g8 = (np.clip(_gray(img), 0, 1) * 255).astype(np.uint8)
    h, w = g8.shape
    orb = cv2.ORB_create(nfeatures=20000, fastThreshold=3, edgeThreshold=15, patchSize=15)
    kps, des = orb.detectAndCompute(g8, None)
    mask = np.zeros((h, w), np.float32)
    if des is None or len(kps) < 10:
        return mask
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3)
    pairs = []
    for ms in matches:
        for m in ms:
            if m.trainIdx == m.queryIdx or m.distance > 12:
                continue
            p, q = np.array(kps[m.queryIdx].pt), np.array(kps[m.trainIdx].pt)
            if np.linalg.norm(p - q) >= min_shift:
                pairs.append((p, q))
    if not pairs:
        return mask
    # copy-move => many pairs share (almost) the same displacement
    disp = np.array([np.round((q - p) / 4) for p, q in pairs])
    keys, counts = np.unique(disp, axis=0, return_counts=True)
    for key, cnt in zip(keys, counts):
        if cnt < min_votes:
            continue
        sel = np.all(disp == key, axis=1)
        for (p, q), s in zip(pairs, sel):
            if s:
                cv2.circle(mask, tuple(int(v) for v in p), 9, 1.0, -1)
                cv2.circle(mask, tuple(int(v) for v in q), 9, 1.0, -1)
    return cv2.GaussianBlur(mask, (0, 0), 4).clip(0, 1) > 0.15


def detect(img: np.ndarray) -> dict[str, np.ndarray]:
    step = step_anomaly(img)
    smooth = smooth_anomaly(img)
    dup = duplicate_blocks(img)
    seam = cv2.dilate((step > 0.5).astype(np.float32), np.ones((5, 5), np.uint8))
    # remove isolated specks: keep components of a meaningful length
    n, lab, stats, _ = cv2.connectedComponentsWithStats((seam > 0).astype(np.uint8), 8)
    keep = np.zeros_like(seam)
    for i in range(1, n):
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 30:
            keep[lab == i] = 1
    smooth_m = cv2.morphologyEx((smooth > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(np.float32)
    combined = np.maximum.reduce([keep, smooth_m, dup])
    return {"step": step, "seam": keep, "smooth": smooth_m, "dup": dup, "combined": combined}


def duplicate_blocks(img: np.ndarray, block: int = 8, stride: int = 2, min_shift: int = 16, min_votes: int = 25) -> np.ndarray:
    """
    Dense copy-move detection (Fridrich-style). Blocks of the high-pass image are compared after
    lexicographic sorting; near-identical blocks whose displacement is shared by many other block
    pairs are marked. Grain/noise never repeats by chance, so cloned regions stand out.
    """
    g = _gray(img).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    h, w = hp.shape
    ys = np.arange(0, h - block + 1, stride)
    xs = np.arange(0, w - block + 1, stride)
    # all blocks as vectors
    win = np.lib.stride_tricks.sliding_window_view(hp, (block, block))[::stride, ::stride]
    vec = win.reshape(len(ys) * len(xs), block * block)
    energy = np.sqrt((vec * vec).mean(axis=1))
    typical = np.median(energy) + 1e-6
    ok = energy > 0.5 * typical  # ignore flat blocks (they match anything)
    pos = np.stack(np.meshgrid(ys, xs, indexing="ij"), -1).reshape(-1, 2)[ok]
    vec = vec[ok] / energy[ok, None]
    # reduce to a few principal components, quantize, sort
    mean = vec.mean(0)
    _, _, vt = np.linalg.svd((vec - mean)[:: max(1, len(vec) // 5000)], full_matrices=False)
    feat = (vec - mean) @ vt[:8].T
    order = np.lexsort(np.round(feat * 4).T[::-1])
    mask = np.zeros((h, w), np.float32)
    shifts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(1, 4):  # compare with the next few blocks in sorted order
        a, b = order[:-k], order[k:]
        dist = np.sqrt(((vec[a] - vec[b]) ** 2).mean(axis=1))
        d = pos[b] - pos[a]
        far = np.hypot(d[:, 0], d[:, 1]) >= min_shift
        good = (dist < 0.25) & far
        for i, j in zip(a[good], b[good]):
            dy, dx = pos[j] - pos[i]
            if dy < 0 or (dy == 0 and dx < 0):
                i, j, dy, dx = j, i, -dy, -dx
            shifts.setdefault((int(dy) // 2, int(dx) // 2), []).append((int(i), int(j)))
    for key, lst in shifts.items():
        if len(lst) < min_votes:
            continue
        for i, j in lst:
            for p in (pos[i], pos[j]):
                mask[p[0] : p[0] + block, p[1] : p[1] + block] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask
