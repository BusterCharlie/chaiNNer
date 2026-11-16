import sys, os, cv2, numpy as np
sys.path.insert(0, os.getcwd()); P = sys.argv[1]; sys.path.insert(0, P)
from nodes.impl.seamless import *
from nodes.impl.seamless import _roll, _wrap_blur, _blend, _resize_mask, _smoothstep
from defects import detect
img = np.load(P + "/out/defect_img.npy"); height = np.load(P + "/out/defect_height.npy")
N = img.shape[0]
r = detect(img)
D = np.maximum(r["seam"], r["smooth"]).astype(np.float32)   # visible damage only (not duplicates)
D = cv2.dilate(D, np.ones((7, 7), np.uint8))

def band(n, frac):  # 0 within `frac` of the borders of a length-n axis, 1 elsewhere
    x = np.abs(np.arange(n) + 0.5 - n / 2) / (n / 2)
    return 1 - _smoothstep(1 - 2 * frac, 1 - frac, x)

def prefer_clean(m, d_this, d_other, keep_zero, keep_one, sigma):
    """Push mask m (1 = this layer) toward the layer without defects, never touching the
    zones that guarantee tiling (keep_zero: m must stay 0, keep_one: m must stay 1)."""
    delta = _wrap_blur(np.clip(d_other - d_this, -1, 1), sigma) * 2
    up = np.clip(delta, 0, None) * (1 - keep_zero)
    down = np.clip(-delta, 0, None) * (1 - keep_one)
    return np.clip(m + up - down, 0, 1).astype(np.float32)

def select_plan(guide, method, D, seed=3, sigma=4):
    plan = create_plan(guide, method, seed)
    h, w = D.shape
    if method == SeamlessMethod.EDGES_QUINCUNX:
        yy, xx = np.mgrid[0:h, 0:w]
        border = 1 - np.minimum(band(w, 0.04)[None, :], band(h, 0.04)[:, None])          # original's seam
        cross = np.maximum(np.abs(xx + 0.5 - w / 2) < 0.04 * w, np.abs(yy + 0.5 - h / 2) < 0.04 * h)  # copy's seam
        cross = _wrap_blur(cross.astype(np.float32), 3)
        plan.masks[0] = prefer_clean(plan.masks[0], D, _roll(D, h // 2, w // 2), border, cross, sigma)
    else:  # two-pass (linear / min-error): X then Y
        mx, my = plan.masks
        bx = np.tile(1 - band(w, 0.04)[None, :], (h, 1)); cx = np.roll(bx, w // 2, axis=1)
        mx = prefer_clean(mx, D, _roll(D, 0, w // 2), bx, cx, sigma)
        D1 = mx * D + (1 - mx) * _roll(D, 0, w // 2)
        by = np.tile(1 - band(h, 0.04)[:, None], (1, w)); cy = np.roll(by, h // 2, axis=0)
        my = prefer_clean(my, D1, _roll(D1, h // 2, 0), by, cy, sigma)
        plan.masks = [mx, my]
    return plan

rows = []
for method in (SeamlessMethod.EDGES_QUINCUNX, SeamlessMethod.EDGES_LINEAR, SeamlessMethod.MIN_ERROR_SEAM):
    base = create_plan(img, method, 3); sel = select_plan(img, method, D)
    a = apply_plan(base, img, MapType.COLOR); b = apply_plan(sel, img, MapType.COLOR)
    da = apply_plan(base, D, MapType.COLOR)[:, :, 0]; db = apply_plan(sel, D, MapType.COLOR)[:, :, 0]
    hb = apply_plan(sel, height, MapType.COLOR)[:, :, 0]
    print(f"{method.value:15s} visible damage left: {100*(da>0.5).mean():5.2f}% -> {100*(db>0.5).mean():5.2f}%   seam {seam_ratio(a):.2f} -> {seam_ratio(b):.2f}")
    def lab(im, t):
        im = (np.clip(im if im.ndim == 3 else np.dstack([im] * 3), 0, 1) * 255).astype(np.uint8).copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1); cv2.putText(im, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return im
    ov = lambda im, d: np.where((d > 0.5)[:, :, None], im * 0.4 + np.array([0, 0, 0.6], np.float32), im)
    rows.append(np.hstack([lab(ov(a, da), f"{method.value}: plain (red = damage)"), lab(ov(b, db), "defect-aware layer choice"), lab(hb, "height, same plan")]))
cv2.imwrite(P + "/out/select.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
