import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rough_proto import curvature_from_normal, curvature_from_height, features, synthesize, render_sphere, gray, color_mask

C, OUT = sys.argv[1], sys.argv[2]
def load(a, k, n=512):
    im = cv2.imread(f"{C}/{a}/{k}.jpg", cv2.IMREAD_COLOR)
    return cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
def lab(im, t, w=256):
    im = np.clip(im, 0, 1)
    if im.ndim == 2: im = np.dstack([im] * 3)
    im = cv2.resize((im * 255).astype(np.uint8), (w, round(w * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (w, 18), (0, 0, 0), -1); cv2.putText(im, t, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA); return im

# 1) curvature: normal vs height
rows = []
for a in ("cobblestone_floor_04", "red_brick", "bark_brown_02", "metal_plate"):
    alb, nor, hgt = load(a, "albedo"), load(a, "normal"), gray(load(a, "height"))
    cn, ch = curvature_from_normal(nor), curvature_from_height(hgt)
    corr = np.corrcoef(cn.ravel(), ch.ravel())[0, 1]
    print(f"{a}: curvature normal vs height correlation {corr:.2f}")
    rows.append(np.hstack([lab(alb, a), lab(nor, "normal"), lab(cn * 0.5 + 0.5, "curvature from normal"), lab(ch * 0.5 + 0.5, "curvature from height"),
                           lab(np.maximum(cn, 0), "convex (edges)"), lab(np.maximum(-cn, 0), "concave (cavities)")]))
cv2.imwrite(f"{OUT}/curvature.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

# 2) roughness: Poly Haven vs synthetic, with sphere previews
rows = []
for a, inv in (("cobblestone_floor_04", False), ("metal_plate", False), ("wood_table_001", False)):
    alb, nor, hgt, ao, ref = load(a, "albedo"), load(a, "normal"), gray(load(a, "height")), load(a, "ao"), gray(load(a, "rough"))
    f = features(alb, nor, hgt, ao)
    sgn = -1 if not inv else 1
    lum_only = synthesize(alb, ref.mean(), ref.std(), {"luminance": sgn}, f)
    layered = synthesize(alb, ref.mean(), max(ref.std(), 0.06), {"luminance": sgn, "concave (normal)": 0.6, "occlusion": 0.4, "convex (normal)": -0.6}, f)
    print(f"{a}: PH roughness mean {ref.mean():.2f} std {ref.std():.3f} | brightness-only corr {np.corrcoef(lum_only.ravel(), ref.ravel())[0,1]:.2f}")
    sph = lambda r: render_sphere(alb, nor, r, size=256)
    rows.append(np.hstack([lab(alb, a), lab(ref, "Poly Haven roughness"), lab(lum_only, "synthetic: brightness"), lab(layered, "+ cavity/AO rougher, edges smoother"),
                           lab(sph(ref), "preview: Poly Haven"), lab(sph(lum_only), "preview: brightness"), lab(sph(layered), "preview: layered")]))
cv2.imwrite(f"{OUT}/roughness.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

# 3) colour pick: rust spots on green paint (green_metal_rust)
alb, nor, ref = load("green_metal_rust", "albedo"), load("green_metal_rust", "normal"), gray(load("green_metal_rust", "rough"))
b = cv2.cvtColor(alb, cv2.COLOR_BGR2LAB)[:, :, 2]
rust = alb[b > np.percentile(b, 99.7)].mean(0)  # the most orange pixels: what a user would pick
mask = color_mask(alb, rust, 0.12, 0.05, blur_sigma=1.0)
f = features(alb, nor)
base = synthesize(alb, ref.mean(), ref.std(), {"luminance": -1}, f)
picked = synthesize(alb, ref.mean(), ref.std(), {"luminance": -1}, f, picks=[(rust, 0.95, 0.12, 0.05)])
print(f"green_metal_rust: rust pick covers {mask.mean() * 100:.1f}% of the image")
row = np.hstack([lab(alb, "green_metal_rust"), lab(mask, "pick: rust colour"), lab(base, "brightness only"), lab(picked, "+ rust -> 0.95"), lab(ref, "Poly Haven roughness"),
                 lab(render_sphere(alb, nor, base, size=256), "preview: brightness"), lab(render_sphere(alb, nor, picked, size=256), "preview: with rust pick")])
cv2.imwrite(f"{OUT}/colorpick.jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 85])
