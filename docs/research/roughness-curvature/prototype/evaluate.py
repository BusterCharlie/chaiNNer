import os, sys, cv2, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from rough_proto import features

C = sys.argv[1]
def load(a, k, n=512):
    im = cv2.imread(f"{C}/{a}/{k}.jpg", cv2.IMREAD_COLOR)
    return None if im is None else cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

rows, names = [], None
X_all, y_all, groups = [], [], []
for a in sorted(os.listdir(C)):
    alb, nor, hgt, ao, rough = (load(a, k) for k in ("albedo", "normal", "height", "ao", "rough"))
    if any(x is None for x in (alb, nor, hgt, ao, rough)):
        continue
    hgt_g = cv2.cvtColor(hgt, cv2.COLOR_BGR2GRAY)
    f = features(alb, nor, hgt_g, ao)
    r = cv2.cvtColor(rough, cv2.COLOR_BGR2GRAY)
    names = names or list(f.keys())
    ys = r[::2, ::2].ravel()
    corr = [spearmanr(f[k][::2, ::2].ravel(), ys)[0] for k in names]
    rows.append((a, corr, float(r.mean()), float(r.std())))
    Z = np.stack([(f[k] - f[k].mean()) / (f[k].std() + 1e-6) for k in names], -1)[::4, ::4].reshape(-1, len(names))
    X_all.append(Z); y_all.append(((r - r.mean()) / (r.std() + 1e-6))[::4, ::4].ravel()); groups.append(a)

corrs = np.array([c for _, c, _, _ in rows])
print(f"{len(rows)} materials. Rank correlation of each cue with the real roughness:")
print(f"{'cue':22s} {'mean':>6s} {'median |r|':>10s} {'r>0':>5s} {'r<0':>5s}")
for i, k in enumerate(names):
    c = corrs[:, i]
    print(f"{k:22s} {np.nanmean(c):+6.2f} {np.nanmedian(np.abs(c)):10.2f} {np.sum(c > 0.1):5d} {np.sum(c < -0.1):5d}")

# models
def fit(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None); return w
res = []
for i, a in enumerate(groups):
    X, y = X_all[i], y_all[i]
    oracle = np.corrcoef(X @ fit(X, y), y)[0, 1]
    Xo = np.concatenate([X_all[j] for j in range(len(groups)) if j != i]); yo = np.concatenate([y_all[j] for j in range(len(groups)) if j != i])
    fixed = np.corrcoef(X @ fit(Xo, yo), y)[0, 1]
    lum = abs(np.corrcoef(X[:, names.index("luminance")], y)[0, 1])
    res.append((a, oracle, fixed, lum))
res = np.array([r[1:] for r in res])
print("\nPearson correlation with the real roughness (1 = perfect):")
print(f"  brightness only (best sign per material): mean {res[:,2].mean():.2f}")
print(f"  one fixed recipe (fitted on the others):   mean {res[:,1].mean():.2f}   (negative on {np.sum(res[:,1] < 0)} materials)")
print(f"  tuned per material (all cues):             mean {res[:,0].mean():.2f}")
wg = fit(np.concatenate(X_all), np.concatenate(y_all))
print("\nfixed-recipe weights (z-scored cues):", ", ".join(f"{k} {v:+.2f}" for k, v in zip(names, wg)))
print("\nreal roughness: mean of per-material means %.2f, typical std %.2f" % (np.mean([r[2] for r in rows]), np.median([r[3] for r in rows])))
np.save(os.path.join(os.path.dirname(__file__), "corrs.npy"), corrs)
print("\nper material (oracle / fixed / brightness):")
for (a, *_), (o, fx, l) in zip(rows, res):
    print(f"  {a:26s} {o:5.2f} {fx:5.2f} {l:5.2f}")
