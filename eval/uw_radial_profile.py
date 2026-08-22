"""Fit the lens-to-lens scale in annuli of the ultra-wide, not nested discs.

A pinhole lens at its logged field of view gives one scale everywhere. A raw
one does not, and the trend with radius is the distortion the PINHOLE
declaration is ignoring.
"""
import math, sys
from pathlib import Path
import cv2, numpy as np

R = Path("/home/myeongcheol/uv_workspace/nav-data-recorder")
sys.path[:0] = [str(R/"eval"), str(R/"tools")]
import wide_fov_probe as W

RINGS = [(0, 300), (300, 600), (600, 900), (900, 1250)]


def annulus_scales(img_uw, img_wide, prescale=W.PRESCALE, equalize=False):
    small = cv2.resize(img_uw, None, fx=prescale, fy=prescale,
                       interpolation=cv2.INTER_AREA)
    if equalize:
        small = W._CLAHE.apply(small); img_wide = W._CLAHE.apply(img_wide)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(small, None)
    k2, d2 = sift.detectAndCompute(img_wide, None)
    if d1 is None or d2 is None:
        return {}
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < W.RATIO * n.distance]
    cx, cy = small.shape[1] / 2.0, small.shape[0] / 2.0
    out = {}
    for lo, hi in RINGS:
        src, dst = [], []
        for m in good:
            p = k1[m.queryIdx].pt
            r = math.hypot(p[0] - cx, p[1] - cy) / prescale
            if lo <= r < hi:
                src.append(p); dst.append(k2[m.trainIdx].pt)
        if len(src) < W.MIN_INLIERS:
            out[(lo, hi)] = (None, len(src))
            continue
        M, mask = cv2.estimateAffinePartial2D(
            np.float32(src).reshape(-1, 1, 2), np.float32(dst).reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=W.RANSAC_PX)
        if M is None:
            out[(lo, hi)] = (None, len(src)); continue
        s = math.hypot(M[0, 0], M[1, 0]) * prescale
        out[(lo, hi)] = (s, int(mask.sum()))
    return out


def profile(sess_dir, n=40, equalize=False, label=""):
    sess = Path(sess_dir)
    acc = {r: [] for r in RINGS}
    inl = {r: 0 for r in RINGS}
    got = 0
    for uw, wide, dt in W.pair_frames(sess, n, 1):
        if abs(dt) > W.MAX_DT_MS:
            continue
        a = cv2.imread(str(sess / uw["file"]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(sess / wide["file"]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        got += 1
        for r, (s, k) in annulus_scales(a, b, equalize=equalize).items():
            if s is not None:
                acc[r].append(s); inl[r] += k
    print(f"  {label or sess.name[-6:]}: {got} pairs")
    vals = []
    for r in RINGS:
        v = acc[r]
        if not v:
            print(f"    r {r[0]:4d}-{r[1]:4d} px   -- too few matches")
            continue
        m = float(np.median(v))
        vals.append((r, m))
        print(f"    r {r[0]:4d}-{r[1]:4d} px   s {m:.4f}   ({len(v)} frames, "
              f"{inl[r]} inliers)")
    if len(vals) >= 2:
        s0, s1 = vals[0][1], vals[-1][1]
        print(f"    spread across annuli: {100*(s1-s0)/s0:+.2f} %  "
              f"({vals[0][0][0]}-{vals[0][0][1]} px -> {vals[-1][0][0]}-{vals[-1][0][1]} px)")
    return vals


print("CONTROL: an ultra-wide frame against itself downscaled — no lens difference,")
print("         so the profile must be flat.")
sess = Path(sys.argv[1] if len(sys.argv) > 1 else
            str(list(Path.home().glob("nav_data/*d06152"))[0]))
uw = W.rows(sess / "frames.jsonl")
img = cv2.imread(str(sess / uw[len(uw)//2]["file"]), cv2.IMREAD_GRAYSCALE)
shrunk = cv2.resize(img, None, fx=0.3200, fy=0.3200, interpolation=cv2.INTER_AREA)
for r, (s, k) in annulus_scales(img, shrunk).items():
    print(f"    r {r[0]:4d}-{r[1]:4d} px   " +
          (f"s {s:.4f}  ({k} inliers)" if s else "-- too few matches"))
print()
print("QUESTION: the two real lenses.")
profile(sess, equalize=True)
