"""Does a walk come back, in the only terms SfM cares about: matched features.

`docs/3DGS.md` now says COLMAP is worth one run on a MultiCam session *with
revisit structure*, and leaves the check undone. This is that check, and it
deliberately does not read a trajectory: Pi3X's is the thing under suspicion,
and a loop measured from a drifting chain is measuring the drift.

Instead, count geometrically verified ORB matches between frames far apart in
TIME. Near pairs give the scale — that is what sequential matching already gets.
A far pair matching at a comparable rate is a loop closure COLMAP would find.
The control is a pair drawn from a different session, which must return nothing.
"""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, "/home/myeongcheol/uv_workspace/nav-data-recorder/tools")
from read_session import Session

orb = cv2.ORB_create(nfeatures=4000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def feats(path, width=640):
    """Both arms must be read at the SAME width.

    The wide arm is recorded at 640x480 and the ultra-wide at 3840x2160, so
    reading each at its own size compares resolutions and calls it revisit.
    640 is the only width both can reach.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[1] > width:
        s = width/img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0]*s)))
    return orb.detectAndCompute(img, None)


def inliers(a, b):
    (k1, d1), (k2, d2) = a, b
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    m = bf.match(d1, d2)
    if len(m) < 8:
        return 0
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 3.0, 0.99)
    return int(mask.sum()) if mask is not None else 0


def run(path, lens_file="frames_wide.jsonl", control_path=None, near=2, far_min=40, cap=90):
    s = Session(path)
    name = lens_file.replace("frames", "").replace(".jsonl", "").strip("_") or "main"
    rows = [r for r in s.stream(lens_file.replace(".jsonl", ""))]
    if not rows:
        return None
    step = max(1, len(rows)//cap)
    idx = list(range(0, len(rows), step))
    F = {}
    for i in idx:
        f = feats(os.path.join(path, rows[i]["file"]))
        if f is not None:
            F[i] = f
    ks = sorted(F)
    nearv, farv, pairs = [], [], []
    for a_i, a in enumerate(ks):
        for b in ks[a_i+1:]:
            n = inliers(F[a], F[b])
            if b - a <= near*step:
                nearv.append(n)
            elif b - a >= far_min:
                farv.append(n)
                pairs.append((a, b, n))
    ctrl = []
    if control_path:
        cs = Session(control_path)
        crows = [r for r in cs.stream("frames")][:40]
        for r in crows[::8]:
            f = feats(os.path.join(control_path, r["file"]))
            if f is None:
                continue
            for a in ks[::10]:
                ctrl.append(inliers(F[a], f))
    pairs.sort(key=lambda p: -p[2])
    return name, len(ks), nearv, farv, ctrl, pairs[:5]


if __name__ == "__main__":
    tgt = "/home/myeongcheol/nav_data/20260820-211848-d06152"
    ctrl = "/home/myeongcheol/nav_data/20260809-074458-cb4586"
    for lens in ("frames_wide.jsonl", "frames.jsonl"):
        r = run(tgt, lens, control_path=ctrl)
        if r is None:
            print(f"{lens}: none"); continue
        name, n, nearv, farv, c, top = r
        print(f"\nd06152 {name}: {n} frames sampled")
        print(f"  adjacent pairs   median {np.median(nearv):7.0f}  n={len(nearv)}")
        print(f"  distant pairs    median {np.median(farv):7.0f}  p95 {np.percentile(farv,95):6.0f}  n={len(farv)}")
        if c:
            print(f"  other session    median {np.median(c):7.0f}  p95 {np.percentile(c,95):6.0f}  n={len(c)}")
        print(f"  strongest distant pairs: " + ", ".join(f"{a}-{b}:{v}" for a,b,v in top))
        thresh = max(np.percentile(c,95) if c else 0, 20)
        print(f"  distant pairs above the control's p95 ({thresh:.0f}): "
              f"{100*np.mean(np.array(farv)>thresh):.1f}%")
