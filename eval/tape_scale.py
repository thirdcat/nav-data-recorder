#!/usr/bin/env python3
"""Absolute scale against a steel tape, which is the only external truth here.

`docs/POSE.md` records why this had to exist: LiDAR is metric but is the thing
being fitted, the accelerometer fails its positive control, and planes, gravity
and loop closure are scale-blind or ARKit-derived. A recorded ruler is the way
out, and this reads it.

    python3 eval/tape_scale.py                 # the sessions with a known tape
    python3 eval/tape_scale.py SESSION --truth 2.0

Isolate the blade as a spatial cluster, not as a colour mask.

Attempt one asked one photograph to show the whole tape and measured a fragment
(0.19 m against 2.000). Attempt two accumulated every yellow surface point and
fitted a line to all of them at once, which fitted the wood grain instead
(3.95 m against 1.000). The missing constraint is that the tape is one
*connected* object a few millimetres wide, and warm-coloured floorboards are
neither connected to it nor thin.

So: gather yellow points, cluster them in space, and score each cluster by what
a tape actually is — long, thin, straight. The winner is reported with its
thinness so the reader can see whether it really was a blade.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
import numpy as np
from PIL import Image
from read_session import Session
from export_3dgs import depth_points
import align_sessions as al

TRUTH = {"20260814-201543-505b2c": 1.000,
         "20260814-201702-e11854": 2.000,
         "20260814-201754-e9d8f7": 2.000}


def yellow_points(S, rows):
    pts = []
    for r in rows:
        p, _, uv = depth_points(S, r, conf_min=2, near=0.2, far=4.0, pix_stride=1)
        if not len(p):
            continue
        im = np.asarray(Image.open(S.frame_path(r)).convert("RGB"), dtype=np.int16)
        h, w = im.shape[:2]
        z = np.asarray(S.depth_frame(r["depth"]))
        u = np.clip((uv[:, 0] * (w / z.shape[1])).astype(int), 0, w - 1)
        v = np.clip((uv[:, 1] * (h / z.shape[0])).astype(int), 0, h - 1)
        c = im[v, u]
        R, G, B = c[:, 0], c[:, 1], c[:, 2]
        sat = (c.max(1) - c.min(1)) / np.maximum(c.max(1), 1)
        keep = (sat > 0.50) & ((R + G) / 2 - B > 95) & (R > 100) & (G > 80)
        if keep.sum():
            pts.append(p[keep])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def clusters(P, link=0.04, min_size=60):
    """Single-link clustering on a voxel grid, breadth first."""
    if not len(P):
        return []
    index = al.NearestVoxel(P, link)
    seen = np.zeros(len(P), bool)
    out = []
    for seed in range(len(P)):
        if seen[seed]:
            continue
        stack, group = [seed], []
        seen[seed] = True
        while stack:
            i = stack.pop()
            group.append(i)
            near, dist = index.query(P[i][None, :] + np.zeros((1, 3)))
            # query returns one neighbour; widen by scanning the cell block
            for d in (-link, 0.0, link):
                for e in (-link, 0.0, link):
                    for f in (-link, 0.0, link):
                        j, dd = index.query((P[i] + np.array([d, e, f]))[None, :])
                        j = int(j[0])
                        if j >= 0 and not seen[j] and np.linalg.norm(P[j] - P[i]) < link:
                            seen[j] = True
                            stack.append(j)
        if len(group) >= min_size:
            out.append(P[np.array(group)])
    return out


for name, truth in TRUTH.items():
    S = Session(f"/home/myeongcheol/nav_data/{name}")
    rows = [r for r in S.posed_images()
            if r.get("depth") is not None and r["pose"].get("tracking") == "normal"]
    P = yellow_points(S, rows)
    print(f"\n{name[-6:]}  truth {truth:.3f} m   {len(P):,} yellow points")
    if len(P) < 100:
        print("   too few to cluster")
        continue
    # Thin the population first: clustering a quarter million points is not the
    # measurement, and the blade survives subsampling.
    if len(P) > 40000:
        P = P[np.random.default_rng(0).choice(len(P), 40000, replace=False)]
    best = None
    for g in clusters(P):
        c = g.mean(0)
        u = np.linalg.svd(g - c, full_matrices=False)[2][0]
        t = (g - c) @ u
        off = np.linalg.norm((g - c) - np.outer(t, u), axis=1)
        length = float(np.percentile(t, 99) - np.percentile(t, 1))
        thick = float(np.percentile(off, 90))
        if thick > 0.05 or length < 0.3:
            continue
        score = length / max(thick, 1e-3)
        if best is None or score > best[0]:
            best = (score, length, thick, len(g))
    if best is None:
        print("   no long thin cluster survived")
        continue
    _, length, thick, n = best
    print(f"   blade {length:.4f} m ({length/truth - 1:+.2%}), "
          f"thickness p90 {thick*100:.1f} cm, {n:,} points")
