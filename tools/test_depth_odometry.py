#!/usr/bin/env python3
"""Self-test for the depth odometry.

Renders depth maps of a room interior from known poses and checks that ICP
recovers the motion between them. The recorded-session fixture cannot do this
job: its depth is synthetic and close to flat, and ICP on a flat surface is
degenerate by construction — it would report failure whatever the code did.

    python3 tools/test_depth_odometry.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_odometry import backproject, icp, normals  # noqa: E402

W, H = 256, 192
FX = FY = 210.0
CX, CY = W / 2, H / 2
K = (FX, FY, CX, CY)

# A room, as an axis-aligned box seen from inside. The dimensions are chosen so
# that the side walls, floor and ceiling are all inside the 62.6° x 49.3° view —
# a larger room puts only the back wall in frame, and a single plane cannot
# constrain lateral motion or yaw at all. That is a property of the scene, not
# of the estimator, and getting it wrong makes the test measure nothing.
LO = np.array([-1.2, -1.1, -1.5])
HI = np.array([1.2, 1.1, 2.5])


def render(position, R):
    """Depth map of the room from this pose, by ray-slab intersection."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    d = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d = d @ R.T                                   # into world
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (LO - position) / d
        t2 = (HI - position) / d
        tmax = np.minimum(np.maximum(t1, t2), 1e9).min(axis=-1)
    # Range-limited like the real sensor, and the depth map stores Z, not range.
    z = tmax * (d @ R)[..., 2]
    z[(tmax > 5.0) | ~np.isfinite(tmax)] = np.nan
    return z


def rot(yaw, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def prep(z):
    valid = np.isfinite(z) & (z > 0.1) & (z < 5.0)
    pts = backproject(np.nan_to_num(z), *K)
    nrm, ok = normals(pts, valid)
    return pts, nrm, ok


def case(name, p1, R1, p2, R2, tol_t=0.02, tol_r=1.0, prior_rotation=False):
    a_pts, _, a_ok = prep(render(p1, R1))
    b_pts, b_nrm, b_ok = prep(render(p2, R2))
    # Truth, expressed the same way ICP reports it: target-from-source.
    truth = np.eye(4)
    truth[:3, :3] = R2.T @ R1
    truth[:3, 3] = R2.T @ (p1 - p2)

    prior = None
    if prior_rotation:
        prior = np.eye(4)
        prior[:3, :3] = truth[:3, :3]
    T, frac = icp(a_pts, a_ok, b_pts, b_nrm, b_ok, K, prior=prior)

    et = float(np.linalg.norm(truth[:3, 3] - T[:3, 3]))
    dR = truth[:3, :3].T @ T[:3, :3]
    er = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    ok = et < tol_t and er < tol_r
    print(f"  {name:32} t {et * 100:5.2f} cm   r {er:5.2f}°   "
          f"inliers {frac * 100:3.0f}%   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("recovering known motion from rendered room depth")
    o = np.zeros(3)
    results = [
        case("forward 20 cm", o, rot(0), np.array([0, 0, 0.2]), rot(0)),
        case("sideways 15 cm", o, rot(0), np.array([0.15, 0, 0]), rot(0)),
        case("yaw 5 deg", o, rot(0), o, rot(math.radians(5))),
        case("walk + turn (5 Hz stride)", o, rot(0),
             np.array([0.18, 0.01, 0.05]), rot(math.radians(4), math.radians(2))),
        case("pitch down 3 deg", o, rot(0), o, rot(0, math.radians(3))),
    ]

    # Degeneracy has to be *detected*, not silently mis-solved: sliding along a
    # single flat wall constrains nothing in that plane, and a pipeline that
    # reports a confident wrong answer there is worse than one that fails.
    flat = np.full((H, W), 2.0)
    a_pts, _, a_ok = prep(flat)
    b_pts, b_nrm, b_ok = prep(flat)
    T, _ = icp(a_pts, a_ok, b_pts, b_nrm, b_ok, K)
    slid = float(np.linalg.norm(T[:3, 3]))
    print(f"  {'flat wall stays put (degenerate)':32} "
          f"drift {slid * 100:5.2f} cm            "
          f"{'PASS' if slid < 0.02 else 'FAIL'}")
    results.append(slid < 0.02)

    # Seeding the true rotation must not make things worse. It did on real
    # data, which turned out to be the harness transposing the prior rather
    # than the estimator disliking it.
    results.append(case("rotation prior does not hurt", o, rot(0),
                        np.array([0.18, 0.01, 0.05]),
                        rot(math.radians(4), math.radians(2)),
                        prior_rotation=True))

    print()
    if all(results):
        print("all passed")
        return 0
    print("FAILURES — the odometry is wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
