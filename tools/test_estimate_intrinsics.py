#!/usr/bin/env python3
"""Self-test for the focal-length estimator.

Feeds it correspondences projected from a known 3D scene through known poses and
known intrinsics, and checks that it recovers the focal lengths. This exercises
the geometry — the basis change out of FLU, the essential matrix from the pose
pair, the Sampson scoring — which is the part that can be quietly wrong.

Feature detection is deliberately not exercised here; that is OpenCV's job, and
real imagery is what tests it.

    python3 tools/test_estimate_intrinsics.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_intrinsics import FLU_TO_CV, estimate, skew  # noqa: E402


def make_scene(fx, fy, width, height, n_frames=12, n_points=400, seed=0):
    """A camera walking a short arc through a cloud of points.

    Returns pairs shaped like `collect_matches` output, so the estimator sees
    exactly what it would from real imagery.
    """
    rng = np.random.default_rng(seed)
    cx, cy = width / 2.0, height / 2.0

    # Points spread through a room ahead of the walk, at depths that give real
    # parallax rather than a plane at infinity.
    pts = np.column_stack([
        rng.uniform(1.0, 6.0, n_points),     # forward
        rng.uniform(-3.0, 3.0, n_points),    # left
        rng.uniform(-1.0, 1.5, n_points),    # up
    ])

    poses = []
    for i in range(n_frames):
        yaw = math.radians(4.0 * i)
        pitch = math.radians(-25.0 + 2.0 * math.sin(i * 0.7))
        roll = math.radians(2.0 * math.sin(i * 1.1))
        # Walking bob. Without some vertical motion the scene constrains fy only
        # weakly — the estimate then reflects the test, not the estimator.
        p = np.array([0.35 * i, 0.05 * i, 0.02 * math.sin(i * 1.3)])
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        Rz = np.array([[cy_, -sy_, 0.0], [sy_, cy_, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        R_flu = Rz @ Ry @ Rx
        poses.append((p, R_flu @ FLU_TO_CV.T))

    def project(p_world, p_cam, R_cam):
        X = R_cam.T @ (p_world - p_cam)
        if X[2] <= 0.1:
            return None
        return np.array([fx * X[0] / X[2] + cx, fy * X[1] / X[2] + cy])

    pairs = []
    for i in range(len(poses) - 2):
        j = i + 2
        p1, R1 = poses[i]
        p2, R2 = poses[j]
        a, b = [], []
        for pt in pts:
            u1 = project(pt, p1, R1)
            u2 = project(pt, p2, R2)
            if u1 is None or u2 is None:
                continue
            if not (0 <= u1[0] < width and 0 <= u1[1] < height):
                continue
            if not (0 <= u2[0] < width and 0 <= u2[1] < height):
                continue
            a.append(u1)
            b.append(u2)
        if len(a) < 30:
            continue
        R = R2.T @ R1
        t = R2.T @ (p1 - p2)
        pairs.append((np.array(a), np.array(b), skew(t) @ R))
    return pairs


def check(name, fx, fy, width, height, tol=0.03):
    pairs = make_scene(fx, fy, width, height)
    got_fx, got_fy, err = estimate(pairs, width, height)
    ex = abs(got_fx - fx) / fx
    ey = abs(got_fy - fy) / fy
    ok = ex < tol and ey < tol
    print(f"  {name:34} fx {fx:6.1f} -> {got_fx:6.1f} ({ex * 100:4.1f}%)   "
          f"fy {fy:6.1f} -> {got_fy:6.1f} ({ey * 100:4.1f}%)   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("recovering known focal lengths from synthetic correspondences")
    results = [
        # A crop export: square pixels, both focal lengths equal.
        check("crop export (square pixels)", 595.4, 595.4, 848, 480),
        # A whole-frame squeeze: 4:3 into 16:9, so fx/fy lands near 1.33.
        check("whole export (4:3 squeezed)", 595.4, 449.3, 848, 480),
        # The deployment camera's own geometry.
        check("OAK-1-W target", 374.9, 374.9, 848, 480),
        # Full-resolution capture, as a sanity check away from 848x480.
        check("source frame 1920x1440", 1295.0, 1295.0, 1920, 1440),
    ]
    print()
    if all(results):
        print("all passed")
        return 0
    print("FAILURES — the estimator's geometry is wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
