#!/usr/bin/env python3
"""Estimate pose from LiDAR depth alone, and score it against ARKit's.

The ultra-wide path needs a pose source that is not ARKit, and depth odometry is
the obvious candidate: it is metric by construction, so the scale problem that
dogs monocular SfM does not arise, and it does not care whether a wall has any
texture on it.

Whether it is *good enough* is a measurement, not an argument — and it can be
made right now, because every recorded session already carries LiDAR depth and
ARKit's own trajectory for the same frames. This runs point-to-plane ICP over
the depth maps and reports how far the result drifts from ARKit.

    python3 tools/depth_odometry.py ~/nav_data/20260807-140619-477750

What it cannot tell you: ARKit is a reference, not ground truth — it drifts
about 0.02 m/s itself. Agreement means the two make the same journey, not that
either is right. Disagreement is still decisive in the direction that matters,
because a depth-only track that cannot match a fused one will not beat it.

Requires numpy.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402


# ------------------------------------------------------------------ geometry

def backproject(depth: np.ndarray, fx: float, fy: float,
                cx: float, cy: float) -> np.ndarray:
    """Depth map to a 3-D point per pixel, in camera coordinates (X right, Y
    down, Z forward — the depth map's own convention, not ARKit's)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def normals(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface normals from neighbouring pixels.

    Point-to-plane ICP converges far faster than point-to-point because it lets
    a point slide along the surface it landed on instead of pinning it to one
    correspondence — which matters here, where correspondences come from
    projection and are approximate by construction.
    """
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1, :] = points[2:, :] - points[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = valid.copy()
    ok[:, :1] = ok[:, -1:] = False
    ok[:1, :] = ok[-1:, :] = False
    ok &= norm[..., 0] > 1e-9
    return np.divide(n, np.maximum(norm, 1e-12)), ok


def icp(src_pts, src_ok, dst_pts, dst_normals, dst_ok, K, iters=20,
        max_dist=0.15, prior=None):
    """Point-to-plane ICP with projective association.

    Correspondences come from projecting a transformed source point into the
    target's image and taking whatever pixel it lands on. That is the
    KinectFusion trick: it is O(1) per point with no spatial index, and on
    depth maps — which are already an image — it is what the data is shaped for.

    Returns the 4x4 transform taking source camera coordinates into target
    camera coordinates, plus the fraction of points that found a match.
    """
    fx, fy, cx, cy = K
    h, w = dst_ok.shape
    T = np.eye(4) if prior is None else prior.copy()

    p = src_pts[src_ok]
    if len(p) < 100:
        return T, 0.0

    # Subsample: ICP does not need every pixel, and this keeps a session's worth
    # of frames to seconds rather than minutes.
    if len(p) > 8000:
        p = p[np.random.default_rng(0).choice(len(p), 8000, replace=False)]

    inlier_frac = 0.0
    for _ in range(iters):
        q = p @ T[:3, :3].T + T[:3, 3]
        z = q[:, 2]
        ok = z > 0.1
        u = np.full(len(q), -1.0)
        v = np.full(len(q), -1.0)
        u[ok] = fx * q[ok, 0] / z[ok] + cx
        v[ok] = fy * q[ok, 1] / z[ok] + cy
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        ok &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if ok.sum() < 100:
            break
        ui, vi = np.clip(ui, 0, w - 1), np.clip(vi, 0, h - 1)
        ok &= dst_ok[vi, ui]

        target = dst_pts[vi, ui]
        normal = dst_normals[vi, ui]
        diff = target - q
        ok &= np.linalg.norm(diff, axis=-1) < max_dist
        if ok.sum() < 100:
            break
        inlier_frac = float(ok.mean())

        qa, na = q[ok], normal[ok]
        # Linearised point-to-plane residual: for a small rotation w and
        # translation t, minimise sum over points of
        # ((q + w x q + t - target) . n)^2. The Jacobian row is [q x n, n].
        A = np.hstack([np.cross(qa, na), na])
        b = np.einsum("ij,ij->i", (target[ok] - qa), na)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        wx, wy, wz = x[:3]
        dR = np.array([[1, -wz, wy], [wz, 1, -wx], [-wy, wx, 1]])
        # Re-orthonormalise: the small-angle matrix above is not a rotation, and
        # composing dozens of them without this walks the estimate off SO(3).
        u_, _, vt_ = np.linalg.svd(dR)
        dR = u_ @ vt_
        step = np.eye(4)
        step[:3, :3] = dR
        step[:3, 3] = x[3:]
        T = step @ T
        if np.linalg.norm(x[:3]) < 1e-6 and np.linalg.norm(x[3:]) < 1e-6:
            break
    return T, inlier_frac


# ------------------------------------------------------------------ scoring

def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ARKit's camera looks down -Z with +Y up; the depth map's own frame is +Z
# forward, +Y down. Columns are the depth axes in ARKit camera coordinates.
ARKIT_TO_DEPTH = np.diag([1.0, -1.0, -1.0])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth depth frame; larger means bigger motion "
                         "between frames, which is where ICP breaks")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-dist", type=float, default=0.15,
                    help="metres; correspondences further apart are rejected")
    ap.add_argument("--rotation-prior", action="store_true",
                    help="seed ICP with ARKit's frame-to-frame rotation and let "
                         "it solve translation from there — the architecture a "
                         "real system uses, where the gyro carries rotation")
    args = ap.parse_args(argv)

    session = Session(args.session)
    index = session.depth_index()
    if not index:
        print("! no depth in this session", file=sys.stderr)
        return 1

    poses = {p["frame"]: p for p in session.poses()}
    entries = [e for e in index[::args.stride] if e.get("frame") in poses]
    entries = entries[:args.limit]
    if len(entries) < 5:
        print("! too few depth frames with matching poses", file=sys.stderr)
        return 1

    span = index[-1]["t"] - index[0]["t"]
    rate = (len(index) - 1) / span if span > 0 else 0.0
    print(f"{session.id}: {len(entries)} depth frames, stride {args.stride}")
    print(f"  depth arrived at {rate:.1f} Hz")
    if rate < 12:
        print(f"  ! that is the old 5 Hz depth gate. ICP converges on small "
              f"motion, so this is the wrong data to judge it on — reinstall "
              f"and re-record.")

    est = [np.eye(4)]           # depth-odometry poses, world-from-camera
    ref = []                    # ARKit, same frames, in the depth convention
    rel_err_t, rel_err_r, inliers, moved = [], [], [], []

    prev = None
    for i, entry in enumerate(entries):
        depth = np.asarray(session.depth_frame(entry), dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0.1) & (depth < 5.0)
        p = poses[entry["frame"]]
        # Intrinsics are quoted for the full-resolution colour frame; the depth
        # map is a fraction of that size and shares the optical axis. `cx` sits
        # within a pixel or two of half the colour width, which is a more robust
        # way to recover the ratio than joining another stream for it.
        s = depth.shape[1] / (2.0 * p["cx"])
        K = (p["fx"] * s, p["fy"] * s, p["cx"] * s, p["cy"] * s)

        pts = backproject(depth, *K)
        nrm, ok = normals(pts, valid)

        R = quat_to_matrix(p["qx"], p["qy"], p["qz"], p["qw"]) @ ARKIT_TO_DEPTH
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = [p["tx"], p["ty"], p["tz"]]
        ref.append(A)

        if prev is not None:
            # ICP solves target-from-source; the trajectory needs its inverse.
            truth = np.linalg.inv(ref[-2]) @ ref[-1]
            prior = np.eye(4)
            if args.rotation_prior:
                # Rotation only. Over 0.2 s a gyro is essentially drift-free, so
                # taking it from ARKit stands in for an IMU without smuggling in
                # the translation that is the thing under test.
                # Transposed: `truth` is previous-from-current, and ICP works
                # in current-from-previous. Seeding it the other way round
                # starts the solve at twice the wrong rotation, which is worse
                # than starting at identity — and looked like evidence against
                # the method rather than a bug in the harness.
                prior[:3, :3] = truth[:3, :3].T
            T, frac = icp(prev["pts"], prev["ok"], pts, nrm, ok, K,
                          max_dist=args.max_dist, prior=prior)
            est.append(est[-1] @ np.linalg.inv(T))
            inliers.append(frac)

            got = np.linalg.inv(T)
            rel_err_t.append(float(np.linalg.norm(truth[:3, 3] - got[:3, 3])))
            moved.append(float(np.linalg.norm(truth[:3, 3])))
            dR = truth[:3, :3].T @ got[:3, :3]
            rel_err_r.append(math.degrees(
                math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))))
        prev = {"pts": pts, "ok": ok}

    est_p = np.array([T[:3, 3] for T in est])
    ref_p = np.array([T[:3, 3] for T in ref])
    travelled = float(np.linalg.norm(np.diff(ref_p, axis=0), axis=1).sum())
    duration = entries[-1]["t"] - entries[0]["t"]

    print(f"  ARKit travelled {travelled:.2f} m over {duration:.1f} s")
    print(f"  ICP inliers: median {np.median(inliers) * 100:.0f}% "
          f"(min {min(inliers) * 100:.0f}%)")
    print()
    print("relative pose error, frame to frame — the honest measure for "
          "odometry")
    print(f"  translation  median {np.median(rel_err_t) * 100:.1f} cm   "
          f"p90 {np.percentile(rel_err_t, 90) * 100:.1f} cm")
    print(f"  rotation     median {np.median(rel_err_r):.2f}°   "
          f"p90 {np.percentile(rel_err_r, 90):.2f}°")
    # Without this the translation error is a number rather than a verdict:
    # reporting "did not move" scores exactly the distance actually moved, so
    # anything above that line is worse than no estimate at all.
    baseline = float(np.median(moved))
    print(f"  the frames are {baseline * 100:.1f} cm apart (median), so "
          f"'assume no motion' scores {baseline * 100:.1f} cm")
    if np.median(rel_err_t) > baseline:
        print(f"  ! ICP is worse than assuming no motion — it is diverging, "
              f"not merely imprecise")

    # Absolute drift, after putting both trajectories in the same frame. Rigid
    # alignment only — no scale term, because both are metric and a scale fit
    # would hide exactly the failure worth seeing.
    ec, rc = est_p - est_p.mean(0), ref_p - ref_p.mean(0)
    U, _, Vt = np.linalg.svd(ec.T @ rc)
    d = np.sign(np.linalg.det(U @ Vt))
    Ropt = U @ np.diag([1, 1, d]) @ Vt
    ate = np.linalg.norm(ec @ Ropt - rc, axis=1)
    print()
    print(f"absolute trajectory error after rigid alignment")
    print(f"  rms {ate.mean() * 100:.1f} cm   max {ate.max() * 100:.1f} cm"
          f"   ({ate.max() / max(travelled, 1e-6) * 100:.1f}% of distance travelled)")

    if np.median(inliers) < 0.3:
        print("\n! ICP is barely associating points. Likely the scene is beyond "
              "the sensor's ~5 m range, or motion between frames is too large — "
              "try --stride 1 or a slower capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
