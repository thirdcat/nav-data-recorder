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
from depth_odometry import Tracker, frame_points, icp  # noqa: E402

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


def prep(z, smooth=True):
    """Exactly what a recorded session goes through, so the two cannot drift
    apart — the tool's own preparation, not a copy of it."""
    return frame_points(z, K, smooth=smooth)


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
    T, frac, _ = icp(a_pts, a_ok, b_pts, b_nrm, b_ok, K, prior=prior)

    et = float(np.linalg.norm(truth[:3, 3] - T[:3, 3]))
    dR = truth[:3, :3].T @ T[:3, :3]
    er = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    ok = et < tol_t and er < tol_r
    print(f"  {name:32} t {et * 100:5.2f} cm   r {er:5.2f}°   "
          f"inliers {frac * 100:3.0f}%   {'PASS' if ok else 'FAIL'}")
    return ok


def walk(n=60, pitch=15.0):
    """A slow lap of the room, as poses.

    The `pitch` default is the whole reason this path is trackable. Held level,
    the camera looks at the far wall and the floor and ceiling fall outside the
    49.3° vertical view — two vertical planes then leave *vertical* motion
    completely unobserved, and the estimate slides along that axis with 95%
    inliers and no complaint. Tilted down, one horizontal surface is in frame
    and the axis is pinned. It is also what a hand holding a phone actually
    does, so this is the realistic pose rather than a convenient one.

    `pitch=0` gives the degenerate path back, which is a case worth keeping.
    """
    out = []
    for i in range(n):
        s = i / (n - 1)
        pos = np.array([0.45 * math.sin(2 * math.pi * s),
                        0.03 * math.sin(6 * math.pi * s),
                        -0.55 + 1.1 * s])
        out.append((pos, rot(math.radians(14 * math.sin(2 * math.pi * s)),
                             math.radians(pitch + 3 * math.sin(4 * math.pi * s)))))
    return out


def accumulate(name, frame_to_frame, tol, path=None, noise=0.005,
               smooth=True, **kw):
    """Track a whole trajectory and measure how far the estimate has walked off.

    Single-step accuracy was never the problem — one map step recovers
    translation to a twentieth of a millimetre. Drift only shows up over dozens
    of frames, so this is the test that has to pass, and the one that caught
    accumulation diverging while every pairwise case above still passed.

    `noise` is a fraction of depth, roughly what the phone's sensor delivers at
    room scale. Without it this measures nothing useful: on perfect depth maps
    frame-to-frame tracking is already exact, so the map has no noise to average
    away and its whole reason for existing is invisible.
    """
    path = path or walk()
    tracker = Tracker(frame_to_frame=frame_to_frame, **kw)
    for i, (pos, R) in enumerate(path):
        z = render(pos, R)
        if noise:
            # Seeded per frame, so a failure is reproducible rather than a mood.
            rng = np.random.default_rng(1000 + i)
            z = z + rng.normal(0.0, noise, z.shape) * z
        pts, nrm, ok = prep(z, smooth=smooth)
        tracker.step(pts, nrm, ok, K)

    # The tracker starts at identity; truth starts at the first pose. Put both
    # in the same frame before comparing, with no fitted alignment — this is
    # dead reckoning from a known start, not a trajectory-shape comparison.
    P0 = np.eye(4)
    P0[:3, :3], P0[:3, 3] = path[0][1], path[0][0]
    err = [float(np.linalg.norm((P0 @ T)[:3, 3] - pos))
           for T, (pos, _) in zip(tracker.poses, path)]
    travelled = sum(float(np.linalg.norm(b[0] - a[0]))
                    for a, b in zip(path, path[1:]))
    worst = max(err)
    passed = worst < tol and math.isfinite(worst)
    print(f"  {name:32} drift {worst * 100:5.2f} cm over "
          f"{travelled:.2f} m   final {err[-1] * 100:5.2f} cm   "
          f"{'PASS' if passed else 'FAIL'}")
    return passed, worst


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
    T, _, cond = icp(a_pts, a_ok, b_pts, b_nrm, b_ok, K)
    slid = float(np.linalg.norm(T[:3, 3]))
    # Staying put is necessary but not sufficient — the estimator also has to
    # *know* it was blind, or nothing downstream can treat the frame differently.
    reported = cond < 1e-3
    print(f"  {'flat wall stays put (degenerate)':32} "
          f"drift {slid * 100:5.2f} cm   reports {cond:.1e}   "
          f"{'PASS' if slid < 0.02 and reported else 'FAIL'}")
    results.append(slid < 0.02 and reported)

    # Seeding the true rotation must not make things worse. It did on real
    # data, which turned out to be the harness transposing the prior rather
    # than the estimator disliking it.
    results.append(case("rotation prior does not hurt", o, rot(0),
                        np.array([0.18, 0.01, 0.05]),
                        rot(math.radians(4), math.radians(2)),
                        prior_rotation=True))

    print()
    print("accumulating a 120-frame walk — where drift actually lives")
    long_walk = walk(120)
    f2f_ok, f2f = accumulate("frame to frame (the baseline)", True, 0.05,
                             path=long_walk)
    map_ok, map_drift = accumulate("frame to model, keyframes fused",
                                   False, 0.02, path=long_walk)
    results += [f2f_ok, map_ok]

    # The map has to beat the baseline, not merely survive — otherwise it is
    # costing a render per frame for nothing. The margin is small here and that
    # is the honest result: this room is fully in view from the first frame, so
    # the baseline never faces the situation the map exists for. What separates
    # them is the *shape* — frame-to-frame is a random walk and grows with the
    # square root of the frame count, while the map is flat.
    better = map_drift <= f2f
    print(f"  {'the map beats the baseline':32} "
          f"{map_drift * 100:5.2f} cm vs {f2f * 100:5.2f} cm       "
          f"{'PASS' if better else 'FAIL'}")
    results.append(better)

    # Folding in every frame is the configuration that diverged: the same
    # surface is re-inserted at dozens of slightly wrong estimated poses, and
    # thickens faster than averaging can sharpen it. Keyframes are the fix, so
    # the fix has to be what makes the difference — not the fusion alone.
    _, every = accumulate("every frame folded in (the old bug)", False, 1e9,
                          path=long_walk, keyframe_dist=0.0, keyframe_angle=0.0)
    print(f"  {'keyframes beat every-frame':32} "
          f"{map_drift * 100:5.2f} cm vs {every * 100:5.2f} cm       "
          f"{'PASS' if map_drift <= every else 'FAIL'}")
    results.append(map_drift <= every)

    # Smoothing the depth first is not a refinement, it is most of the result:
    # normals come from a two-pixel baseline, which at two metres is two
    # centimetres, and a centimetre of per-pixel noise makes them nearly random.
    _, raw = accumulate("unsmoothed depth (what noise costs)", False, 1e9,
                        path=long_walk, smooth=False)
    print(f"  {'smoothing earns its place':32} "
          f"{map_drift * 100:5.2f} cm vs {raw * 100:5.2f} cm       "
          f"{'PASS' if map_drift <= raw else 'FAIL'}")
    results.append(map_drift <= raw)

    print()
    print("a walk whose geometry goes blind — the estimator must not invent")
    # Held level the camera sees two walls and no floor, and vertical motion
    # becomes unobservable. Nothing can recover it; what is being tested is that
    # the tracker coasts on the prediction instead of accepting the enormous,
    # confident update a singular normal equation offers. Before the eigenvalue
    # cut this ran to 160 cm — a whole trajectory lost to one blind frame.
    blind_ok, blind = accumulate("level walk, floor out of frame", False, 0.25,
                                 path=walk(120, pitch=0.0))
    results.append(blind_ok)
    print(f"  {'(unbounded before the fix: 160 cm)':32} "
          f"now {blind * 100:5.1f} cm")

    print()
    print("reading a recorded session — the plumbing the rendered tests skip")
    results += session_cases()

    print()
    if all(results):
        print("all passed")
        return 0
    print("FAILURES — the odometry is wrong")
    return 1


def session_cases() -> list[bool]:
    """Run the command-line tool over a generated session, end to end.

    Everything above feeds the tracker rendered frames directly, which leaves
    the whole path from a session on disk untested: the pose join on `frame`,
    the depth-to-colour intrinsic ratio, the ARKit-to-depth axis convention,
    and which frames are considered scorable at all. Those are not accuracy
    questions — the fixture's depth is nearly flat and its numbers mean nothing
    — so what is checked is *which frames were used*, which is the part that
    silently changed a result before.
    """
    import io
    import contextlib
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_test_session
    import depth_odometry

    def run(*flags):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = make_test_session.build(tmp)
            with contextlib.redirect_stdout(out):
                code = depth_odometry.main([path, *flags])
        return code, out.getvalue()

    results = []
    code, text = run()
    header = text.splitlines()[0] if text else ""
    # The fixture opens with 41 frames of `limited:initializing`, four of which
    # carry depth. ARKit parks the pose near the origin until it converges, so
    # scoring against those frames measures the reference starting up rather
    # than the estimator — the exporter has always dropped them and this now
    # does too.
    dropped = "dropped 4 frame(s)" in text and header.startswith(
        "20260807-014530-fixture: 96 depth frames")
    print(f"  {'unconverged frames are not scored':32} "
          f"{'PASS' if code == 0 and dropped else 'FAIL'}")
    results.append(code == 0 and dropped)

    code, text = run("--include-unconverged")
    kept = text.splitlines()[0].startswith(
        "20260807-014530-fixture: 100 depth frames")
    print(f"  {'--include-unconverged keeps them':32} "
          f"{'PASS' if code == 0 and kept else 'FAIL'}")
    results.append(code == 0 and kept)

    # The rate ICP saw is not the rate the depth arrived at once anything
    # decimates it, and the output has to distinguish them: a 5 Hz measurement
    # that reads as 30 Hz is how a result gets written up at the wrong rate.
    code, text = run("--stride", "4")
    said = "depth arrived at 5.0 Hz, scored at 1.2 Hz" in text
    print(f"  {'--stride is reported as scored rate':32} "
          f"{'PASS' if code == 0 and said else 'FAIL'}")
    results.append(code == 0 and said)
    return results


if __name__ == "__main__":
    raise SystemExit(main())
