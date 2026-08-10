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
from depth_odometry import (LocalMap, Tracker, frame_points,  # noqa: E402
                            icp)

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


def _render_boxes(position, R, boxes, max_range=5.0):
    """Render the nearest hit from a collection of axis-aligned boxes."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    d = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    world_d = d @ R.T                            # into world
    hit_distance = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in boxes:
            t1 = (lo - position) / world_d
            t2 = (hi - position) / world_d
            entry = np.minimum(t1, t2).max(axis=-1)
            exit = np.maximum(t1, t2).min(axis=-1)
            # A box can be an obstacle outside the camera, unlike the outer
            # room that contains the camera. Only a forward ray/box overlap is
            # a hit; otherwise a slab crossing behind the camera would win.
            hit = (exit >= np.maximum(entry, 0.0))
            distance = np.where(hit, np.where(entry > 0.0, entry, exit), np.inf)
            hit_distance = np.minimum(hit_distance, distance)
    # Range-limited like the real sensor, and the depth map stores Z, not range.
    z = hit_distance * d[..., 2]
    z[(hit_distance > max_range) | ~np.isfinite(hit_distance)] = np.nan
    return z


def render(position, R):
    """Depth map of the original small room, by ray-slab intersection."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    d = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d = d @ R.T                                   # into world
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (LO - position) / d
        t2 = (HI - position) / d
        tmax = np.minimum(np.maximum(t1, t2), 1e9).min(axis=-1)
    # Keep the original small-box convention byte-for-byte: existing pairwise
    # and accumulation assertions are deliberately regression tests for it.
    z = tmax * (d @ R)[..., 2]
    z[(tmax > 5.0) | ~np.isfinite(tmax)] = np.nan
    return z


# The original six-sided box above is intentionally kept for the pairwise
# tests. This is a separate, room-scale scene: the outer shell is large enough
# that it does not fill the view by itself, while partitions and furniture make
# the visible surfaces change as the camera walks around corners.
REALISTIC_BOXES = (
    (np.array([-3.8, -1.5, -3.0]), np.array([3.8, 1.7, 7.0])),
    # An L-shaped partition, leaving a route around its outside corner.
    (np.array([-0.45, -1.5, -0.55]), np.array([-0.20, 0.85, 2.45])),
    (np.array([-0.45, -1.5, 2.25]), np.array([2.00, 0.85, 2.50])),
    # Low furniture blocks. Their tops add horizontal geometry without making
    # the camera path pass through a solid box.
    (np.array([-2.35, -0.45, 0.10]), np.array([-1.15, 1.05, 1.30])),
    (np.array([0.75, -0.50, -0.35]), np.array([1.90, 1.00, 0.70])),
    (np.array([-2.10, -0.60, 1.10]), np.array([-0.75, 0.90, 1.70])),
    # A central island stays just outside the smooth loop, putting broad
    # near-range surfaces in view without placing the camera inside it.
    (np.array([-1.82, -0.50, -1.30]), np.array([1.82, 1.00, 1.30])),
)


def render_realistic(position, R):
    """Depth map for the room-scale multi-box scene."""
    return _render_boxes(position, R, REALISTIC_BOXES)


def render_realistic_uncut(position, R):
    """Same room geometry without the sensor-range cutoff, for statistics."""
    return _render_boxes(position, R, REALISTIC_BOXES, max_range=np.inf)


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


def realistic_walk(n=144, pitch=15.0):
    """Walk a smooth room-scale loop around the partition and furniture."""
    # The loop is deliberately smooth: a 90-degree pose jump at a polygonal
    # waypoint would test ICP's basin of convergence rather than the scene.
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    out = []
    for angle in theta:
        xz = np.array([2.70 * math.sin(angle),
                       0.90 + 3.00 * math.cos(angle)])
        tangent = np.array([2.70 * math.cos(angle),
                            -3.00 * math.sin(angle)])
        yaw = math.atan2(tangent[0], tangent[1])
        pos = np.array([xz[0], 0.0, xz[1]])
        out.append((pos, rot(yaw, math.radians(pitch))))
    return out


CORRELATED_NOISE_CELL = 16


def add_depth_noise(z, rng, amount=0.005, model="iid"):
    """Add selectable depth noise while preserving the existing IID model.

    The correlated model is a deliberately simple proxy for sparse LiDAR
    samples guided by RGB and then upsampled: it makes one random value per
    16x16-pixel cell and repeats it over the depth map, then adds a small IID
    component. Sixteen pixels is a modeling estimate of a plausible correlation
    length, not a measurement from a recorded session; measuring that value is
    outside this test's scope. The RMS scale remains `amount` (0.5% by default).
    """
    if not amount:
        return z
    if model == "iid":
        relative = rng.normal(0.0, amount, z.shape)
    elif model == "correlated":
        cell = CORRELATED_NOISE_CELL
        coarse_shape = ((H + cell - 1) // cell, (W + cell - 1) // cell)
        correlated = rng.normal(0.0, amount * math.sqrt(0.96), coarse_shape)
        correlated = np.repeat(np.repeat(correlated, cell, axis=0), cell, axis=1)
        correlated = correlated[:H, :W]
        iid = rng.normal(0.0, amount * math.sqrt(0.04), z.shape)
        relative = correlated + iid
    else:
        raise ValueError(f"unknown depth noise model: {model}")
    return z + relative * z


def accumulate(name, frame_to_frame, tol, path=None, noise=0.005,
               smooth=True, render_fn=render, noise_model="iid",
               report=True, return_tracker=False, **kw):
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
        z = render_fn(pos, R)
        # Seeded per frame, so a failure is reproducible rather than a mood.
        rng = np.random.default_rng(1000 + i)
        z = add_depth_noise(z, rng, noise, noise_model)
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
    if report:
        print(f"  {name:32} drift {worst * 100:5.2f} cm over "
              f"{travelled:.2f} m   final {err[-1] * 100:5.2f} cm   "
              f"{'PASS' if passed else 'FAIL'}")
    if return_tracker:
        return passed, worst, tracker
    return passed, worst


def scene_distance_stats(path, render_fn):
    """Return ray-range percentiles from the uncut scene geometry."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    ray_z = 1.0 / np.sqrt(((u - CX) / FX) ** 2
                          + ((v - CY) / FY) ** 2 + 1.0)
    distances = []
    for position, R in path:
        z = render_fn(position, R)
        distances.append((z / ray_z)[np.isfinite(z)])
    values = np.concatenate(distances)
    return np.percentile(values, [5, 50, 95])


def measured_drift(path, render_fn, **tracker_options):
    """Run one non-asserting experiment with the correlated noise model."""
    _, drift, tracker = accumulate(
        "experiment", False, math.inf, path=path, noise_model="correlated",
        render_fn=render_fn, report=False, return_tracker=True,
        **tracker_options)
    return drift, tracker


def comparison_results(path, render_fn):
    """Measure the two design choices without turning either into an assertion."""
    projective, _ = measured_drift(
        path, render_fn, voxel=0.03, projective_association=True)
    nearest, baseline_tracker = measured_drift(
        path, render_fn, voxel=0.03, projective_association=False)
    voxel_005, _ = measured_drift(
        path, render_fn, voxel=0.05, projective_association=False)
    return {
        "projective": projective,
        "nearest": nearest,
        "voxel_005": voxel_005,
        "voxel_003": nearest,
        "tracker": baseline_tracker,
    }


def print_realistic_scene_report():
    """Print the requested measurements for both scenes and the real sessions."""
    old_path = walk(72)
    new_path = realistic_walk()
    old = comparison_results(old_path, render)
    new = comparison_results(new_path, render_realistic)
    p05, median, p95 = scene_distance_stats(new_path, render_realistic_uncut)
    keyframes = new["tracker"].keyframes

    print()
    print("realistic-scene measurements (correlated noise; lower drift is better)")
    print(f"  scene ray-range p05/median/p95: {p05:.2f} / {median:.2f} / "
          f"{p95:.2f} m (depth returns above 5.0 m are invalid)")
    print(f"  keyframes: {keyframes}/{len(new_path)} "
          f"({100 * keyframes / len(new_path):.1f}%)")
    print("  comparison                         real data       existing box       "
          "realistic room")
    print("  nearest / projective (cm)          NN 20x better   "
          f"{old['nearest'] * 100:6.2f} / {old['projective'] * 100:6.2f}   "
          f"{new['nearest'] * 100:6.2f} / {new['projective'] * 100:6.2f}")
    print("  voxel 0.05 / 0.03 (cm)             0.05 better     "
          f"{old['voxel_005'] * 100:6.2f} / {old['voxel_003'] * 100:6.2f}   "
          f"{new['voxel_005'] * 100:6.2f} / {new['voxel_003'] * 100:6.2f}")


def fusion(path, mode, lam=0.02, drift=0.004, seed=7):
    """Dead-reckon a path from ARKit alone, depth alone, or the two fused.

    ARKit is simulated as truth plus a per-frame random walk, which is what its
    drift looks like at this timescale. The point of the case is not that fusion
    is more accurate on good geometry — that much is unsurprising — but that it
    is still an improvement on the degenerate path, where depth alone is at its
    worst. A fusion that helped only when depth was already good would be worth
    nothing, because that is not when help is needed.
    """
    rng = np.random.default_rng(seed)
    est = [np.eye(4)]
    prev = None
    for i, (pos, R) in enumerate(path):
        noise = np.random.default_rng(1000 + i)
        z = render(pos, R)
        z = z + noise.normal(0.0, 0.005, z.shape) * z
        pts, nrm, ok = prep(z)
        if prev is not None:
            truth = np.eye(4)
            truth[:3, :3] = path[i][1].T @ path[i - 1][1]
            truth[:3, 3] = path[i][1].T @ (path[i - 1][0] - path[i][0])
            noisy = truth.copy()
            noisy[:3, 3] = noisy[:3, 3] + rng.normal(0.0, drift, 3)
            if mode == "arkit":
                T = noisy
            elif mode == "depth":
                T, _, _ = icp(prev[0], prev[2], pts, nrm, ok, K)
            else:
                T, _, _ = icp(prev[0], prev[2], pts, nrm, ok, K,
                              prior=noisy.copy(), anchor=lam)
            est.append(est[-1] @ np.linalg.inv(T))
        prev = (pts, nrm, ok)
    P0 = np.eye(4)
    P0[:3, :3], P0[:3, 3] = path[0][1], path[0][0]
    return max(float(np.linalg.norm((P0 @ T)[:3, 3] - p))
               for T, (p, _) in zip(est, path))


def fusion_case(name, path):
    a = fusion(path, "arkit")
    d = fusion(path, "depth")
    f = fusion(path, "fused")
    ok = f < a and f < d
    print(f"  {name:32} arkit {a * 100:5.2f}  depth {d * 100:6.2f}  "
          f"fused {f * 100:5.2f} cm   {'PASS' if ok else 'FAIL'}")
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

    # The map path has a second branch nothing reached: when none of the map is
    # in view it registers against the previous frame instead. Every rendered
    # test keeps the room in frame from the first pose, so the branch never ran
    # here — and it was stepping backwards, composing a previous-from-current
    # transform as though it were current-from-previous. Emptying the map is
    # the branch's own trigger, and the only way to enter it deliberately.
    step = np.array([0.0, 0.0, 0.06])
    f0 = prep(render(o, rot(0)))
    f1 = prep(render(step, rot(0)))
    tracker = Tracker()
    tracker.step(*f0, K)
    tracker.map = LocalMap()
    tracker.step(*f1, K)
    got = tracker.poses[-1][:3, 3]
    fell_back = float(np.linalg.norm(got - step))
    print(f"  {'map out of view falls back':32} "
          f"t {fell_back * 100:5.2f} cm   "
          f"{'PASS' if fell_back < 0.02 else 'FAIL'}")
    print(f"  {'(stepped backwards before: 12 cm)':32} "
          f"now {fell_back * 100:5.2f} cm")
    results.append(fell_back < 0.02)

    print()
    print("accumulating a 120-frame walk — where drift actually lives")
    long_walk = walk(120)
    f2f_ok, f2f = accumulate("frame to frame (the baseline)", True, 0.05,
                             path=long_walk)
    map_ok, map_drift = accumulate("frame to map, keyframes fused",
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

    # Folding in every frame was the configuration that diverged under
    # projective association: the same surface went in at dozens of slightly
    # wrong estimated poses and thickened faster than averaging could sharpen
    # it, so keyframes were the fix and this asserted they beat every-frame.
    #
    # Nearest-neighbour association removed that ordering. Synthetically
    # every-frame is now ahead, and on the two loop-closed sessions the two
    # split one each — 1.7% vs 3.3% for keyframes, 2.9% vs 1.9% against. So
    # neither is established and neither is asserted; the numbers are printed
    # because the ordering flipping is the finding. FAST-LIO2 keeps no
    # keyframes at all, which is the direction this points.
    _, every = accumulate("every frame folded in", False, 1e9,
                          path=long_walk, keyframe_dist=0.0, keyframe_angle=0.0)
    print(f"  {'keyframes vs every-frame (no ordering)':38} "
          f"{map_drift * 100:5.2f} cm vs {every * 100:5.2f} cm")

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

    print("\nfusion: depth corrects ARKit only where the geometry earns it")
    results.append(fusion_case("good geometry", walk(40, pitch=15.0)))
    results.append(fusion_case("degenerate geometry", walk(40, pitch=0.0)))

    print_realistic_scene_report()

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
