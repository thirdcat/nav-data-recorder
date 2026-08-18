#!/usr/bin/env python3
"""Score a trajectory's rotation against gravity, with no reference trajectory.

`docs/POSE.md` reports this measurement — 3.19° for the wide control, 2.78° for
the reprojected ultra-wide — but the code that produced it was never committed,
so the numbers could not be reproduced or re-run on a new session. This is that
code, written back from the description in the page.

**The idea.** CoreMotion reports gravity in the *device* frame. Gravity is one
fixed world direction, so a correct set of camera rotations carries every
frame's reading onto the same world vector, and whatever scatter is left is the
rotation error. Nothing here needs a reference trajectory, which is what makes
it usable on the multi-cam sessions that have none.

**Why `motion.jsonl` and not `pose.gravX/Y/Z`.** In an ARKit session the pose
file's gravity field is derived from ARKit's own poses, so scoring ARKit against
it is an identity and returns zero no matter what. `motion.jsonl` comes from
CoreMotion, which never saw any estimator. On a multi-cam session there is no
pose file at all and this is the only gravity there is.

**The device-to-camera rotation is fitted, not assumed.** Applying the pose
straight to device-frame gravity made every arm worse in `docs/POSE.md`,
including the control, and the reason is worth keeping: `R C g` is `R C Rᵀ`
applied to world gravity, so it depends on `R` and a fixed convention does not
cancel. `C` is solved for instead, alternating a Wahba/Kabsch solve against the
current mean direction. What survives that fit is the part no convention can
explain.

    python3 eval/gravity_residual.py SESSION --trajectory pi3traj/<id>.npz
    python3 eval/gravity_residual.py SESSION --arkit        # the positive control
    python3 eval/gravity_residual.py --self-test            # no session needed

**Read the control first.** A scorer that cannot recognise a trajectory already
known to be good is not measuring rotation, and this repository has shipped one
instrument that passed its self-consistency check and died on injected error.
`--self-test` injects known rotation noise and reports what comes back; `--arkit`
scores a tracker whose accuracy is independently established.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))

REPO = Path(__file__).resolve().parent.parent


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation `C` minimising `sum |C a_i - b_i|^2`, right-handed."""
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def residual_angles(rot: np.ndarray, g: np.ndarray, c: np.ndarray) -> tuple:
    """Per-frame angle between `R C g` and the mean of those directions."""
    w = unit(np.einsum("nij,jk,nk->ni", rot, c, g))
    m = unit(w.mean(axis=0))
    return np.degrees(np.arccos(np.clip(w @ m, -1.0, 1.0))), m


def _alternate(rot: np.ndarray, g: np.ndarray, c: np.ndarray,
               iterations: int) -> tuple:
    m = unit(np.einsum("nij,jk,nk->ni", rot, c, g).mean(axis=0))
    for _ in range(iterations):
        # Given the target `m`, every frame wants `C g_i` to equal `R_i^T m`.
        target = np.einsum("nji,j->ni", rot, m)
        new = kabsch(g, target)
        _, m = residual_angles(rot, g, new)
        if np.allclose(new, c, atol=1e-12):
            return new, m
        c = new
    return c, m


def fit_device_to_camera(rot: np.ndarray, g: np.ndarray, iterations: int = 100,
                         restarts: int = 64, seed: int = 0) -> tuple:
    """Alternate the mean world direction and the Wahba solve for `C`.

    **Multi-start, because the alternation has local minima and the self-test
    found one.** Seeded at the identity it converged to 3.88° on synthetic data
    whose true answer is zero — a wrong answer that looked like a plausible
    rotation error, which is the kind this instrument exists to report. Each
    start is a coordinate descent and the objective is not convex, so the fix is
    to try many and keep the best rather than to trust the first.
    """
    rng = np.random.default_rng(seed)
    best_c, best_m, best = None, None, np.inf
    starts = [np.eye(3)] + [kabsch(unit(rng.normal(size=(3, 3))), np.eye(3))
                            for _ in range(restarts)]
    for start in starts:
        c, m = _alternate(rot, g, start, iterations)
        ang, _ = residual_angles(rot, g, c)
        if float(np.mean(ang)) < best:
            best, best_c, best_m = float(np.mean(ang)), c, m
    return best_c, best_m


def load_motion(session: Path) -> tuple:
    t, g = [], []
    for line in (session / "motion.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("gx") is None:
            continue
        t.append(r["t"])
        g.append((r["gx"], r["gy"], r["gz"]))
    if not t:
        raise SystemExit("this session's motion.jsonl carries no gravity")
    order = np.argsort(np.asarray(t))
    return np.asarray(t)[order], unit(np.asarray(g, dtype=np.float64)[order])


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    q = np.stack([qw, qx, qy, qz], axis=-1)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def load_arkit(session: Path) -> tuple:
    t, q = [], []
    for line in (session / "pose.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t.append(r["t"])
        q.append((r["qw"], r["qx"], r["qy"], r["qz"]))
    a = np.asarray(q, dtype=np.float64)
    return np.asarray(t), quat_to_rot(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def pair_gravity(pose_t: np.ndarray, mt: np.ndarray, mg: np.ndarray,
                 max_dt: float) -> tuple:
    """Nearest motion sample per pose, and how far off each one was."""
    idx = np.clip(np.searchsorted(mt, pose_t), 1, len(mt) - 1)
    left = np.abs(pose_t - mt[idx - 1]) <= np.abs(mt[idx] - pose_t)
    idx = np.where(left, idx - 1, idx)
    dt = np.abs(pose_t - mt[idx])
    keep = dt <= max_dt
    return mg[idx], dt, keep


def report(name: str, rot: np.ndarray, g: np.ndarray) -> dict:
    raw, _ = residual_angles(rot, g, np.eye(3))
    c, _ = fit_device_to_camera(rot, g)
    fitted, _ = residual_angles(rot, g, c)
    out = {"n": len(rot), "raw": float(np.mean(raw)),
           "fitted": float(np.mean(fitted)),
           "median": float(np.median(fitted)),
           "p90": float(np.percentile(fitted, 90))}
    print(f"  {name:<34s} {out['n']:5d} {out['raw']:7.2f}   "
          f"{out['fitted']:7.2f}   {out['median']:7.2f}   {out['p90']:7.2f}")
    return out


def self_test() -> int:
    """Inject known rotation error and report what the score does with it.

    Proportionality is not calibration, so this does not merely check that more
    noise scores worse. It checks the two ends: a perfect set of rotations must
    return ~0, and a known injected scatter must come back at roughly its own
    size rather than at some arbitrary multiple of it.
    """
    rng = np.random.default_rng(0)
    n = 200
    # A plausible walk: gravity fixed in the world, the camera swinging about it.
    g_world = unit(np.array([0.0, -1.0, 0.0]))
    truth = np.empty((n, 3, 3))
    for i in range(n):
        ax = unit(rng.normal(size=3))
        ang = rng.uniform(-0.6, 0.6)
        k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        truth[i] = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)
    c_true = kabsch(unit(rng.normal(size=(3, 3))), np.eye(3))
    # Device-frame gravity is whatever puts `R C g` on the world direction.
    g_dev = np.einsum("ij,nkj,k->ni", c_true.T, truth, g_world)

    def inject(scale_deg, perpendicular_only):
        rot = truth.copy()
        if not scale_deg:
            return rot
        for i in range(n):
            if perpendicular_only:
                # Rotation about an axis square to gravity moves gravity by the
                # full angle; this is the arm where attenuation should vanish.
                ax = unit(np.cross(g_world, unit(rng.normal(size=3))))
            else:
                ax = unit(rng.normal(size=3))
            ang = np.radians(rng.normal(scale=scale_deg))
            k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
            rot[i] = (np.eye(3) + np.sin(ang) * k
                      + (1 - np.cos(ang)) * (k @ k)) @ rot[i]
        return rot

    ok = True
    print("  Gravity cannot see rotation about gravity, so a uniformly random")
    print("  injection must come back attenuated. The second column is the same")
    print("  test with that blind axis removed, and it must NOT be attenuated —")
    print("  which is what says the attenuation is geometry and not a defect.\n")
    print(f"  {'injected':>9s}   {'random axis':>13s}  {'ratio':>6s}   "
          f"{'perp to gravity':>15s}  {'ratio':>6s}")
    for scale in (0.0, 1.0, 3.0, 8.0):
        row = []
        for perp in (False, True):
            rot = inject(scale, perp)
            c, _ = fit_device_to_camera(rot, g_dev)
            ang, _ = residual_angles(rot, g_dev, c)
            row.append(float(np.mean(ang)))
        r0 = row[0] / scale if scale else float("nan")
        r1 = row[1] / scale if scale else float("nan")
        print(f"  {scale:8.1f}°   {row[0]:12.3f}°  {r0:6.2f}   "
              f"{row[1]:14.3f}°  {r1:6.2f}")
        if scale == 0.0 and max(row) > 0.05:
            ok = False
        if scale:
            # E[sin(angle between a uniform axis and gravity)] = pi/4, so a
            # random-axis injection should land near 0.79 and never at 1.
            if not 0.45 <= r0 <= 0.95:
                ok = False
            # With the blind axis excluded there is nothing left to hide in.
            if not 0.80 <= r1 <= 1.20:
                ok = False
    print(f"\n  {'PASS' if ok else 'FAIL'} — zero returns zero, a random-axis "
          f"injection attenuates to ~0.79, and\n         a gravity-perpendicular "
          f"injection returns at full size")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?")
    ap.add_argument("--trajectory", help="npz from eval/pi3_poseless.py")
    ap.add_argument("--arkit", action="store_true",
                    help="score this session's own ARKit poses — the control")
    ap.add_argument("--max-dt", type=float, default=0.02,
                    help="widest pose-to-motion gap to accept, seconds")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.session or not (a.trajectory or a.arkit):
        raise SystemExit("need a session and either --trajectory or --arkit")

    session = Path(a.session)
    mt, mg = load_motion(session)

    if a.arkit:
        pose_t, rot = load_arkit(session)
        label = f"{session.name} ARKit"
    else:
        d = np.load(a.trajectory)
        pose_t, rot = d["t"], d["estimate"][:, :3, :3]
        label = Path(a.trajectory).stem
        broken = int(d["broken_from"]) if "broken_from" in d else -1
        if broken >= 0:
            print(f"  note: this trajectory is flagged not trustworthy after "
                  f"frame {broken}; scoring all of it anyway, and again below "
                  f"restricted to the good prefix")

    g, dt, keep = pair_gravity(pose_t, mt, mg, a.max_dt)
    if keep.sum() < 10:
        raise SystemExit(f"only {keep.sum()} poses had gravity within "
                         f"{a.max_dt * 1000:.0f} ms")
    print(f"{keep.sum()} of {len(pose_t)} poses matched a motion sample; "
          f"gap median {np.median(dt[keep]) * 1000:.1f} ms, "
          f"worst {dt[keep].max() * 1000:.1f} ms")
    print(f"\n  {'arm':<34s} {'poses':>5s} {'raw':>7s}   {'fitted':>7s}   "
          f"{'median':>7s}   {'p90':>7s}")
    report(label, rot[keep], g[keep])

    if not a.arkit:
        d = np.load(a.trajectory)
        broken = int(d["broken_from"]) if "broken_from" in d else -1
        if broken >= 0:
            good = keep & (d["frame"] <= broken)
            if good.sum() >= 10:
                report(f"{label} (to frame {broken})", rot[good], g[good])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
