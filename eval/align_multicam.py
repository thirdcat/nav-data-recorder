#!/usr/bin/env python3
"""Put a MultiCam session into an ARKit session's world, and say when it cannot.

MultiCam sessions have no tracker, so nothing has ever scored their poses
against a reference: `docs/POSE.md` grades them with `surface_thickness` and
`gravity_residual` precisely because both work without one. When an ARKit walk
of the same room exists, that gap closes — and on 2026-08-20 one was recorded
63 seconds after a MultiCam walk of the same room.

    eval/.venv/bin/python eval/align_multicam.py TARGET SOURCE TRAJ.npz \\
        --control OTHER_SESSION

`tools/align_sessions.py` already solves the hard part, but its coarse search
assumes both clouds stand upright: it takes the vertical from the floor and
sweeps yaw alone. ARKit supplies that by running `worldAlignment = .gravity`. A
Pi3X poseless trajectory is in its solver's own gauge and supplies nothing, so
the missing two rotational degrees of freedom have to come from somewhere that
never saw the estimator.

**CoreMotion supplies them, and the claim is checked rather than assumed.**
`gravity_residual.fit_device_to_camera` returns the mean gravity direction in
whatever world the rotations live in. Run on an ARKit session, where the answer
is `-Y` by construction, it returns `[0.0033, -1.0000, 0.0071]` — 0.4 degrees.
`--check-gravity` runs exactly that before trusting the same fit on a session
where the answer is not known.

**The length is the lever, and most of this session does not place.** Scored
against an ARKit walk of the same room, with a different apartment as the
control that fitness alone cannot do without:

```
  window of d06152    fitness@5cm   control   ratio
   0-25 s                0.061       0.109     0.56
  10-30 s                0.577       0.046    12.59
  15-27 s                0.722       0.102     7.10
  38-57 s                0.126       0.186     0.68
  full 57.8 s            0.045       0.106     0.43
```

For scale, through this same harness a published real pair scores **0.849** and
a known-different room **0.457**. So a twelve-second window places and the whole
walk does not, and the failure is not subtle — at full length the different
apartment wins.

The reason is in the cloud rather than in the room: `d06152`'s own surfaces are
4.81 cm thick at the 10 cm cell, and `fitness@5cm` asks points to land within
5 cm of a surface. A 57.8 s Pi3X chain is not straight enough to place rigidly.
The photographs are not in doubt — SIFT retrieval puts the best pair at 165
inliers against a different-apartment ceiling of 15.

Note that `0-25 s` fails while `10-30 s` succeeds, so length is not the only
term: the overlapping views sit at t+20-21 s and t+44-49 s, and a window has to
contain one of them with room to spare. Report the window, not just the score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
import align_sessions as A  # noqa: E402
import gravity_residual as G  # noqa: E402
import pi3_poseless as P  # noqa: E402
from export_3dgs import (ARKIT_TO_COLMAP, DEPTH_FAR_M, DEPTH_NEAR_M,  # noqa: E402
                         depth_points, matrix_to_quat)
from read_session import Session  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Above this, `fitness@5cm` is a placement. Below it sits 0.457, which is what a
# known-different room reaches through this harness — so a score that merely
# beats its own control is not enough.
PLACED = 0.55
CONTROL_RATIO = 2.0


def world_gravity(session_dir: str, rot: np.ndarray, t: np.ndarray,
                  max_dt: float = 0.02) -> tuple[np.ndarray, float, int]:
    """Which way is down, in the world these rotations live in.

    CoreMotion never saw either estimator, which is the whole reason this can
    settle a gauge that the estimator itself cannot.
    """
    mt, mg = G.load_motion(Path(session_dir))
    g, _, keep = G.pair_gravity(t, mt, mg, max_dt)
    if keep.sum() < 20:
        raise SystemExit(f"only {int(keep.sum())} poses paired with a motion sample")
    c, m = G.fit_device_to_camera(rot[keep], g[keep])
    ang, _ = G.residual_angles(rot[keep], g[keep], c)
    return G.unit(m), float(np.mean(ang)), int(keep.sum())


def upright(g: np.ndarray) -> np.ndarray:
    """Rotation taking a world whose gravity is `g` to one where it is -Y."""
    a = g / np.linalg.norm(g)
    b = np.array([0.0, -1.0, 0.0])
    v, c = np.cross(a, b), float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def multicam_cloud(session_dir: str, traj_npz: str, gravity: np.ndarray, *,
                   window: tuple[float, float] | None = None, voxel: float = 0.05,
                   conf_min: int = 2, stride: int = 2, lens: str = "wide") -> dict:
    """The dict `align_sessions.session_cloud` returns, for a session with no ARKit.

    The poses are handed back in the shape `depth_points` reads rather than
    back-projected here, so the normals come from the same code the ARKit side
    uses. Standing in for them — camera -Z, say — makes every surface pass the
    45 degree vertical test, and `coarse_align` is built on the assumption that
    only walls enter its correlation.
    """
    session = Session(session_dir)
    manifest = json.loads((Path(session_dir) / "manifest.json").read_text())
    depth_calib, _ = P.depth_calibration(manifest)
    rows = P.pair_by_time(session, 0.05,
                          "frames" if lens == "ultrawide" else "frames_wide")
    traj = np.load(traj_npz)
    keep = np.ones(len(traj["t"]), bool)
    if window is not None:
        rel = traj["t"] - float(traj["t"].min())
        keep = (rel >= window[0]) & (rel <= window[1])
    by = {int(f): T for f, T, k in zip(traj["frame"], traj["estimate"], keep) if k}

    name = f"iphone17-1_{'ultrawide' if lens == 'ultrawide' else 'wide'}.json"
    calib = P.load_calibration(REPO / "calib" / name)
    row0 = rows[0]
    K = P.intrinsics_for(calib, int(row0["width"]), int(row0["height"]),
                         P.active_hfov(manifest, lens))
    R = upright(gravity)

    pts, nrm, centres, aims = [], [], [], []
    for row in rows:
        T = by.get(int(row["frame"]))
        if T is None:
            continue
        qw, qx, qy, qz = matrix_to_quat(T[:3, :3] @ ARKIT_TO_COLMAP)
        row = dict(row)
        row["pose"] = {"qx": qx, "qy": qy, "qz": qz, "qw": qw,
                       "tx": float(T[0, 3]), "ty": float(T[1, 3]),
                       "tz": float(T[2, 3]), "fx": K[0, 0], "fy": K[1, 1],
                       "cx": K[0, 2], "cy": K[1, 2], "tracking": "normal"}
        points, normals, _ = depth_points(session, row, conf_min=conf_min,
                                          near=DEPTH_NEAR_M, far=DEPTH_FAR_M,
                                          pix_stride=stride)
        if not len(points):
            continue
        pts.append(points @ R.T)
        nrm.append(normals @ R.T)
        centres.append(R @ T[:3, 3])
        aims.append(R @ T[:3, 2])
    if not pts:
        raise SystemExit("no frame of this session fell inside the window")

    points = np.concatenate(pts)
    normals = np.concatenate(nrm)
    colours = np.zeros((len(points), 3), dtype=np.uint8)
    points, normals, _, counts = A.voxel_average(points, normals, colours, voxel)
    return {"id": Path(session_dir).name[-6:], "points": points, "normals": normals,
            "counts": counts,
            "vertical": np.abs(normals @ A.WORLD_UP) < math.cos(math.radians(45.0)),
            "centres": np.asarray(centres), "aims": np.asarray(aims)}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="an ARKit session; its world is the answer")
    ap.add_argument("source", help="the MultiCam session to place")
    ap.add_argument("trajectory", help="pi3traj npz for the source's wide arm")
    ap.add_argument("--control", required=True,
                    help="a session known to be a different room; fitness "
                         "without one means nothing")
    ap.add_argument("--window", type=float, nargs=2, metavar=("FROM", "TO"),
                    help="seconds into the source session; most of a long walk "
                         "does not place")
    ap.add_argument("--check-gravity", action="store_true",
                    help="fit the target's own gravity first — it must come "
                         "back -Y, or the same fit on the source is unreadable")
    a = ap.parse_args(argv)

    if a.check_gravity:
        rows = [json.loads(x) for x in open(Path(a.target) / "pose.jsonl") if x.strip()]
        rows = [r for r in rows if r.get("tracking") == "normal" and r.get("qw")]
        rot = np.stack([G.quat_to_rot(r["qw"], r["qx"], r["qy"], r["qz"])
                        for r in rows])
        m, res, n = world_gravity(a.target, rot, np.array([r["t"] for r in rows]))
        off = math.degrees(math.acos(max(-1.0, min(1.0, float(m @ [0, -1, 0])))))
        print(f"  gravity control on the target: {np.round(m, 4)}, {off:.2f} deg "
              f"from -Y, residual {res:.2f} deg over {n} poses")
        if off > 2.0:
            raise SystemExit("the target's own gravity is not -Y — the fit is "
                             "not describing this data, so the source's is not "
                             "readable either")

    traj = np.load(a.trajectory)
    g, res, n = world_gravity(a.source, traj["estimate"][:, :3, :3], traj["t"])
    print(f"  source gravity in its own world: {np.round(g, 4)}, "
          f"residual {res:.2f} deg over {n} poses")

    target = A.session_cloud(a.target)
    control = A.session_cloud(a.control)
    window = tuple(a.window) if a.window else None
    source = multicam_cloud(a.source, a.trajectory, g, window=window)
    print(f"  clouds: target {len(target['points']):,}  source "
          f"{len(source['points']):,}  control {len(control['points']):,}")

    q = A.align_clouds(target, source, dof=6)["fitness"]["5cm"]
    c = A.align_clouds(control, source, dof=6)["fitness"]["5cm"]
    ratio = q / c if c > 0 else float("inf")
    print(f"\n  fitness@5cm  question {q:.3f}   control {c:.3f}   ratio {ratio:.2f}")
    print(f"  (a published real pair scores 0.849 here, a different room 0.457)")
    if q >= PLACED and ratio >= CONTROL_RATIO:
        print("  PLACED")
        return 0
    print(f"  NOT PLACED — needs fitness >= {PLACED} and >= {CONTROL_RATIO}x its "
          f"control.\n  A score that only beats its control is not a placement: "
          f"a wrong room reaches 0.457.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
