#!/usr/bin/env python3
"""Score a trajectory's position by how thick the surfaces it builds come out.

A pose error smears a wall. The sensor's own floor is known — two clouds from
alternate frames of one session, with zero alignment error by construction, sit
0.77 cm apart — so anything beyond that belongs to the trajectory. It needs no
reference, which is what makes it usable on the multi-cam sessions that have
none, and unlike the revisit measure it does not depend on a long-baseline
photometric match succeeding.

    python3 eval/surface_thickness.py SESSION --trajectory pi3traj/<id>.npz

**Two cell sizes, because one cannot cover the range.** Thickness is measured
inside a voxel and a voxel only sees error smaller than itself, so the cell sets
both the ceiling and the floor:

```
  voxel     reads up to    empty-room floor
   10 cm        5 cm           3.68 cm
   20 cm       10 cm           5.75
   40 cm       20 cm           9.17
   80 cm       20 cm          15.92
```

The floor rises because a large cell contains real surface curvature and counts
it as thickness; at 80 cm a 1-2 cm injection is invisible. So the fine cell is
reported for the 0.5-5 cm range where good trajectories live, and the coarse one
for the 5-20 cm range where bad ones do. A number from the wrong cell is not
wrong so much as blind.

Calibrated by injecting known per-frame jitter, which is the check the revisit
measure failed: there, recovering a 5 cm injection returned 45 cm and a 20 cm
injection returned 32.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from read_session import Session  # noqa: E402
from pi3_poseless import load_calibration, pair_by_time  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FINE, COARSE = 0.10, 0.40


def frame_clouds(session: Session, rows, trajectory) -> list:
    """Each frame's depth as points in its own camera frame, with its pose."""
    wide = load_calibration(REPO / "calib" / "iphone17-1_wide.json")
    by = {int(f): T for f, T in zip(trajectory["frame"], trajectory["estimate"])}
    out = []
    for row in rows:
        T = by.get(int(row["frame"]))
        if T is None:
            continue
        z = np.asarray(session.depth_frame(row["depth"]), dtype=np.float32)
        h, w = z.shape
        sx, sy = w / wide["width"], h / wide["height"]
        fx, fy = wide["fx"] * sx, wide["fy"] * sy
        cx, cy = wide["cx"] * sx, wide["cy"] * sy
        v, u = np.mgrid[0:h, 0:w]
        ok = np.isfinite(z) & (z > 0.3) & (z < 4.0)
        # Every other row and column: the grid is dense enough that the full set
        # only costs memory.
        ok[::2] = False
        ok[:, ::2] = False
        if not ok.any():
            continue
        zz = z[ok]
        out.append((np.stack([(u[ok] - cx) * zz / fx,
                              (v[ok] - cy) * zz / fy, zz], axis=1), T))
    return out


def assemble(clouds, jitter=0.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for cam, T in clouds:
        shift = rng.normal(scale=jitter, size=3) if jitter else 0.0
        parts.append(cam @ T[:3, :3].T + (T[:3, 3] + shift))
    return np.concatenate(parts)


def thickness_cm(points: np.ndarray, voxel: float, min_points: int = 40) -> float:
    """Median 5-95 spread across a cell's own best-fit plane, over full cells."""
    cells = np.floor(points / voxel).astype(np.int64)
    key = ((cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663)
           ^ (cells[:, 2] * 83492791))
    order = np.argsort(key)
    key, sorted_points = key[order], points[order]
    _, start = np.unique(key, return_index=True)
    stop = np.append(start[1:], len(key))
    spreads = []
    for a, b in zip(start, stop):
        if b - a < min_points:
            continue
        group = sorted_points[a:b]
        centred = group - group.mean(0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][2]
        spreads.append(np.percentile(np.abs(centred @ normal), 90) * 2)
    return float(np.median(spreads)) * 100 if spreads else float("nan")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--calibrate", action="store_true",
                    help="inject known per-frame jitter and report what each "
                         "cell size does with it, which is what says whether a "
                         "reading means anything")
    a = ap.parse_args(argv)

    session = Session(a.session)
    rows = pair_by_time(session, 0.05)
    clouds = frame_clouds(session, rows, np.load(a.trajectory))
    if len(clouds) < 5:
        raise SystemExit(f"only {len(clouds)} frames had both depth and a pose")

    points = assemble(clouds)
    fine, coarse = thickness_cm(points, FINE), thickness_cm(points, COARSE)
    print(f"{len(clouds)} frames, {len(points):,} points")
    print(f"  fine   ({FINE * 100:.0f} cm cells): {fine:.2f} cm   "
          f"— reads position error up to about 5 cm")
    print(f"  coarse ({COARSE * 100:.0f} cm cells): {coarse:.2f} cm   "
          f"— reads up to about 20 cm, floor near 9 cm on a good trajectory")

    if a.calibrate:
        print(f"\n{'injected':>9} {'fine':>8} {'coarse':>8}")
        for j in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
            p = assemble(clouds, jitter=j)
            print(f"{j * 100:>8.0f}cm {thickness_cm(p, FINE):>8.2f} "
                  f"{thickness_cm(p, COARSE):>8.2f}")
        print("\n  a cell only sees error smaller than itself; where a column "
              "stops moving,\n  that cell has stopped measuring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
