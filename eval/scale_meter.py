#!/usr/bin/env python3
"""Ask the depth sensor whether a trajectory's metric scale is right.

`docs/POSE.md` recorded that there was no reference-free way to score this:
path length needs a truth, plane thickness is scale-invariant, loop closure only
speaks for loops. The second of those is wrong about the thing that matters
here. Plane thickness *of one cloud* is scale-invariant, but this measure does
not scale one cloud — it re-poses many clouds through a trajectory and leaves
the LiDAR's own metres alone. Stretch the trajectory and the same wall, seen
from two places, lands twice.

    eval/.venv/bin/python eval/scale_meter.py SESSION TRAJ.npz --lens wide

So: sweep a uniform scale on the trajectory's increments, measure 40 cm surface
thickness at each, and read off where the minimum sits. `k` is what the
trajectory has to be multiplied by; `1/k` is what its scale currently is
relative to the depth.

**Calibrated against a trajectory already believed right**, which is the check
inject-and-recover cannot be a substitute for. ARKit is metric to about one per
cent by the steel-tape measurement in `docs/POSE.md`, and on `1696fa` this reads
its scale as 1.000. Pre-scaling that same trajectory by 1.15 moves the minimum
to 0.870, recovering 1.149; by 0.90, to 1.110, recovering 0.901.

**It refuses when the sweep is flat, and that happens often.** A minimum is only
a measurement if the curve rises out of it by more than the 0.7 cm this data
resolves. On the 760-pose ARKit session the well is 0.8-1.2 cm deep at plus or
minus five per cent. On the 93-159-pose poseless dual-lens arms it is 0.02-0.56
cm and two of six minima sit at the edge of the ladder — **this tool says
nothing about those trajectories**, and printing their argmin as a scale would
be the failure shape this repo keeps meeting: a fallback that looks like a
result. The likely difference is pose density, since 1696fa walks the same ten
metres with seven times the poses, but that has not been tested.
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
from read_session import Session  # noqa: E402
from pi3_poseless import depth_calibration, pair_by_time  # noqa: E402
from surface_thickness import COARSE, FINE, thickness_cm  # noqa: E402

# The smallest difference this data resolves, from the jitter injection in
# `surface_thickness`: 1 cm of injected error moves the coarse cell 0.69.
READABLE_CM = 0.7
LADDER = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)


def rescaled(poses: np.ndarray, k: float) -> np.ndarray:
    """The same trajectory with every position increment multiplied by `k`.

    The first pose and every rotation stay where they are, so this moves scale
    and nothing else. A whole-trajectory multiply would move the origin too and
    the thickness would then be reading a translation as well.
    """
    out = poses.copy()
    p = poses[:, :3, 3]
    out[0, :3, 3] = p[0]
    out[1:, :3, 3] = p[0] + np.cumsum(np.diff(p, axis=0) * k, axis=0)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("trajectory")
    # Same trap as `surface_thickness`: the two frame indices overlap in
    # numbering, so the wrong arm builds a cloud from mismatched instants and
    # prints a number for it instead of raising.
    ap.add_argument("--lens", choices=("ultrawide", "wide"), default="ultrawide",
                    help="which arm's images the trajectory was posed from")
    ap.add_argument("--reference", action="store_true",
                    help="score the npz's `reference` (ARKit) rather than its "
                         "`estimate` — this is how the tool was calibrated")
    ap.add_argument("--pre", type=float, default=1.0,
                    help="scale the trajectory by this before sweeping, so the "
                         "sweep has to find its inverse; a known-wrong input")
    ap.add_argument("--ladder", type=float, nargs="*", default=list(LADDER))
    a = ap.parse_args(argv)

    session = Session(a.session)
    manifest_path = Path(a.session) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    depth_calib, depth_source = depth_calibration(manifest)
    stream = "frames" if a.lens == "ultrawide" else "frames_wide"
    rows = pair_by_time(session, 0.05, stream)

    raw = np.load(a.trajectory)
    poses = raw["reference" if a.reference else "estimate"]
    frames = raw["frame"]
    finite = np.isfinite(poses.reshape(len(poses), -1)).all(axis=1)
    poses, frames = poses[finite], frames[finite]
    if a.pre != 1.0:
        poses = rescaled(poses, a.pre)

    print(f"  {a.lens} arm, depth grid from {depth_source}")
    print(f"  {len(poses)} poses"
          + (f", pre-scaled by {a.pre:g}" if a.pre != 1.0 else ""))
    print(f"\n{'k':>7} {'walked m':>9} {'fine cm':>9} {'coarse cm':>10}")

    ladder = sorted(a.ladder)
    coarse = []
    for k in ladder:
        moved = rescaled(poses, k)
        by = {int(f): T for f, T in zip(frames, moved)}
        parts = []
        for row in rows:
            T = by.get(int(row["frame"]))
            if T is None:
                continue
            z = np.asarray(session.depth_frame(row["depth"]), dtype=np.float32)
            h, w = z.shape
            sx, sy = w / depth_calib["width"], h / depth_calib["height"]
            fx, fy = depth_calib["fx"] * sx, depth_calib["fy"] * sy
            cx, cy = depth_calib["cx"] * sx, depth_calib["cy"] * sy
            v, u = np.mgrid[0:h, 0:w]
            ok = np.isfinite(z) & (z > 0.3) & (z < 4.0)
            ok[::2] = False
            ok[:, ::2] = False
            if not ok.any():
                continue
            zz = z[ok]
            cam = np.stack([(u[ok] - cx) * zz / fx,
                            (v[ok] - cy) * zz / fy, zz], axis=1)
            parts.append(cam @ T[:3, :3].T + T[:3, 3])
        if len(parts) < 5:
            raise SystemExit(f"only {len(parts)} frames had both depth and a pose")
        points = np.concatenate(parts)
        walked = np.linalg.norm(np.diff(moved[:, :3, 3], axis=0), axis=1).sum()
        c = thickness_cm(points, COARSE)
        coarse.append(c)
        print(f"{k:>7.3f} {walked:>9.2f} {thickness_cm(points, FINE):>9.2f} {c:>10.2f}")

    coarse = np.asarray(coarse)
    i = int(np.argmin(coarse))
    print()
    if i in (0, len(ladder) - 1):
        print(f"  the minimum is at the end of the ladder ({ladder[i]:.3f}), so it "
              f"is not bracketed.\n  Widen --ladder; this is not a scale.")
        return 1
    # How far the curve climbs out of the minimum on the shallower side. One
    # side rising is not a well, and a well shallower than the readable
    # difference is not a measurement.
    well = float(min(coarse[i - 1], coarse[i + 1]) - coarse[i])
    print(f"  minimum {coarse[i]:.2f} cm at k={ladder[i]:.3f}; the shallower "
          f"neighbour is {well:.2f} cm above it")
    if well < READABLE_CM:
        print(f"  that well is under the {READABLE_CM:.1f} cm this data resolves — "
              f"**the sweep is flat and says nothing**\n  about this trajectory's "
              f"scale. A flat surface is how `revisit_drift` failed.")
        return 1
    print(f"  the trajectory's scale is {1 / ladder[i]:.3f} of what the depth "
          f"wants (1.000 = agrees)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
