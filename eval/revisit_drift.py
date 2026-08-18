#!/usr/bin/env python3
"""Score a trajectory's *position* with no reference, using its own revisits.

Everything else here scores rotation. Gravity gives a reference-free check on
the rotations because CoreMotion measures it independently, but nothing measures
translation the same way — and `docs/POSE.md` has the case that makes this
matter: on 2994fa every join residual sat between 0.1 and 0.5 cm while the
trajectory ended 2.45 m out. Internal consistency is not accuracy.

A revisit is the one place a trajectory can be caught. When it places two frames
from opposite ends of the walk in the same spot, it is making a claim the
photographs can check: they should show the same surfaces from the same angle.
Whatever correction is needed to make them agree is the drift accumulated
between them, and no reference was consulted to find it.

    python3 eval/revisit_drift.py SESSION --trajectory pi3traj/<id>.npz

Works on a session with no poses, which is the point — `--trajectory` supplies
what `pose.jsonl` would have.
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
import photo_align as pa  # noqa: E402
from read_session import Session  # noqa: E402
from pi3_poseless import (active_hfov, depth_in_ultrawide, intrinsics_for,  # noqa: E402
                          load_calibration, pair_by_time)

REPO = Path(__file__).resolve().parent.parent


def bundles_from_trajectory(session, rows, K, estimate, frames):
    """`photo_align`'s frame bundles, posed from the npz instead of pose.jsonl."""
    by = {int(f): T for f, T in zip(frames, estimate)}
    out = {}
    for k, row in enumerate(rows):
        T = by.get(int(row["frame"]))
        if T is None:
            continue
        z = np.asarray(session.depth_frame(row["depth"]), dtype=np.float32)
        conf = None
        ok = np.isfinite(z) & (z > pa.DEPTH_NEAR_M) & (z < pa.DEPTH_FAR_M)
        if ok.sum() < 500:
            continue
        out[k] = {"z": z, "ok": ok,
                  "grey": pa.grey(session.frame_path(row), z.shape[1], z.shape[0]),
                  "pose": T, "shape": z.shape,
                  # The depth grid is the wide camera's, so its intrinsics are
                  # the image's scaled to that grid — the same relation the
                  # ARKit path assumes and for the same reason.
                  "K": (K[0, 0] * z.shape[1] / row["width"],
                        K[1, 1] * z.shape[0] / row["height"],
                        K[0, 2] * z.shape[1] / row["width"],
                        K[1, 2] * z.shape[0] / row["height"])}
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--min-gap", type=int, default=40,
                    help="how far apart in the walk two frames must be to count "
                         "as a revisit rather than as neighbours")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--span", type=float, default=0.30)
    ap.add_argument("--passes", type=int, default=7)
    a = ap.parse_args(argv)

    session = Session(a.session)
    d = np.load(a.trajectory)
    estimate, frames = d["estimate"], d["frame"]

    manifest_path = Path(a.session) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rows = pair_by_time(session, 0.05)
    calib = load_calibration(REPO / "calib" / "iphone17-1_ultrawide.json")
    K = intrinsics_for(calib, rows[0]["width"], rows[0]["height"], active_hfov(manifest))

    bundles = bundles_from_trajectory(session, rows, K, estimate, frames)
    order = sorted(bundles)
    lite = [{"pose": bundles[k]["pose"]} for k in order]
    pairs = pa.overlapping_pairs(lite, lite, np.eye(4), limit=a.pairs,
                                 min_index_gap=a.min_gap)
    if not pairs:
        print("no revisits: this walk never comes back to a place it has been")
        return 1

    print(f"{len(pairs)} revisits at least {a.min_gap} frames apart\n")
    print(f"{'from':>6} {'to':>5} {'gap':>5} {'before':>8} {'after':>8} "
          f"{'drift cm':>9} {'walked m':>8} {'% of path':>8}")
    drifts = []
    for i, j in pairs:
        ref, src = bundles[order[i]], bundles[order[j]]
        # The trajectory's own claim about how these two frames relate.
        relative = np.linalg.inv(ref["pose"]) @ src["pose"]
        before = pa.reproject_score(ref, src, relative)[0]
        if not np.isfinite(before):
            print(f"{order[i]:>6} {order[j]:>5} {abs(order[j]-order[i]):>5}   "
                  f"too few shared pixels to score")
            continue
        refined, history = pa.refine({0: ref}, {0: src}, [(0, 0)], relative,
                                     span=a.span, passes=a.passes)
        shift = float(np.linalg.norm(refined[:3, 3] - relative[:3, 3]))
        # Drift accumulates with distance, so the raw centimetres are not
        # comparable between two walks of different length or between revisits
        # that span different amounts of it. Normalise by the path actually
        # walked between the two frames.
        lo, hi = sorted((order[i], order[j]))
        walked = float(np.linalg.norm(
            np.diff(np.stack([bundles[k]["pose"][:3, 3] for k in order[lo:hi + 1]]),
                    axis=0), axis=1).sum())
        drifts.append((shift, walked))
        print(f"{order[i]:>6} {order[j]:>5} {abs(order[j]-order[i]):>5} "
              f"{before:>8.4f} {history[-1]['score']:>8.4f} {shift * 100:>9.1f} "
              f"{walked:>8.2f} {shift / max(walked, 1e-6) * 100:>8.2f}")
    if not drifts:
        print("no revisit could be scored")
        return 1
    v = np.array([d for d, _ in drifts])
    w = np.array([p for _, p in drifts])
    rel = v / np.maximum(w, 1e-6) * 100
    print(f"\n  drift over a revisit: median {np.median(v) * 100:.1f} cm over "
          f"{np.median(w):.2f} m walked = {np.median(rel):.2f}% of path")
    print(f"  scored {len(drifts)} of {len(pairs)} revisits")
    print("  this is the correction the photographs demand where the trajectory")
    print("  claims it has returned — accumulated position error, no reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
