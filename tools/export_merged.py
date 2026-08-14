#!/usr/bin/env python3
"""Build one 3DGS training set out of several aligned sessions.

A recording stops at about 35 seconds, so a room is several walks. `align_set.py`
puts them in one frame; this turns that into something a trainer reads.

    python3 tools/align_set.py ~/nav_data/a ~/nav_data/b --out t.json
    python3 tools/export_merged.py t.json /tmp/gs_merged --depth-format npy

The reference session goes first and **only its frames are held out**, so a
single-session export of the same reference is scored on the very same
photographs. That is the whole point: it makes "did the second walk help" a
question with an answer, rather than two runs measured on different frames.

    python3 tools/export_merged.py t.json /tmp/gs_one --only-reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_3dgs import export_merged  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transforms", help="the JSON written by align_set.py --out")
    ap.add_argument("out")
    ap.add_argument("--sessions-root", default=os.path.expanduser("~/nav_data"),
                    help="where the session directories live")
    ap.add_argument("--only-reference", action="store_true",
                    help="export the reference alone, with the identical split — "
                         "the control arm for a merged run")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--conf-min", type=int, default=2)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--pix-stride", type=int, default=1)
    ap.add_argument("--sharp-ratio", type=float, default=0.0)
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--guard-m", type=float, default=0.0,
                    help="drop training frames within this distance and 10 degrees "
                         "of a held-out one; the same band is applied to every "
                         "session, so the arms stay comparable")
    ap.add_argument("--depth-format", choices=("png16", "npy"), default="png16")
    ap.add_argument("--no-depth", action="store_true")
    a = ap.parse_args(argv)

    with open(a.transforms) as fh:
        doc = json.load(fh)
    reference = doc["reference"]
    transforms = doc["transforms"]
    if reference not in transforms:
        raise SystemExit(f"{reference} is not in the transform set")

    # Reference first; the rest in a stable order so two runs of this command
    # produce the same dataset.
    order = [reference] + sorted(k for k in transforms if k != reference)
    if a.only_reference:
        order = [reference]

    specs = []
    for session_id in order:
        path = os.path.join(a.sessions_root, session_id)
        if not os.path.isdir(path):
            raise SystemExit(f"session directory not found: {path}")
        specs.append((path, np.asarray(transforms[session_id], dtype=float)))

    report = export_merged(
        specs, a.out, downscale=a.downscale, conf_min=a.conf_min, voxel=a.voxel,
        max_points=a.max_points, pix_stride=a.pix_stride,
        image_mode="resize" if a.downscale > 1 else "symlink",
        with_depth=not a.no_depth, holdout_every=a.holdout_every,
        sharp_ratio=a.sharp_ratio, guard_m=a.guard_m,
        depth_format=a.depth_format)

    print(f"reference {report['reference'][-6:]}, "
          f"{report['images']} images, {report['holdout']} held out")
    for entry in report["merged_from"]:
        print(f"  {entry['session'][-6:]}  {entry['images']:>4} images"
              + ("   <- reference, the only source of held-out frames"
                 if entry["session"] == report["reference"] else ""))
    if report.get("guarded_out"):
        print(f"  guard band {report['guard_band_m']} m dropped "
              f"{report['guarded_out']} training frames")
    print(f"  points {report['points']} from {report['points_raw']} raw")
    print(f"  wrote  {report['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
