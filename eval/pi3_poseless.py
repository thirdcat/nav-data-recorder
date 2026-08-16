#!/usr/bin/env python3
"""Pose a session that has no tracker: the ultra-wide walks.

`MultiCamRecorder` reaches the ultra-wide lens by leaving ARKit, so its sessions
carry images, depth and IMU but **no `pose.jsonl`**. Everything else in `eval/`
assumes ARKit's pose is there — for the intrinsics, and as the reference to
score against — so this supplies the first from `calib/` and does without the
second.

    python3 eval/pi3_poseless.py ~/nav_data/<id> --out pi3traj/<id>.npz

Two joins have to be made by hand and both are worth stating.

**Images to depth, by time.** In an ARKit session the image and its depth come
out of one `ARFrame` and share a frame number. Here they are two independent
outputs of a multi-cam session with their own counters, so the only honest key
is the presentation timestamp, which both take from the same clock. Each image
takes the nearest depth frame and the gap is recorded; a pairing wider than
`--max-dt` is dropped rather than used.

**Pixels to bearings, from the calibration.** `calib/` carries the ultra-wide's
factory intrinsics at 4032x3024, and video frames are smaller, so `fx, fy, cx,
cy` scale by the image size. The lens is wide enough that a pinhole is a poor
description of its edges — `tools/rectify_ultrawide.py` exists for that — so
`--rectified` says the frames have already been through it and the intrinsics
should come from the rectifier's output instead.

There is no reference trajectory here and there cannot be. That is the point of
the mode, and it means the usual ATE-against-ARKit is unavailable: what this
writes has to be judged by the checks that need no reference.
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

REPO = Path(__file__).resolve().parent.parent


def load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text())
    w, h = data["reference_dimensions"]
    k = data["intrinsics"]
    return {"fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"],
            "width": float(w), "height": float(h)}


def intrinsics_for(calib: dict, width: int, height: int) -> np.ndarray:
    """Scale the factory intrinsics to this image size.

    Both axes are scaled independently rather than by one factor: a video format
    can crop as well as resize, and assuming a uniform scale would put the
    principal point in the wrong place for any format that does.
    """
    sx, sy = width / calib["width"], height / calib["height"]
    return np.array([[calib["fx"] * sx, 0.0, calib["cx"] * sx],
                     [0.0, calib["fy"] * sy, calib["cy"] * sy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def pair_by_time(session: Session, max_dt: float) -> list[dict]:
    """One row per image with the nearest depth frame, and how far off it was."""
    depth_rows = sorted(session.depth_index(), key=lambda d: d["t"])
    if not depth_rows:
        raise SystemExit("this session has no depth index")
    times = np.array([d["t"] for d in depth_rows])
    rows, dropped = [], 0
    for entry in session.stream("frames"):
        k = int(np.searchsorted(times, entry["t"]))
        best = min([i for i in (k - 1, k) if 0 <= i < len(depth_rows)],
                   key=lambda i: abs(times[i] - entry["t"]), default=None)
        if best is None:
            dropped += 1
            continue
        dt = float(entry["t"] - times[best])
        if abs(dt) > max_dt:
            dropped += 1
            continue
        rows.append({**entry, "depth": depth_rows[best], "depth_dt": dt})
    if dropped:
        print(f"  {dropped} images had no depth within {max_dt * 1000:.0f} ms and were dropped")
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--out", required=True, help="npz for the chained trajectory")
    ap.add_argument("--calibration",
                    default=str(REPO / "calib" / "iphone17-1_ultrawide.json"))
    ap.add_argument("--rectified", default=None,
                    help="JSON written by tools/rectify_ultrawide.py, when the "
                         "frames have already been reprojected to a pinhole")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--max-dt", type=float, default=0.05,
                    help="widest image-to-depth gap to accept, seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the pairing and the intrinsics, load no model")
    a = ap.parse_args(argv)

    session = Session(a.session)
    manifest_path = Path(a.session) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("kind") != "multicam":
        print(f"  note: manifest kind is {manifest.get('kind')!r}, not 'multicam' — "
              f"if this session has poses, eval/pi3_chain.py is the right tool")

    rows = pair_by_time(session, a.max_dt)
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} images paired with depth")
    gaps = np.array([abs(r["depth_dt"]) for r in rows])
    print(f"{len(rows)} images paired with depth; "
          f"gap median {np.median(gaps) * 1000:.1f} ms, worst {gaps.max() * 1000:.1f} ms")

    calib = load_calibration(Path(a.calibration))
    if a.rectified:
        rect = json.loads(Path(a.rectified).read_text())
        calib = {"fx": rect["fx"], "fy": rect.get("fy", rect["fx"]),
                 "cx": rect["cx"], "cy": rect["cy"],
                 "width": float(rect["width"]), "height": float(rect["height"])}
        print(f"  intrinsics from the rectifier: f {calib['fx']:.1f} at "
              f"{calib['width']:.0f}x{calib['height']:.0f}")
    K = intrinsics_for(calib, rows[0]["width"], rows[0]["height"])
    print(f"  intrinsics at {rows[0]['width']}x{rows[0]['height']}: "
          f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}")
    fov = 2 * np.degrees(np.arctan(rows[0]["width"] / (2 * K[0, 0])))
    print(f"  implied horizontal field of view {fov:.1f} deg")

    if a.dry_run:
        print("\n  dry run: the pairing and intrinsics above are what the model "
              "would be handed.")
        return 0

    # Imported here so --dry-run works on a machine with no GPU and no torch,
    # which is where the pairing is usually checked first.
    import torch  # noqa: F401
    from pi3_chain import chain_poseless  # noqa: F401

    raise SystemExit(
        "the model path is not wired yet — run with --dry-run to check the "
        "pairing and intrinsics, which is what this session needs verified "
        "before any GPU time is spent on it")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
