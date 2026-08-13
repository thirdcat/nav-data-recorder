#!/usr/bin/env python3
"""Draw two trajectories from one `--dump-poses` NPZ as top-down tracks.

Written for the case ARKit cannot referee: a dark room, where the depth
estimator is the one that held up and ARKit is the suspect. When the reference
is under suspicion, ATE against it stops being a score, and the only judge left
is the person who walked the route. So this draws both tracks at the same scale
and reports the checks that need no reference at all — loop closure, path
length, and how flat the walk is against gravity.

Two things it has to get right, both of which cost me a wrong reading first:

* **The estimator's world frame is not gravity aligned.** It starts at the first
  camera pose, so with the phone aimed at the floor — the standard posture here
  — the whole walk sits on an oblique plane, and a per-axis range then reads as
  four metres of altitude change that never happened. Each track is projected
  onto its own gravity-perpendicular plane, using each frame's camera-frame
  gravity rotated into that estimator's own world.
* **Yaw is arbitrary and differs between the two.** Panels 1 and 2 are not
  rotations of each other and are not meant to be compared by orientation. Only
  panel 3, after a rigid fit, shows a difference that is a disagreement rather
  than a choice of frame.

One caveat the numbers cannot carry: the point-to-plane residual that
`depth_odometry.py` prints is the quantity ICP minimises, so it is scored on its
own homework. Loop closure and flatness-against-gravity are the checks that are
not.

    python3 eval/compare_tracks.py traj/<id>.npz out.svg

SVG rather than a plotting library, because this repo is numpy-only and these
tools have to run wherever the sessions land.
"""

import argparse
from pathlib import Path

import numpy as np

PANEL = 460.0
PAD = 42.0


def world_gravity(poses, gravity):
    """Per-frame camera gravity rotated into this estimator's own world frame."""
    world = np.einsum("nij,nj->ni", poses[:, :3, :3], gravity)
    mean = world.mean(0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        raise ValueError("gravity averages to zero; the frames disagree entirely")
    return mean / norm, world / np.linalg.norm(world, axis=1, keepdims=True)


def floor_frame(poses, gravity):
    """Positions projected onto the plane perpendicular to gravity."""
    up, _ = world_gravity(poses, gravity)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    p = poses[:, :3, 3]
    return np.stack([p @ e1, p @ e2], 1)


def walk_stats(poses):
    """Path length and how far the walk ended from where it started, in metres."""
    p = poses[:, :3, 3]
    return (float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
            float(np.linalg.norm(p[-1] - p[0])))


def flatness(poses, gravity):
    """Thickness of the walk about its own plane, and that plane's tilt off gravity."""
    p = poses[:, :3, 3]
    centred = p - p.mean(0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][2]
    up, _ = world_gravity(poses, gravity)
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal @ up), 0.0, 1.0))))
    return float(np.sqrt(((centred @ normal) ** 2).mean())), tilt


def rigid_fit_2d(source, target):
    """Rotation and translation only. Scale is the thing worth seeing, not hiding."""
    sc, tc = source - source.mean(0), target - target.mean(0)
    U, _, Vt = np.linalg.svd(sc.T @ tc)
    R = U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return sc @ R + target.mean(0)


def _panel(tracks, x0, scale, title, subtitle):
    body = []
    for points, colour, width in tracks:
        centred = points - points.mean(0)
        xy = [(x0 + PANEL / 2 + a * scale, PANEL / 2 + b * scale) for a, b in centred]
        line = " ".join(f"{a:.1f},{b:.1f}" for a, b in xy)
        body.append(f'<polyline points="{line}" fill="none" stroke="{colour}" '
                    f'stroke-width="{width}" stroke-linejoin="round" opacity="0.9"/>')
        body.append(f'<circle cx="{xy[0][0]:.1f}" cy="{xy[0][1]:.1f}" r="5" fill="#16a34a"/>')
        body.append(f'<rect x="{xy[-1][0] - 4:.1f}" y="{xy[-1][1] - 4:.1f}" '
                    f'width="8" height="8" fill="#111827"/>')
    return f"""
  <g>
    <rect x="{x0 + 5}" y="5" width="{PANEL - 10}" height="{PANEL - 10}"
          fill="none" stroke="#d4d4d8"/>
    {''.join(body)}
    <text x="{x0 + PANEL / 2}" y="{PANEL + 22}" text-anchor="middle"
          font-family="ui-monospace,monospace" font-size="14" fill="#111827">{title}</text>
    <text x="{x0 + PANEL / 2}" y="{PANEL + 41}" text-anchor="middle"
          font-family="ui-monospace,monospace" font-size="12" fill="#52525b">{subtitle}</text>
  </g>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="a traj NPZ from depth_odometry.py --dump-poses")
    ap.add_argument("svg", nargs="?", help="where to write the drawing")
    ap.add_argument("--estimate-name", default="depth ICP (ours)")
    ap.add_argument("--reference-name", default="ARKit")
    args = ap.parse_args(argv)

    with np.load(args.npz, allow_pickle=False) as data:
        est, ref = data["estimate"], data["reference"]
        gravity = data["gravity"]

    est2, ref2 = floor_frame(est, gravity), floor_frame(ref, gravity)
    est_len, est_gap = walk_stats(est)
    ref_len, ref_gap = walk_stats(ref)
    est_thick, est_tilt = flatness(est, gravity)
    ref_thick, ref_tilt = flatness(ref, gravity)
    fitted = rigid_fit_2d(est2, ref2)
    apart = np.linalg.norm(fitted - ref2, axis=1)

    name = Path(args.npz).stem
    print(f"{name}: {len(est)} frames")
    print(f"  {'':22s}{'estimate':>12s}{'reference':>12s}")
    print(f"  {'path length (m)':22s}{est_len:12.2f}{ref_len:12.2f}")
    print(f"  {'end-start (m)':22s}{est_gap:12.3f}{ref_gap:12.3f}")
    print(f"  {'out-of-plane rms (cm)':22s}{est_thick * 100:12.1f}{ref_thick * 100:12.1f}")
    print(f"  {'plane vs gravity (deg)':22s}{est_tilt:12.2f}{ref_tilt:12.2f}")
    print(f"  path length ratio {est_len / ref_len:.4f}")
    print(f"  disagreement after rigid fit: median {np.median(apart) * 100:.1f} cm, "
          f"max {apart.max() * 100:.1f} cm")
    print("  end-start is loop closure only if the walk returned to its start")

    if not args.svg:
        return 0

    span = max(np.ptp(est2, 0).max(), np.ptp(ref2, 0).max())
    scale = (PANEL - 2 * PAD) / max(span, 1e-6)
    width = PANEL * 3 + 40
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}"
     height="{PANEL + 78:.0f}" viewBox="0 0 {width:.0f} {PANEL + 78:.0f}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  {_panel([(est2, '#2563eb', 2.0)], 0, scale, args.estimate_name,
          f'{est_len:.2f} m walked · ends {est_gap:.2f} m from start')}
  {_panel([(ref2, '#dc2626', 2.0)], PANEL + 20, scale, args.reference_name,
          f'{ref_len:.2f} m walked · ends {ref_gap:.2f} m from start')}
  {_panel([(ref2, '#dc2626', 2.6), (fitted, '#2563eb', 1.8)], 2 * PANEL + 40, scale,
          'both, after rigid fit',
          f'median {np.median(apart) * 100:.0f} cm apart · max {apart.max() * 100:.0f} cm')}
  <text x="{width / 2}" y="{PANEL + 68}" text-anchor="middle"
        font-family="ui-monospace,monospace" font-size="12" fill="#71717a">
    {name} · each track flattened onto its own gravity plane · same scale ·
    green = start, black = end · yaw is arbitrary, so panels 1 and 2 need not share an orientation</text>
</svg>
"""
    Path(args.svg).write_text(svg)
    print(f"  wrote {args.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
