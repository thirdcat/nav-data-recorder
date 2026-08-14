#!/usr/bin/env python3
"""The control arm `fuse_rate.py` was accepted without.

`HANDOVER.md` §6 records the problem: both of the fusion's acceptance criteria
were guaranteed to pass. The anchor criterion is an identity — the fusion writes
the Pi3X pose into anchor rows and the scoring compares that array with itself,
which is why `delta_cm` came out as exactly `0.0` rather than a rounded zero.
The between-anchor criterion compares against ICP alone, which drifts, so
beating it is nearly automatic.

**What was never run is the arm that would make the comparison mean something:
interpolate the Pi3X anchors and use no ICP at all.** If that scores the same,
the fusion has demonstrated that it emits 30 Hz and nothing about ICP's local
accuracy — which is the premise the whole design rests on.

    python3 eval/fuse_control.py                 # every session with both files
    python3 eval/fuse_control.py --session cb4586

A second arm is here for the other doubt in the same section. The fusion spreads
its endpoint correction by left-multiplying a rigid interpolation, so the
rotation part turns the camera about the **world origin**. Far from the origin
that is a lever arm: a small angle becomes centimetres of translation on the
intermediate frames, and the reported ATE is 3–8 cm. `anchor` applies the same
correction about the left anchor's own position instead. Same inputs, same
scoring, one line different.

No GPU. Reads `traj/` and `pi3traj/` only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuse_rate as fr  # noqa: E402


def interpolate_between(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Rigid interpolation from one absolute pose to another, in the body frame.

    Body frame and not world: interpolating the two world poses component-wise
    would drag the camera along a chord through the room rather than along the
    motion that connects them.
    """
    relative = fr._inverse(start) @ end
    return start @ fr._interpolate_rigid(relative, alpha)


def fuse_pi3_only(traj, pi3, anchors) -> np.ndarray:
    """Pi3X anchors interpolated. The control: no ICP anywhere in this arm."""
    icp = traj["estimate"]
    pi3_estimate = pi3["estimate"]
    out = np.empty_like(icp, dtype=np.float64)

    first, last = int(anchors[0]), int(anchors[-1])
    # Outside the anchor span there is nothing to interpolate towards, so the
    # ends hold. That is a real limitation of the control and it is why the
    # comparison below is made on the between-anchor rows.
    out[:first] = pi3_estimate[0]
    out[last + 1:] = pi3_estimate[-1]

    for number, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        left, right = int(left), int(right)
        start, end = pi3_estimate[number], pi3_estimate[number + 1]
        span = traj["t"][right] - traj["t"][left]
        for k in range(left, right + 1):
            alpha = (traj["t"][k] - traj["t"][left]) / span
            out[k] = interpolate_between(start, end, alpha)
        out[left], out[right] = start, end
    return out


def fuse_anchor_centred(traj, pi3, anchors) -> np.ndarray:
    """The fusion, with the endpoint correction turned about the left anchor."""
    icp = traj["estimate"]
    pi3_estimate = pi3["estimate"]
    out = np.empty_like(icp, dtype=np.float64)

    first, last = int(anchors[0]), int(anchors[-1])
    for k in range(first - 1, -1, -1):
        out[k] = pi3_estimate[0] @ fr._inverse(icp[first]) @ icp[k]
    for k in range(last + 1, len(icp)):
        out[k] = pi3_estimate[-1] @ fr._inverse(icp[last]) @ icp[k]

    for number, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        left, right = int(left), int(right)
        start, end = pi3_estimate[number], pi3_estimate[number + 1]
        predicted_end = start @ fr._inverse(icp[left]) @ icp[right]
        correction = end @ fr._inverse(predicted_end)

        # Move the origin to the anchor, interpolate there, move back. The
        # rotation then pivots about a point inside the room rather than about
        # wherever the estimator happened to start.
        pivot = np.eye(4)
        pivot[:3, 3] = start[:3, 3]
        pivot_inv = np.eye(4)
        pivot_inv[:3, 3] = -start[:3, 3]
        local = pivot_inv @ correction @ pivot

        span = traj["t"][right] - traj["t"][left]
        for k in range(left, right + 1):
            predicted = start @ fr._inverse(icp[left]) @ icp[k]
            alpha = (traj["t"][k] - traj["t"][left]) / span
            out[k] = pivot @ fr._interpolate_rigid(local, alpha) @ pivot_inv @ predicted
        out[left], out[right] = start, end
    return out


def compare(traj_path: Path, pi3_path: Path) -> dict:
    traj = fr._load(traj_path, "traj")
    pi3 = fr._load(pi3_path, "pi3")

    fused, masks = fr.fuse(traj, pi3)
    anchors = masks["anchor_indices"]
    arms = {
        "fusion": fused,
        "pi3_only": fuse_pi3_only(traj, pi3, anchors),
        "anchor_centred": fuse_anchor_centred(traj, pi3, anchors),
    }

    reference = traj["reference"][:, :3, 3]
    between = masks["between_mask"]
    icp_between = fr._ate(traj["estimate"][between, :3, 3], reference[between])

    result = {
        "session": traj_path.stem,
        "frames": int(len(traj["frame"])),
        "anchors": int(len(pi3["frame"])),
        "between_frames": int(between.sum()),
        "icp_alone_cm": icp_between * 100.0,
    }
    for name, poses in arms.items():
        result[name + "_cm"] = fr._ate(poses[between, :3, 3], reference[between]) * 100.0
    # The number the section exists for: does ICP's local motion earn its place?
    result["icp_contribution_cm"] = result["pi3_only_cm"] - result["fusion_cm"]
    result["pivot_change_cm"] = result["fusion_cm"] - result["anchor_centred_cm"]
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traj-dir", default=str(fr.DEFAULT_TRAJ_DIR))
    ap.add_argument("--pi3-dir", default=str(fr.DEFAULT_PI3_DIR))
    ap.add_argument("--session", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    traj_dir, pi3_dir = Path(a.traj_dir), Path(a.pi3_dir)
    names = ([a.session] if a.session else
             sorted(p.stem for p in pi3_dir.glob("*.npz")
                    if (traj_dir / p.name).exists()))
    rows = []
    for name in names:
        try:
            rows.append(compare(traj_dir / f"{name}.npz", pi3_dir / f"{name}.npz"))
        except Exception as exc:
            print(f"{name}: {type(exc).__name__}: {exc}")

    if a.json:
        print(json.dumps(rows, indent=1))
        return 0

    print("between-anchor ATE against ARKit, centimetres — lower is better")
    print(f"{'session':>8} {'frames':>7} {'ICP alone':>10} {'Pi3X only':>10} "
          f"{'fusion':>8} {'anchor-piv':>11} {'ICP buys':>9}")
    for r in rows:
        print(f"{r['session']:>8} {r['between_frames']:>7} {r['icp_alone_cm']:>10.2f} "
              f"{r['pi3_only_cm']:>10.2f} {r['fusion_cm']:>8.2f} "
              f"{r['anchor_centred_cm']:>11.2f} {r['icp_contribution_cm']:>+9.2f}")

    if rows:
        buys = np.array([r["icp_contribution_cm"] for r in rows])
        pivot = np.array([r["pivot_change_cm"] for r in rows])
        wins = int((buys > 0).sum())
        print(f"\nICP helps in {wins} of {len(rows)} sessions, "
              f"median {np.median(buys):+.2f} cm")
        print(f"origin pivot vs anchor pivot: median {np.median(pivot):+.2f} cm "
              f"(positive means the anchor pivot is better)")
        print()
        if abs(float(np.median(buys))) < 0.5 and wins <= len(rows) * 0.6:
            print("The fusion is not distinguishable from interpolating the Pi3X")
            print("anchors alone. It demonstrates a 30 Hz output; ICP's local")
            print("accuracy remains unmeasured by this comparison.")
        else:
            print("ICP's relative motion changes the between-anchor error, so the")
            print("fusion is measuring something the interpolation does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
