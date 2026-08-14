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


def interval_conditioning(anchors, cond) -> np.ndarray:
    """Score each anchor interval by its worst frame.

    An interval is only as good as the frame where ICP loses the geometry, so
    the minimum is the right summary and the mean is the wrong one.
    """
    return np.array([np.nanmin(cond[int(a):int(b) + 1])
                     for a, b in zip(anchors[:-1], anchors[1:])])


def fuse_gated(traj, pi3, anchors, cond, drop_worst: float) -> tuple:
    """The fusion, but ICP only carries the intervals it is qualified for.

    Measured over 11 054 frame pairs, per-frame `frame_conditioning` predicts
    ICP's *local* translation error at Spearman -0.446: the best decile runs
    4.3 mm against a 19 mm step, the worst 19.1 mm. The ICP residual predicts
    nothing (r = -0.057), which is why the gate is on conditioning.

    The gate ranks intervals and drops the worst `drop_worst` of them, rather
    than thresholding every frame against a session-wide quantile. That first
    form is what this function used to do and it is a trap: requiring *every*
    frame of an interval to clear the 75th percentile admits 13.5 % of intervals
    at 5 Hz anchors and **0 %** at 0.31 Hz, so the sparse-anchor arm scored an
    exact zero that read as "ICP is neutral" when it meant "the gate never
    fired". Ranking holds the admitted fraction fixed at every anchor rate,
    which is the only way the rates are comparable.

    An interval that fails the gate falls back to interpolating its two anchors
    — never to nothing.
    """
    scores = interval_conditioning(anchors, cond)
    keep = np.argsort(-scores)[:len(scores) - max(1, int(round(drop_worst * len(scores))))]
    icp = traj["estimate"]
    pi3_estimate = pi3["estimate"]
    out = fuse_pi3_only(traj, pi3, anchors).copy()

    for number in sorted(int(i) for i in keep):
        left, right = int(anchors[number]), int(anchors[number + 1])
        start, end = pi3_estimate[number], pi3_estimate[number + 1]
        predicted_end = start @ fr._inverse(icp[left]) @ icp[right]
        correction = end @ fr._inverse(predicted_end)
        span = traj["t"][right] - traj["t"][left]
        for k in range(left, right + 1):
            predicted = start @ fr._inverse(icp[left]) @ icp[k]
            alpha = (traj["t"][k] - traj["t"][left]) / span
            out[k] = fr._interpolate_rigid(correction, alpha) @ predicted
        out[left], out[right] = start, end
    return out, len(keep)


def compare(traj_path: Path, pi3_path: Path) -> dict:
    traj = fr._load(traj_path, "traj")
    pi3 = fr._load(pi3_path, "pi3")

    fused, masks = fr.fuse(traj, pi3)
    anchors = masks["anchor_indices"]
    with np.load(traj_path) as d:
        cond = (np.asarray(d["frame_conditioning"], dtype=np.float64)
                if "frame_conditioning" in d.files else None)
    arms = {
        "fusion": fused,
        "pi3_only": fuse_pi3_only(traj, pi3, anchors),
        "anchor_centred": fuse_anchor_centred(traj, pi3, anchors),
    }
    gated_used = None
    if cond is not None:
        arms["gated"], gated_used = fuse_gated(traj, pi3, anchors, cond, 0.50)

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
    if gated_used is not None:
        result["gated_intervals"] = int(gated_used)
        result["gated_gain_cm"] = result["pi3_only_cm"] - result["gated_cm"]
    return result


def anchor_sweep(traj_path: Path, pi3_path: Path,
                 factors=(1, 2, 3, 4, 6, 8)) -> dict:
    """How fast does the error grow as the anchors are thinned?

    Pi3X anchors sit exactly on the image frames — 5 Hz, because that is the
    still rate — so "more anchors" is not a solver setting, it is a capture
    setting with a thermal bill. Before paying it, the shape of the curve can be
    read off the data already recorded by throwing anchors away.

    Every arm is scored on the **same rows**: the depth frames that are not
    image frames, and therefore never an anchor in any arm. Scoring on all rows
    would hand each configuration its own anchors as free exact answers, and the
    densest one would win for that reason alone.
    """
    traj = fr._load(traj_path, "traj")
    pi3 = fr._load(pi3_path, "pi3")
    reference = traj["reference"][:, :3, 3]

    by_frame = {int(f): i for i, f in enumerate(traj["frame"])}
    full = np.asarray([by_frame[int(f)] for f in pi3["frame"]], dtype=np.int64)

    # Rows that are never an anchor, and lie inside the span every arm covers.
    span = np.zeros(len(traj["frame"]), dtype=bool)
    span[full[0]:full[-1] + 1] = True
    scored = span.copy()
    scored[full] = False

    out = {"session": traj_path.stem, "scored_rows": int(scored.sum()), "arms": []}
    for k in factors:
        keep = np.arange(0, len(full), k)
        if keep[-1] != len(full) - 1:
            keep = np.append(keep, len(full) - 1)
        if len(keep) < 2:
            continue
        thinned = {"estimate": pi3["estimate"][keep], "frame": pi3["frame"][keep],
                   "t": pi3["t"][keep], "reference": pi3["reference"][keep]}
        poses = fuse_pi3_only(traj, thinned, full[keep])
        out["arms"].append({
            "every": k,
            "anchors": int(len(keep)),
            "hz": round(5.0 / k, 2),
            "ate_cm": fr._ate(poses[scored, :3, 3], reference[scored]) * 100.0,
        })
    return out


def gate_sweep(traj_path: Path, pi3_path: Path,
               factors=(1, 4, 8, 16), seeds=(0, 1, 2, 3, 4)) -> dict:
    """Where does a conditioning gate on ICP start to pay, and does it select?

    Two questions the dense-anchor table cannot answer, because at 5 Hz the
    anchors are 0.2 s and ~12 cm apart and a straight line across that is
    already right to millimetres — there is nothing for a 4 mm local constraint
    to improve. Thinning the anchors opens the gap. 0.31 Hz is not a hypothetical
    setting: thermal throttling stops image writes while poses keep recording
    (`HANDOVER.md` §2), so a hot session takes exactly this shape.

    The controls are the point. `random` admits the same *number* of intervals
    with no regard for conditioning, and `worst` admits the badly conditioned
    ones. If ranking by conditioning beats neither, then any gain belongs to
    "ICP helps across long gaps" and not to the gate.
    """
    traj = fr._load(traj_path, "traj")
    pi3 = fr._load(pi3_path, "pi3")
    with np.load(traj_path) as d:
        if "frame_conditioning" not in d.files:
            raise KeyError("no frame_conditioning in this trajectory")
        cond = np.asarray(d["frame_conditioning"], dtype=np.float64)
    reference = traj["reference"][:, :3, 3]

    by_frame = {int(f): i for i, f in enumerate(traj["frame"])}
    full = np.asarray([by_frame[int(f)] for f in pi3["frame"]], dtype=np.int64)
    scored = np.zeros(len(traj["frame"]), dtype=bool)
    scored[full[0]:full[-1] + 1] = True
    scored[full] = False

    def place(picked, anchors, thinned):
        icp, pe = traj["estimate"], thinned["estimate"]
        out = fuse_pi3_only(traj, thinned, anchors).copy()
        for number in sorted(int(i) for i in picked):
            left, right = int(anchors[number]), int(anchors[number + 1])
            start, end = pe[number], pe[number + 1]
            correction = end @ fr._inverse(start @ fr._inverse(icp[left]) @ icp[right])
            span = traj["t"][right] - traj["t"][left]
            for k in range(left, right + 1):
                alpha = (traj["t"][k] - traj["t"][left]) / span
                out[k] = (fr._interpolate_rigid(correction, alpha)
                          @ (start @ fr._inverse(icp[left]) @ icp[k]))
            out[left], out[right] = start, end
        return out

    out = {"session": traj_path.stem, "rates": []}
    for k in factors:
        keep = np.arange(0, len(full), k)
        if keep[-1] != len(full) - 1:
            keep = np.append(keep, len(full) - 1)
        if len(keep) < 3:
            continue
        thinned = {"estimate": pi3["estimate"][keep], "frame": pi3["frame"][keep],
                   "t": pi3["t"][keep], "reference": pi3["reference"][keep]}
        anchors = full[keep]
        scores = interval_conditioning(anchors, cond)
        n = len(scores)
        quarter = max(1, int(round(0.25 * n)))
        base = fr._ate(fuse_pi3_only(traj, thinned, anchors)[scored, :3, 3],
                       reference[scored]) * 100.0

        arms = {
            "all": np.arange(n),
            "drop_worst_25": np.argsort(-scores)[:n - quarter],
            "drop_worst_50": np.argsort(-scores)[:n - max(1, int(round(0.50 * n)))],
            "best_25": np.argsort(-scores)[:quarter],
            "worst_25": np.argsort(scores)[:quarter],
        }
        row = {"every": k, "hz": round(5.0 / k, 2), "intervals": n,
               "interp_cm": base, "gain_cm": {}}
        for name, picked in arms.items():
            row["gain_cm"][name] = base - fr._ate(
                place(picked, anchors, thinned)[scored, :3, 3], reference[scored]) * 100.0
        # The control that separates "the gate selects" from "long gaps favour ICP".
        row["gain_cm"]["random_25"] = float(np.median([
            base - fr._ate(place(np.random.default_rng(s).permutation(n)[:quarter],
                                 anchors, thinned)[scored, :3, 3],
                           reference[scored]) * 100.0
            for s in seeds]))
        out["rates"].append(row)
    return out


def format_gate_sweep(sweeps: list) -> str:
    order = ["all", "drop_worst_25", "drop_worst_50", "best_25", "random_25", "worst_25"]
    lines = ["ICP inside the anchor gaps, cm recovered against pure interpolation",
             "(positive means ICP helped; every arm scored on never-anchor rows)", ""]
    rates = sorted({r["hz"] for s in sweeps for r in s["rates"]}, reverse=True)
    lines.append(f"{'anchors':>9} {'interp':>8}  " + " ".join(f"{n:>14}" for n in order))
    for hz in rates:
        rows = [r for s in sweeps for r in s["rates"] if r["hz"] == hz]
        interp = np.median([r["interp_cm"] for r in rows])
        cells = []
        for name in order:
            g = np.array([r["gain_cm"][name] for r in rows])
            cells.append(f"{np.median(g):>+7.2f}/{g.mean():>+6.2f}")
        lines.append(f"{hz:>7.2f}Hz {interp:>8.2f}  " + " ".join(f"{c:>14}" for c in cells))
    lines += ["", "  each cell is median/mean over sessions — they disagree on purpose:",
              "  a gate that caps the tail moves the mean and leaves the median alone."]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traj-dir", default=str(fr.DEFAULT_TRAJ_DIR))
    ap.add_argument("--pi3-dir", default=str(fr.DEFAULT_PI3_DIR))
    ap.add_argument("--session", default=None)
    ap.add_argument("--anchor-sweep", action="store_true",
                    help="thin the anchors and report how the error grows — the "
                         "cheap way to price a higher still rate before recording one")
    ap.add_argument("--gate-sweep", action="store_true",
                    help="thin the anchors and gate ICP on depth conditioning, "
                         "against random and worst-quartile controls")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    traj_dir, pi3_dir = Path(a.traj_dir), Path(a.pi3_dir)
    names = ([a.session] if a.session else
             sorted(p.stem for p in pi3_dir.glob("*.npz")
                    if (traj_dir / p.name).exists()))
    if a.gate_sweep:
        sweeps = []
        for name in names:
            try:
                sweeps.append(gate_sweep(traj_dir / f"{name}.npz", pi3_dir / f"{name}.npz"))
            except Exception as exc:
                print(f"{name}: {type(exc).__name__}: {exc}")
        if not sweeps:
            print("nothing to sweep")
            return 1
        print(json.dumps(sweeps, indent=2) if a.json else format_gate_sweep(sweeps))
        return 0

    if a.anchor_sweep:
        sweeps = []
        for name in names:
            try:
                sweeps.append(anchor_sweep(traj_dir / f"{name}.npz",
                                           pi3_dir / f"{name}.npz"))
            except Exception as exc:
                print(f"{name}: {type(exc).__name__}: {exc}")
        if not sweeps:
            return 1
        factors = [arm["every"] for arm in sweeps[0]["arms"]]
        print("ATE (cm) of interpolated Pi3X anchors, scored on the depth frames")
        print("that are never anchors — so every column is measured on the same rows")
        header = "  ".join(f"{5.0/k:>5.2f}Hz" for k in factors)
        print(f"{'session':>8} {'rows':>6}  {header}")
        for s_ in sweeps:
            cells = "  ".join(f"{arm['ate_cm']:>7.2f}" for arm in s_["arms"])
            print(f"{s_['session']:>8} {s_['scored_rows']:>6}  {cells}")
        table = np.array([[arm["ate_cm"] for arm in s_["arms"]] for s_ in sweeps])
        print(f"{'median':>8} {'':>6}  " +
              "  ".join(f"{v:>7.2f}" for v in np.median(table, axis=0)))
        rel = table / table[:, [0]]
        print(f"{'vs 5 Hz':>8} {'':>6}  " +
              "  ".join(f"{v:>6.2f}x" for v in np.median(rel, axis=0)))
        return 0

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
          f"{'fusion':>8} {'gated':>8} {'used':>5} {'gated buys':>11}")
    for r in rows:
        g = r.get("gated_cm")
        print(f"{r['session']:>8} {r['between_frames']:>7} {r['icp_alone_cm']:>10.2f} "
              f"{r['pi3_only_cm']:>10.2f} {r['fusion_cm']:>8.2f} "
              + (f"{g:>8.2f} {r['gated_intervals']:>5} {r['gated_gain_cm']:>+11.2f}"
                 if g is not None else f"{'-':>8} {'-':>5} {'-':>11}"))

    if rows:
        gated = [r["gated_gain_cm"] for r in rows if "gated_gain_cm" in r]
        if gated:
            g = np.array(gated)
            print(f"\ngated ICP helps in {int((g > 0).sum())} of {len(g)} sessions, "
                  f"median {np.median(g):+.2f} cm")
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
